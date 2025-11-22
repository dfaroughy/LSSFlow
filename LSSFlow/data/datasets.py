import numpy as np
import h5py
import torch
from scipy.interpolate import interp1d
from collections import namedtuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class DataCoupling(Dataset):
    """ Creates a source-target coupling dataset for training/inference 
    """
    def __init__(self, target=None, source=None):

        self.attributes=[]

        self.target = target
        self.source = source
        self.length = 0

        if target is not None: 
            self.attributes.append('target')
            self.target = target
            self.length = len(self.target)

        if source is not None: 
            self.attributes.append('source')
            self.source = source
            self.length = len(self.source)
        
        self.databatch = namedtuple('databatch', self.attributes)

    def __getitem__(self, idx):
        return self.databatch(*[getattr(self, attr)[idx] for attr in self.attributes])

    def __len__(self):
        return self.length
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class UchuuVoxelizedData:
    def __init__(self, 
                 datafile='/pscratch/sd/d/dfarough/LargeScaleStructure/data/Uchuu1000-Pl18_z0p00_hlist_4.h5', 
                 box_size=36, 
                 voxel_size=10):

        print('INFO: fetching data from {} '.format(datafile))

        self.box_size = box_size
        self.voxel_size = voxel_size
        self.num_side = int(round(box_size / voxel_size)) 
        self.num_voxels = self.num_side ** 3   
        
        f_pos = h5py.File(datafile, 'r')
        pos = np.stack([np.asarray(f_pos['x']),
                        np.asarray(f_pos['y']),
                        np.asarray(f_pos['z'])]).T

        pos = torch.tensor(pos, dtype=torch.float32)
        mask = (pos[...,0] <= box_size) & (pos[...,1] <= box_size) & (pos[...,2] <= box_size)

        print('INFO: voxelizing {} points into {} voxels within box size {} Mpc'.format(mask.sum().item(), self.num_voxels,box_size))
        
        self.local_patches = self.build_voxel_grid(pos[mask])
        
    def sample(self, voxel_idx=0, normalize=False):
        start = self.offsets[voxel_idx]
        end = self.offsets[voxel_idx + 1]
        if normalize:
            return self.normalize_data(self.local_patches[start:end], voxel_idx) 
        return self.local_patches[start:end]

    def get_3D_coord(self, voxel_idx=0):
        z = voxel_idx // (self.num_side * self.num_side)
        y = (voxel_idx % (self.num_side * self.num_side)) // self.num_side
        x = voxel_idx % self.num_side
        return (x, y, z)

    def get_voxel_center(self, voxel_idx=0):
        x_idx, y_idx, z_idx = self.get_3D_coord(voxel_idx)
        center_x = (x_idx + 0.5) * self.voxel_size
        center_y = (y_idx + 0.5) * self.voxel_size
        center_z = (z_idx + 0.5) * self.voxel_size
        return (center_x, center_y, center_z)

    def normalize_data(self, points, voxel_idx=0):
        center_x, center_y, center_z = self.get_voxel_center(voxel_idx)
        shift = torch.tensor([center_x, center_y, center_z], dtype=points.dtype, device=points.device)
        scale = self.voxel_size / 2.0
        self.normalization_params = {'shift': shift, 'scale': scale}
        return (points - shift) / scale

    def build_voxel_grid(self, points):
        """
        points: (N, 3) tensor with coordinates in [0, box_size) in each dim.
        Returns:
            points_sorted : (N, 3) tensor, points sorted by voxel id
            offsets       : (n_voxels + 1,) long tensor; slice [offsets[v]:offsets[v+1]] gives
                            the rows in points_sorted that belong to voxel v
            n_side        : number of voxels per side (should be 50)
        """

        device = points.device
                  
        # Create bin boundaries for each dimension
        boundaries = torch.arange(self.num_side + 1, dtype=points.dtype, device=device) * self.voxel_size
        ix = torch.bucketize(points[:, 0], boundaries, right=False) - 1
        iy = torch.bucketize(points[:, 1], boundaries, right=False) - 1
        iz = torch.bucketize(points[:, 2], boundaries, right=False) - 1
        
        # Clamp to handle any points exactly at box_size 
        ix = ix.clamp(0, self.num_side - 1)
        iy = iy.clamp(0, self.num_side - 1) 
        iz = iz.clamp(0, self.num_side - 1)

        # Flatten 3D voxel index (ix, iy, iz) -> voxel_id
        voxel_ids = ix + self.num_side * (iy + self.num_side * iz)  # (N,)

        # Count how many points per voxel
        counts = torch.bincount(voxel_ids, minlength=self.num_voxels)

        # offsets[v]   = start index of voxel v in the sorted points array
        # offsets[v+1] = end index (exclusive)
        self.offsets = torch.zeros(self.num_voxels + 1, dtype=torch.long, device=device)
        self.offsets[1:] = torch.cumsum(counts, dim=0)

        # Sort points by voxel id so each voxel occupies a contiguous block
        order = torch.argsort(voxel_ids)
        points_sorted = points[order]

        return points_sorted

    def display(self, voxel_idx, normalize=False, figsize=(12, 6),  dpi=120, **kwargs):
        data = self.sample(voxel_idx, normalize).cpu().numpy()
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(data[:,0], data[:,1], data[:,2], **kwargs)
        plt.show()


class Uniform3D:
    def __init__(self, support='cube'):
        self.support = support
        
    def sample(self, num_points=10_000, device='cpu'):
        if self.support == 'ball':
            return self.uniform_ball(num_points).to(device)
        elif self.support == 'cube':
            return torch.distributions.uniform.Uniform(low=-1, high=1).sample((num_points,3)).to(device)
            
    def uniform_ball(self, num_points=10_000, R=1):
        m_costheta = torch.distributions.uniform.Uniform(low=-1, high=1)
        m_phi = torch.distributions.uniform.Uniform(low=0, high=2*np.pi)
        radius = torch.ones(num_points) * R
        costheta = m_costheta.sample((num_points,))
        sintheta = torch.sqrt(1 - costheta**2)
        phi = m_phi.sample((num_points,))
        x = radius * sintheta * torch.cos(phi)
        y = radius * sintheta * torch.sin(phi)
        z = radius * costheta
        pos = torch.stack([x,y,z]).T
        return pos.to(torch.float32)



class UchuuCentered:
    def __init__(self, datafile='Uchuu1000-Pl18_z0p00_hlist_4.h5', radius=36):
        f_pos = h5py.File(datafile, 'r')
        pos = np.stack([np.asarray(f_pos['x']),
                        np.asarray(f_pos['y']),
                        np.asarray(f_pos['z'])]).T
        origin = np.array([500,500,500])
        self.pos = pos - origin
        self.radius = radius

    def sample(self):

        distance = np.sqrt(np.sum((self.pos)**2, axis=1))
        phi = np.arctan2(self.pos[:,1],self.pos[:,0])
        theta = np.arccos(self.pos[:,2] / distance)

        mask = distance < self.radius
        masked_distance = distance[mask]
        masked_theta = theta[mask]
        masked_phi = phi[mask]

        masked_x = masked_distance * np.sin(masked_theta) * np.cos(masked_phi)
        masked_y = masked_distance * np.sin(masked_theta) * np.sin(masked_phi)
        masked_z = masked_distance * np.cos(masked_theta)

        masked_centered_pos = np.stack([masked_x.T,masked_y.T,masked_z.T]).T

        masked_distance = np.sqrt(np.sum((masked_centered_pos)**2, axis=1))
        max_masked_distance = np.max(masked_distance)
        masked_normed_centered_pos = masked_centered_pos / max_masked_distance
        return torch.tensor(masked_normed_centered_pos).to(torch.float32)


def inverse_sample_decorator(dist):
    def wrapper(pnts, x_min=-1, x_max=1, n=1e5, **kwargs):
        x = np.linspace(x_min, x_max, int(n))
        cumulative = np.cumsum(dist(x, **kwargs))
        cumulative -= cumulative.min()
        f = interp1d(cumulative/cumulative.max(), x)
        return f(np.random.random(pnts))
    return wrapper


@inverse_sample_decorator
def rsquared(x, amp=1.0):
    return amp*x*x

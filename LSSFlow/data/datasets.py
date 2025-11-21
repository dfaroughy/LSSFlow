import numpy as np
import h5py
import torch
from scipy.interpolate import interp1d
from collections import namedtuple
from torch.utils.data import Dataset


class DataCoupling(Dataset):

    """ Creates a source-target coupling dataset for training/inference 
    """
    def __init__(self, source, target=None):

        self.attributes=['source']
        self.source = source

        if target is not None: 
            self.attributes.append('target')
            self.target = target
        
        self.databatch = namedtuple('databatch', self.attributes)

    def __getitem__(self, idx):
        return self.databatch(*[getattr(self, attr)[idx] for attr in self.attributes])

    def __len__(self):
        return len(self.source)
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


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


class SphericalUniform:
    def __init__(self, radius=1):
        self.radius = radius

    def sample(self, num_points=10_000):

        m_costheta = torch.distributions.uniform.Uniform(low=-1, high=1)
        m_phi = torch.distributions.uniform.Uniform(low=0, high=2*np.pi)
        self.radius = torch.from_numpy(np.array(rsquared(num_points, x_min=0, x_max=1)))
        costheta = m_costheta.sample((num_points,))
        sintheta = torch.sqrt(1 - costheta**2)
        phi = m_phi.sample((num_points,))

        #...cartesian coordinates

        x = self.radius * sintheta * torch.cos(phi)
        y = self.radius * sintheta * torch.sin(phi)
        z = self.radius * costheta
        pos = torch.stack([x,y,z]).T

        return pos.to(torch.float32)


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

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


class ToyLargeScaleStructure:
    def __init__(self, box_size=100.0, num_clusters=12, filament_density=0.5, void_fraction=0.3):
        """
        Toy model of Large Scale Structure with clusters, filaments, and voids.
        
        Args:
            box_size: Size of the simulation box
            num_clusters: Number of galaxy clusters to generate
            filament_density: Density parameter for filamentary structures
            void_fraction: Fraction of volume that should be voids (low density)
        """
        self.box_size = box_size
        self.num_clusters = num_clusters
        self.filament_density = filament_density
        self.void_fraction = void_fraction
        
        # Pre-generate cluster centers in a more structured way to ensure connectivity
        self.cluster_centers = self._generate_structured_clusters()
        self.cluster_masses = torch.exp(torch.randn(num_clusters) * 0.3 + 1.5)  # More uniform masses
    
    def _generate_structured_clusters(self):
        """Generate cluster centers that are more likely to form connected filaments"""
        centers = []
        
        # Create a basic grid structure with some randomness
        grid_size = int(np.ceil(self.num_clusters ** (1/3)))
        spacing = self.box_size / (grid_size + 1)
        
        for i in range(self.num_clusters):
            # Grid position
            x_grid = (i % grid_size) + 1
            y_grid = ((i // grid_size) % grid_size) + 1
            z_grid = (i // (grid_size * grid_size)) + 1
            
            # Add randomness but keep structure
            x = x_grid * spacing + torch.randn(1) * spacing * 0.3
            y = y_grid * spacing + torch.randn(1) * spacing * 0.3
            z = z_grid * spacing + torch.randn(1) * spacing * 0.3
            
            # Clamp to box boundaries
            x = torch.clamp(x, spacing * 0.5, self.box_size - spacing * 0.5)
            y = torch.clamp(y, spacing * 0.5, self.box_size - spacing * 0.5)
            z = torch.clamp(z, spacing * 0.5, self.box_size - spacing * 0.5)
            
            centers.append(torch.tensor([x, y, z]).squeeze())
        
        return torch.stack(centers)

    def sample(self, num_points=10_000):
        """
        Generate points following a toy LSS distribution with:
        - High density clusters (halos)
        - Connecting filaments
        - Large voids with very low density
        """
        points = []
        
        # Calculate how many points go to each component
        cluster_fraction = 0.25  # Reduced cluster fraction
        filament_fraction = 0.55  # Increased filament fraction significantly
        
        n_cluster = int(num_points * cluster_fraction)
        n_filament = int(num_points * filament_fraction)
        n_void = num_points - n_cluster - n_filament
        
        # 1. Generate cluster points (high density regions)
        if n_cluster > 0:
            cluster_points = self._generate_cluster_points(n_cluster)
            points.append(cluster_points)
        
        # 2. Generate filament points (connecting structures)
        if n_filament > 0:
            filament_points = self._generate_filament_points(n_filament)
            points.append(filament_points)
        
        # 3. Generate void points (low density regions)
        if n_void > 0:
            void_points = self._generate_void_points(n_void)
            points.append(void_points)
        
        # Combine all points
        all_points = torch.cat(points, dim=0)
        
        # Shuffle to remove any ordering bias
        perm = torch.randperm(all_points.shape[0])
        all_points = all_points[perm]
        
        # Normalize to [-1, 1] range
        all_points = (all_points / self.box_size) * 2 - 1
        
        return all_points.to(torch.float32)
    
    def _generate_cluster_points(self, n_points):
        """Generate points around cluster centers with NFW-like profile"""
        points = []
        points_per_cluster = torch.multinomial(self.cluster_masses, n_points, replacement=True)
        
        for i in range(self.num_clusters):
            n_cluster_points = (points_per_cluster == i).sum().item()
            if n_cluster_points == 0:
                continue
                
            center = self.cluster_centers[i]
            
            # Generate radial distances with NFW-like profile (simplified)
            u = torch.rand(n_cluster_points)
            # Approximate NFW profile: denser at center, falls off as r^-2
            r = torch.sqrt(-torch.log(1 - u * 0.99)) * 3.0
            
            # Random directions
            theta = torch.acos(2 * torch.rand(n_cluster_points) - 1)
            phi = 2 * np.pi * torch.rand(n_cluster_points)
            
            # Convert to Cartesian
            x = center[0] + r * torch.sin(theta) * torch.cos(phi)
            y = center[1] + r * torch.sin(theta) * torch.sin(phi)
            z = center[2] + r * torch.cos(theta)
            
            cluster_points = torch.stack([x, y, z], dim=1)
            
            # Keep points within box
            cluster_points = torch.clamp(cluster_points, 0, self.box_size)
            points.append(cluster_points)
        
        return torch.cat(points, dim=0) if points else torch.empty(0, 3)
    
    def _generate_filament_points(self, n_points):
        """Generate points along filaments connecting clusters"""
        if self.num_clusters < 2:
            # Fall back to random points if not enough clusters
            return torch.rand(n_points, 3) * self.box_size
        
        points = []
        total_filament_points = 0
        
        # Create a network of filaments connecting clusters
        # Connect each cluster to its nearest neighbors
        for i in range(self.num_clusters):
            # Calculate distances to other clusters
            distances = []
            for j in range(self.num_clusters):
                if i != j:
                    dist = torch.norm(self.cluster_centers[i] - self.cluster_centers[j])
                    distances.append((dist, j))
            
            # Sort by distance and connect to 2-3 nearest neighbors
            distances.sort()
            max_connections = min(3, len(distances))
            
            for k in range(max_connections):
                dist, j = distances[k]
                
                # Skip if we've already created this connection
                if j < i:  # Only create each connection once
                    continue
                
                # Number of points for this filament based on distance and mass
                connection_strength = (self.cluster_masses[i] * self.cluster_masses[j]) / (dist + 1e-6)
                filament_points_count = int(n_points * connection_strength / torch.sum(self.cluster_masses) * 0.1)
                filament_points_count = min(filament_points_count, n_points // 4)  # Limit per filament
                
                if filament_points_count > 0 and total_filament_points < n_points:
                    actual_points = min(filament_points_count, n_points - total_filament_points)
                    filament_segment = self._points_along_filament(
                        self.cluster_centers[i], 
                        self.cluster_centers[j], 
                        actual_points
                    )
                    points.append(filament_segment)
                    total_filament_points += actual_points
        
        # Create additional web-like structure
        remaining_points = n_points - total_filament_points
        if remaining_points > 0:
            web_points = self._generate_web_structure(remaining_points)
            points.append(web_points)
        
        return torch.cat(points, dim=0) if points else torch.rand(n_points, 3) * self.box_size
    
    def _points_along_filament(self, start, end, n_points):
        """Generate points along a filament between two cluster centers"""
        # Parameter along the line
        t = torch.linspace(0, 1, n_points)
        
        # Add some curvature to make filaments more realistic
        # Create a slightly curved path instead of straight line
        mid_point = (start + end) / 2
        
        # Add a random perpendicular offset to create curvature
        direction = end - start
        direction_norm = direction / torch.norm(direction)
        
        # Create perpendicular vector for curvature
        if abs(direction_norm[0]) < 0.9:
            perp = torch.tensor([1.0, 0.0, 0.0])
        else:
            perp = torch.tensor([0.0, 1.0, 0.0])
        
        perp = perp - torch.dot(perp, direction_norm) * direction_norm
        perp = perp / torch.norm(perp)
        
        # Create curved path
        curve_strength = torch.norm(end - start) * 0.1  # 10% of distance
        curve_offset = perp * curve_strength * torch.randn(1).item()
        
        # Quadratic bezier curve: P(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        # where P1 is the control point (mid_point + offset)
        control_point = mid_point + curve_offset
        
        curve_points = []
        for t_val in t:
            point = (1 - t_val)**2 * start + 2 * (1 - t_val) * t_val * control_point + t_val**2 * end
            curve_points.append(point)
        
        curve_points = torch.stack(curve_points)
        
        # Add perpendicular scatter around the curved filament
        direction = end - start
        direction = direction / torch.norm(direction)
        
        # Create two perpendicular vectors
        if abs(direction[0]) < 0.9:
            perp1 = torch.tensor([1.0, 0.0, 0.0])
        else:
            perp1 = torch.tensor([0.0, 1.0, 0.0])
        
        perp1 = perp1 - torch.dot(perp1, direction) * direction
        perp1 = perp1 / torch.norm(perp1)
        perp2 = torch.cross(direction, perp1)
        
        # Add scatter perpendicular to filament
        filament_width = 1.5  # Filament thickness
        scatter1 = torch.randn(n_points) * filament_width
        scatter2 = torch.randn(n_points) * filament_width
        
        scattered_points = (curve_points + 
                          scatter1.unsqueeze(1) * perp1.unsqueeze(0) + 
                          scatter2.unsqueeze(1) * perp2.unsqueeze(0))
        
        # Keep within bounds
        scattered_points = torch.clamp(scattered_points, 0, self.box_size)
        
        return scattered_points
    
    def _generate_web_structure(self, n_points):
        """Generate additional web-like filamentary structure"""
        points = []
        
        # Create random interconnected segments that form a web
        n_segments = max(5, n_points // 50)  # More segments for better connectivity
        
        # Use existing cluster centers as anchor points, but also create intermediate nodes
        anchor_points = self.cluster_centers.clone()
        
        # Add some intermediate anchor points between clusters
        for i in range(self.num_clusters):
            for j in range(i + 1, min(i + 3, self.num_clusters)):  # Connect to nearby clusters
                # Add intermediate point
                intermediate = (self.cluster_centers[i] + self.cluster_centers[j]) / 2
                intermediate += torch.randn(3) * 5  # Add some randomness
                intermediate = torch.clamp(intermediate, 0, self.box_size)
                anchor_points = torch.cat([anchor_points, intermediate.unsqueeze(0)], dim=0)
        
        points_per_segment = n_points // n_segments
        
        for i in range(n_segments):
            if len(points) * points_per_segment >= n_points:
                break
                
            # Pick two random anchor points
            idx1, idx2 = torch.randperm(len(anchor_points))[:2]
            start = anchor_points[idx1]
            end = anchor_points[idx2]
            
            # Create segment
            if points_per_segment > 0:
                segment_points = self._points_along_filament(start, end, points_per_segment)
                points.append(segment_points)
        
        # Fill remaining with random points that follow filamentary structure
        current_count = sum(p.shape[0] for p in points) if points else 0
        remaining = n_points - current_count
        
        if remaining > 0:
            # Create points along random directions from cluster centers
            random_points = []
            for _ in range(remaining):
                # Pick random cluster center
                center_idx = torch.randint(0, self.num_clusters, (1,)).item()
                center = self.cluster_centers[center_idx]
                
                # Random direction
                direction = torch.randn(3)
                direction = direction / torch.norm(direction)
                
                # Random distance (favoring closer to center)
                exp_dist = torch.distributions.Exponential(torch.tensor(1.0))
                distance = exp_dist.sample() * 8.0
                
                point = center + direction * distance
                point = torch.clamp(point, 0, self.box_size)
                random_points.append(point)
            
            if random_points:
                points.append(torch.stack(random_points))
        
        return torch.cat(points, dim=0) if points else torch.rand(n_points, 3) * self.box_size
    
    def _generate_void_points(self, n_points):
        """Generate points in void regions (avoiding clusters and filaments)"""
        points = []
        attempts = 0
        max_attempts = n_points * 5  # Reduced attempts since we have fewer void points
        
        while len(points) < n_points and attempts < max_attempts:
            # Generate candidate points
            batch_size = min(n_points - len(points), 1000)
            candidates = torch.rand(batch_size, 3) * self.box_size
            
            # Check distance to all clusters
            valid_mask = torch.ones(candidates.shape[0], dtype=torch.bool)
            
            for center in self.cluster_centers:
                distances = torch.norm(candidates - center.unsqueeze(0), dim=1)
                # Reject points too close to clusters (create voids)
                valid_mask &= distances > 12.0  # Larger void regions
            
            valid_points = candidates[valid_mask]
            if valid_points.shape[0] > 0:
                points.append(valid_points)
            
            attempts += 1
        
        if points:
            all_void_points = torch.cat(points, dim=0)
            # Take only what we need
            if all_void_points.shape[0] > n_points:
                indices = torch.randperm(all_void_points.shape[0])[:n_points]
                all_void_points = all_void_points[indices]
            elif all_void_points.shape[0] < n_points:
                # Fill remaining with very sparse random points
                remaining = n_points - all_void_points.shape[0]
                extra_points = torch.rand(remaining, 3) * self.box_size
                all_void_points = torch.cat([all_void_points, extra_points], dim=0)
            return all_void_points
        else:
            # Fallback: return sparse random points
            return torch.rand(n_points, 3) * self.box_size



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

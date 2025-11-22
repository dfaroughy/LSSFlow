
import numpy as np
import torch

def build_voxel_grid(points, box_size=1000.0, voxel_size=100.0):
    """
    points: (N, 3) tensor with coordinates in [0, box_size) in each dim.
    Returns:
        points_sorted : (N, 3) tensor, points sorted by voxel id
        offsets       : (n_voxels + 1,) long tensor; slice [offsets[v]:offsets[v+1]] gives
                        the rows in points_sorted that belong to voxel v
        n_side        : number of voxels per side (should be 50)
    """
    device = points.device
    n_side = int(round(box_size / voxel_size))   # 50
    n_voxels = n_side ** 3                       # 125000

    # Create bin boundaries for each dimension using torch.bucketize
    # boundaries[i] defines the left edge of bin i, with an extra boundary at the end
    boundaries = torch.arange(n_side + 1, dtype=points.dtype, device=device) * voxel_size
    
    # Use bucketize to assign points to bins - this handles boundary cases properly
    # bucketize returns bin indices in [0, n_side], we subtract 1 to get [0, n_side-1]
    # Points exactly at boundaries go to the left bin (right=False)
    ix = torch.bucketize(points[:, 0], boundaries, right=False) - 1
    iy = torch.bucketize(points[:, 1], boundaries, right=False) - 1
    iz = torch.bucketize(points[:, 2], boundaries, right=False) - 1
    
    # Clamp to handle any points exactly at box_size (should go to last bin)
    ix = ix.clamp(0, n_side - 1)
    iy = iy.clamp(0, n_side - 1) 
    iz = iz.clamp(0, n_side - 1)

    # Flatten 3D voxel index (ix, iy, iz) -> single integer voxel_id in [0, n_voxels)
    voxel_ids = ix + n_side * (iy + n_side * iz)  # (N,)

    # Count how many points per voxel
    counts = torch.bincount(voxel_ids, minlength=n_voxels)

    # Prefix sum of counts -> offsets array
    # offsets[v]   = start index of voxel v in the sorted points array
    # offsets[v+1] = end index (exclusive)
    offsets = torch.zeros(n_voxels + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    # Sort points by voxel id so each voxel occupies a contiguous block
    order = torch.argsort(voxel_ids)
    points_sorted = points[order]

    return points_sorted, offsets, n_side


def get_voxel_points(points_sorted, offsets, voxel_id):
    """
    Return all points inside voxel with flat index voxel_id.
    """
    start = offsets[voxel_id]
    end = offsets[voxel_id + 1]
    return points_sorted[start:end] 


def compute_power_spectrum_cube(points, box_size, grid_size):
    """
    Compute an approximate 3D power spectrum P(k) from a point cloud
    living inside a cubic box with periodic boundary conditions.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Galaxy positions in R^3, assumed to be in [-box_size/2, box_size/2) for each dimension.
    box_size : float
        Physical size of the cubic box.
    grid_size : int
        Number of grid cells per dimension for the voxel grid (e.g. 128 or 256).

    Returns
    -------
    k_vals : 1D ndarray
        Bin centers in k-space.
    Pk : 1D ndarray
        Spherically averaged power spectrum P(k) for each bin.
    """

    points = np.asarray(points, dtype=np.float32)

    # Shift points from [-box_size/2, box_size/2] to [0, box_size]
    points = points + box_size / 2.0

    # Grid spacing
    dx = box_size / grid_size

    # Cloud-in-cell assignment of points to a 3D grid
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    coords = points / dx  # convert to grid coordinates in [0, grid_size)

    for x, y, z in coords:
        i = int(np.floor(x)) % grid_size
        j = int(np.floor(y)) % grid_size
        k = int(np.floor(z)) % grid_size

        dx1 = x - i
        dy1 = y - j
        dz1 = z - k

        dx0 = 1.0 - dx1
        dy0 = 1.0 - dy1
        dz0 = 1.0 - dz1

        # Distribute mass to 8 neighbors (periodic boundary conditions)
        density[i,                   j,                   k                  ] += dx0 * dy0 * dz0
        density[(i + 1) % grid_size, j,                   k                  ] += dx1 * dy0 * dz0
        density[i,                   (j + 1) % grid_size, k                  ] += dx0 * dy1 * dz0
        density[i,                   j,                   (k + 1) % grid_size] += dx0 * dy0 * dz1
        density[(i + 1) % grid_size, (j + 1) % grid_size, k                  ] += dx1 * dy1 * dz0
        density[(i + 1) % grid_size, j,                   (k + 1) % grid_size] += dx1 * dy0 * dz1
        density[i,                   (j + 1) % grid_size, (k + 1) % grid_size] += dx0 * dy1 * dz1
        density[(i + 1) % grid_size, (j + 1) % grid_size, (k + 1) % grid_size] += dx1 * dy1 * dz1

    # Compute overdensity field δ = ρ / <ρ> - 1
    density_mean = np.mean(density)
    
    if density_mean == 0:
        raise ValueError("Mean density is zero; check input points / grid_size.")

    delta = density / density_mean - 1.0

    # FFT and power spectrum
    delta_k = np.fft.fftn(delta)
    delta_k = np.fft.fftshift(delta_k)
    power = np.abs(delta_k) ** 2

    # Construct k-grid and bin isotropically
    kfreq = np.fft.fftfreq(grid_size, d=dx)
    kfreq = np.fft.fftshift(kfreq)

    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    kk = np.sqrt(kx**2 + ky**2 + kz**2)

    k_flat = kk.ravel()
    p_flat = power.ravel()

    # Ignore k=0 mode
    nonzero = k_flat > 0
    k_flat = k_flat[nonzero]
    p_flat = p_flat[nonzero]

    # Bin in k (log-space)
    kmin = k_flat.min()
    kmax = k_flat.max()
    nbins = grid_size // 2

    k_edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    bin_idx = np.digitize(k_flat, k_edges)

    k_vals = np.zeros(nbins)
    Pk = np.zeros(nbins)

    for b in range(1, nbins + 1):
        mask_b = bin_idx == b
        if np.any(mask_b):
            k_vals[b - 1] = np.mean(k_flat[mask_b])
            Pk[b - 1] = np.mean(p_flat[mask_b])
        else:
            k_vals[b - 1] = np.nan
            Pk[b - 1] = np.nan

    valid = ~np.isnan(Pk)
    return k_vals[valid], Pk[valid]


def compute_power_spectrum_sphere(points, grid_size):
    """
    Compute an approximate 3D power spectrum P(k) from a point cloud
    living inside a unit sphere (radius 1, centered at origin).

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Galaxy positions in R^3, assumed to satisfy ||x|| <= 1 after rescaling.
    grid_size : int
        Number of grid cells per dimension for the voxel grid (e.g. 128 or 256).

    Returns
    -------
    k_vals : 1D ndarray
        Bin centers in k-space.
    Pk : 1D ndarray
        Spherically averaged power spectrum P(k) for each bin.
    """

    points = np.asarray(points, dtype=np.float32)

    r = np.linalg.norm(points, axis=1)
    mask = r <= 1.0
    points = points[mask]

    #...Embed sphere in a periodic cube of size L=2 ([-1,1]^3).

    dx = 2.0 / grid_size
    points_box = points + 1.0  # shift [-1,1] -> [0,2]

    #...cloud-in-cell assignment of points to a 3D grid

    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    coords = points_box / dx # convert to grid coordinates in [0, grid_size)

    for x, y, z in coords:
        i = int(np.floor(x)) % grid_size
        j = int(np.floor(y)) % grid_size
        k = int(np.floor(z)) % grid_size

        dx1 = x - i
        dy1 = y - j
        dz1 = z - k

        dx0 = 1.0 - dx1
        dy0 = 1.0 - dy1
        dz0 = 1.0 - dz1

        # Distribute mass to 8 neighbors (periodic)
        density[i,                 j,                 k                ] += dx0 * dy0 * dz0
        density[(i + 1) % grid_size, j,               k                ] += dx1 * dy0 * dz0
        density[i,                 (j + 1) % grid_size, k              ] += dx0 * dy1 * dz0
        density[i,                 j,                 (k + 1) % grid_size] += dx0 * dy0 * dz1
        density[(i + 1) % grid_size, (j + 1) % grid_size, k           ] += dx1 * dy1 * dz0
        density[(i + 1) % grid_size, j,               (k + 1) % grid_size] += dx1 * dy0 * dz1
        density[i,                 (j + 1) % grid_size, (k + 1) % grid_size] += dx0 * dy1 * dz1
        density[(i + 1) % grid_size, (j + 1) % grid_size, (k + 1) % grid_size] += dx1 * dy1 * dz1

    #...overdensity field δ = ρ / <ρ> - 1

    density_mean = np.mean(density)
    
    if density_mean == 0:
        raise ValueError("Mean density is zero; check input points / grid_size.")

    delta = density / density_mean - 1.0

    #...FFT and power spectrum

    delta_k = np.fft.fftn(delta)
    delta_k = np.fft.fftshift(delta_k)
    power = np.abs(delta_k) ** 2

    #...Construct k-grid and bin isotropically
    # Note: np.fft.fftfreq uses spacing dx; k in units of 1/length

    kfreq = np.fft.fftfreq(grid_size, d=dx)
    kfreq = np.fft.fftshift(kfreq)

    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    kk = np.sqrt(kx**2 + ky**2 + kz**2)

    k_flat = kk.ravel()
    p_flat = power.ravel()

    # Ignore k=0 mode
    nonzero = k_flat > 0
    k_flat = k_flat[nonzero]
    p_flat = p_flat[nonzero]

    # Bin in k (log-space is convenient)
    kmin = k_flat.min()
    kmax = k_flat.max()
    nbins = grid_size // 2

    k_edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    bin_idx = np.digitize(k_flat, k_edges)

    k_vals = np.zeros(nbins)
    Pk = np.zeros(nbins)

    for b in range(1, nbins + 1):
        mask_b = bin_idx == b
        if np.any(mask_b):
            k_vals[b - 1] = np.mean(k_flat[mask_b])
            Pk[b - 1] = np.mean(p_flat[mask_b])
        else:
            k_vals[b - 1] = np.nan
            Pk[b - 1] = np.nan

    valid = ~np.isnan(Pk)
    return k_vals[valid], Pk[valid]


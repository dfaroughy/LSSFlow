import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
mpl.rc('axes',edgecolor='k')
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.color'] = 'k'
plt.rcParams['ytick.color'] = 'k'
plt.rcParams['text.usetex'] = False


def plot_lss_results(truth, target, figsize=(12, 6),  dpi=120, marker_size=0.01, colors=['k','r'],, marker=',', alpha=1.0, save_fig=None, orientation=(30, -60), color_by_density=False, cmap='magma'):

    truth = truth.cpu().numpy()
    target = target.cpu().numpy()
    data_list = [truth, target]
    elev, azim = orientation

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        
        c = colors[i]
        ax.scatter3D(data[:,0], data[:,1], data[:,2], s=marker_size, c=c, marker=marker, alpha=alpha)
        
        # Draw full cube
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        # Construct corners
        corners = np.array([
            [xlim[0], ylim[0], zlim[0]],
            [xlim[1], ylim[0], zlim[0]],
            [xlim[1], ylim[1], zlim[0]],
            [xlim[0], ylim[1], zlim[0]],
            [xlim[0], ylim[0], zlim[1]],
            [xlim[1], ylim[0], zlim[1]],
            [xlim[1], ylim[1], zlim[1]],
            [xlim[0], ylim[1], zlim[1]]
        ])
        
        # Edges
        edges = [
            (0,1), (1,2), (2,3), (3,0), # Bottom
            (4,5), (5,6), (6,7), (7,4), # Top
            (0,4), (1,5), (2,6), (3,7)  # Vertical
        ]
        
        for start, end in edges:
            ax.plot3D(
                [corners[start,0], corners[end,0]],
                [corners[start,1], corners[end,1]],
                [corners[start,2], corners[end,2]],
                'k--', linewidth=0.5
            )
        
        ax.set_axis_off()

    if save_fig is not None:
        plt.savefig(save_fig, dpi=dpi)
        
    plt.show()
    plt.close()


def plot_lss_slices(truth, target,  threshold, figsize=(12, 12), dpi=120, marker_size=0.01, colors=['k','r'], marker='o', alpha=1.0, save_fig=None, color_by_density=False, cmap='magma'):
    truth = truth.cpu().numpy()
    target = target.cpu().numpy()
    
    fig = plt.figure(figsize=figsize, dpi=dpi)

    dims = [(0, 1, 2), (1, 2, 0), (0, 2, 1)]
    
    for i, (xi, xj, xk) in enumerate(dims):
        # Truth
        zmask = np.abs(truth[:,xk])<threshold
        data = truth[zmask]
        c = colors[0]

        ax = fig.add_subplot(3, 2, 2*i + 1)
        ax.scatter(data[:,xi], data[:,xj], s=marker_size, c=c, marker=marker, alpha=alpha)
        ax.set_aspect('equal')
        ax.grid(False)
        
        # Target
        zmask = np.abs(target[:,xk])<threshold
        data = target[zmask]
        c = colors[1]
        ax = fig.add_subplot(3, 2, 2*i + 2)
        ax.scatter(data[:,xi], data[:,xj], s=marker_size, c=c, marker=marker, alpha=alpha)
        ax.set_aspect('equal')
        ax.grid(False)

    if save_fig is not None:
        plt.savefig(save_fig, dpi=dpi)



def plot_power_spectrum_sphere(k, Pk_truth, Pk_target, figsize=(8,6)):

    if isinstance(k, torch.Tensor):
        k = k.cpu().numpy()

    if isinstance(Pk_truth, torch.Tensor):
        Pk_truth = Pk_truth.cpu().numpy()

    if isinstance(Pk_target, torch.Tensor):
        Pk_target = Pk_target.cpu().numpy()

    fig, _ = plt.subplots(figsize=figsize)

    # --- top panel ---
    ax1 = plt.subplot2grid((4,1), (0,0), rowspan=2)
    ax1.loglog(k, Pk_truth, label='truth', c='b', lw=1.)
    ax1.loglog(k, Pk_target, label='target', c='r', lw=1.)
    ax1.set_ylabel(r"$P(k)$")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- bottom panel ---
    ax2 = plt.subplot2grid((4,1), (2,0), rowspan=1, sharex=ax1)
    ax2.scatter(k, Pk_target / Pk_truth, c='k', s=5)
    ax2.axhline(1.0, color='k', lw=1., ls='--')
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"Ratio")
    ax2.grid(alpha=0.3)

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()
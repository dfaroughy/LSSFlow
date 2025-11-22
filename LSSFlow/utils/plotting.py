import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

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

def plot_lss_results(truth, target, figsize=(12, 6),  dpi=120, marker_size=0.01, colors=['b','r'], marker='o', alpha=1.0):

    truth = truth.cpu().numpy()
    target = target.cpu().numpy()

    fig = plt.figure(figsize=figsize, dpi=120)

    ax1 = fig.add_subplot(121,projection='3d')
    ax1.scatter3D(truth[:,0], truth[:,1], truth[:,2], s=marker_size, c=colors[0], marker=marker, alpha=alpha)
    
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.scatter3D(target[:,0], target[:,1], target[:,2], s=marker_size, c=colors[1], marker=marker, alpha=alpha)


def plot_lss_slices(truth, target, figsize=(12, 12), dpi=120, marker_size=0.01, colors=['b','r'], marker='o', alpha=1.0):

    truth = truth.cpu().numpy()
    target = target.cpu().numpy()
    fig = plt.figure(figsize=figsize, dpi=dpi)

    #------------------------------------------------------------------
    xi, xj, xk = 0, 1, 2
    zmask = np.abs(truth[:,xk])<0.07
    trajectories_zmask = np.abs(target[:,xk])<0.07
    ax = fig.add_subplot(321)
    ax.scatter(truth[zmask,xi],truth[zmask,xj],s=marker_size, c=colors[0], marker=marker, alpha=alpha)
    ax.set_aspect('equal')
    ax = fig.add_subplot(322)
    ax.scatter(target[trajectories_zmask,xi],target[trajectories_zmask,xj], s=marker_size, c=colors[1], marker=marker, alpha=alpha)
    ax.set_aspect('equal')

    #------------------------------------------------------------------
    xi, xj, xk = 1, 2, 0
    zmask = np.abs(truth[:,xk])<0.07
    trajectories_zmask = np.abs(target[:,xk])<0.07
    ax = fig.add_subplot(323)
    ax.scatter(truth[zmask,xi],truth[zmask,xj],s=marker_size, c=colors[0], marker=marker, alpha=alpha)
    ax.set_aspect('equal')  
    ax = fig.add_subplot(324)
    ax.scatter(target[trajectories_zmask,xi],target[trajectories_zmask,xj], s=marker_size, c=colors[1], marker=marker, alpha=alpha)
    ax.set_aspect('equal')      

    #------------------------------------------------------------------
    xi, xj, xk = 0, 2, 1
    zmask = np.abs(truth[:,xk])<0.07
    trajectories_zmask = np.abs(target[:,xk])<0.07
    ax = fig.add_subplot(325)
    ax.scatter(truth[zmask,xi],truth[zmask,xj],s=marker_size, c=colors[0], marker=marker, alpha=alpha)
    ax.set_aspect('equal')  
    ax = fig.add_subplot(326)
    ax.scatter(target[trajectories_zmask,xi],target[trajectories_zmask,xj], s=marker_size, c=colors[1], marker=marker, alpha=alpha)
    ax.set_aspect('equal')  


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
import yaml
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, random_split

from datasets import UchuuCentered, DataCoupling

def build_dataloaders(source, target, batch_size, train_split=1.0):
    target_data = target.sample()
    source_data = source.sample(num_points=len(target_data))
    dataset = DataCoupling(source_data, target_data)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_size == 0:
        return train, None
    val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train, val


def plot_lss_results(trajectories):

    trajectories = trajectories.cpu().numpy()

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
    plt.rcParams['text.usetex'] = True
    
    fig = plt.figure(figsize=(12, 6), dpi=200)
    lss = UchuuCentered(datafile='/home/df630/LargeScaleStructureFlows/data/Uchuu1000-Pl18_z0p00_hlist_4.h5', radius=36)
    truth = lss.sample()
    truth = truth.cpu().numpy()
    fig = plt.figure(figsize=(12, 6), dpi= 120)
    ax = fig.add_subplot(121,projection='3d')
    ax.scatter3D(truth[:,0],truth[:,1],truth[:,2],s=0.01,c='b')
    ax = fig.add_subplot(122,projection='3d')
    ax.scatter3D(trajectories[-1,:100000,0],trajectories[-1,:100000,1],trajectories[-1,:100000,2],s=0.01,c='r')



@rank_zero_only
def set_logger():
    logger = CometLogger(api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz',
                         project='LargeScaleStructure',
                         workspace='dfaroughy',
                         offline_directory='/home/df630/LargeScaleStructureFlows/Results'
                        )

    if config.experiment_id is None: # if new experiment

        config_dict = {k: v for k, v in vars(config.__class__).items() if not k.startswith("__")}
        logger.experiment.log_parameters(config_dict)
        logger.experiment.add_tags(config.tags)

        if logger.experiment.get_key() is not None:

            config.experiment_id = logger.experiment.get_key()
            path = f"{config.offline_directory}/{config.project}/{config.experiment_id}"
            os.makedirs(path, exist_ok=True)

            with open(os.path.join(path, "config.yaml"), "w") as f:
                yaml.safe_dump(config_dict, f, sort_keys=False, default_flow_style=False)

    return logger
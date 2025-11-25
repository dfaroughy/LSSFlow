from pathlib import Path

from utils import BASE_DIR, PROJECT
from data.datasets import UchuuVoxelizedData
from utils.experiments import Configs, RunExperiment

#==========================================
exp_id = 'dd0d18717efb495d8243fbad80a62c03'
#==========================================

config = Configs(load_config=Path(BASE_DIR, PROJECT, exp_id, "config.yaml"))
lss = UchuuVoxelizedData(datafile=config.datafile, box_size=config.box_size, voxel_size=config.voxel_size)
target = lss.sample(voxel_idx=config.voxel_idx, normalize=True)
experiment = RunExperiment(config=config, target=target)
experiment.resume(exp_id)
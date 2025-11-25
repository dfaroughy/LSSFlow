from pathlib import Path

from utils import BASE_DIR, PROJECT
from data.datasets import Uniform3D
from utils.experiments import Configs, RunExperiment


exp_id = 'dd0d18717efb495d8243fbad80a62c03' 

config = Configs(load_config=Path(BASE_DIR, PROJECT, exp_id, "config.yaml"),  
                 ckpt = 'best',
                 num_points = 153_155,
                 num_steps = 128,
                 batch_size = 2048
                 )

#=================================================

uniform = Uniform3D(support='cube')
noise = uniform.sample(config.num_points)
experiment = RunExperiment(config=config, source=noise)               
experiment.generate(exp_id)
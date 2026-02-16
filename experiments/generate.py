from pathlib import Path

from utils import BASE_DIR, PROJECT
from data.datasets import Uniform3D
from utils.experiments import Configs, RunExperiment


exp_id = 'a5ee61bc36d3419384e4dc3346a21fb6' 

config = Configs(load_config=Path(BASE_DIR, PROJECT, exp_id, "config.yaml"),  
                 source_support = 'ball',
                 ckpt = 'best',
                 num_points = 490_483,
                 num_steps = 128,
                 batch_size = 2048
                 )

#=================================================

uniform = Uniform3D(support=config.source_support)
noise = uniform.sample(config.num_points)
experiment = RunExperiment(config=config, source=noise)               
experiment.generate(exp_id)
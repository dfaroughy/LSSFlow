from pathlib import Path

from utils import BASE_DIR, PROJECT
from data.datasets import AbbacusData
from utils.experiments import Configs, RunExperiment

#==========================================
exp_id = 'a5ee61bc36d3419384e4dc3346a21fb6'
#==========================================

config = Configs(load_config=Path(BASE_DIR, PROJECT, exp_id, "config.yaml"))
abbacus = AbbacusData(datafile=config.datafile, radius=config.radius)
target = abbacus.sample()
experiment = RunExperiment(config=config, target=target)
experiment.resume(exp_id)

import torch
import pytorch_lightning as L
from dataclasses import dataclass
from pathlib import Path

from generative_models.cfm import ConditionalFlowMatching
from data.datasets import Uniform3D
from utils.experiments import build_dataloaders


#=================================================
@dataclass
class Config:
    num_nodes = 1
    exp_id = 'e5fc6bf518fa44d28292204205af8926'
    base_dir = '/pscratch/sd/d/dfarough/LargeScaleStructure'
    ckpt = 'best'
    num_points = 153_155
    num_steps = 250
    batch_size = 2048
    flow = 'uniform'
    dim = 3
    dim_fourier = 256
    n_embd = 256
    num_blocks = 8
    dropout = 0.1
    sigma = 1e-5            # cfm hyperparameter
    gamma = 1.0             # fourier feature hyperparameter
    mass_reg = 1.0
    use_mass_reg = True
    use_OT = True
    tag = 'cfm_uchuu_uniform'
#=================================================

config = Config()
path_model = Path(config.base_dir, config.exp_id)
uniform = Uniform3D(support='cube')
source = uniform.sample(config.num_points)

predict_dataloader, _ = build_dataloaders(source=source, 
                                          batch_size=config.batch_size)

model = ConditionalFlowMatching.load_from_checkpoint(path_model/'checkpoints'/f'{config.ckpt}.ckpt',  
                                                     map_location="cpu", 
                                                     config=config)

trainer = L.Trainer(accelerator='gpu', 
                    devices=[0], 
                    num_nodes=config.num_nodes,
                    inference_mode=False)

sample_batched = trainer.predict(model, predict_dataloader)

trajectories = torch.cat(sample_batched, dim=0)  # (N, num_timesteps, 3) 
trajectories = trajectories.permute(1,0,2)  # (num_timesteps, N, 3)
gen_sample = trajectories[-1]

torch.save(gen_sample, path_model / 'gen_sample.pt')
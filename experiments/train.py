import pytorch_lightning as L
from dataclasses import dataclass

from generative_models.cfm import ConditionalFlowMatching
from data.datasets import SphericalUniform, UchuuCentered
from utils.experiments import build_dataloaders, set_comet_logger


#==================

@dataclass
class Config:
    num_nodes = 4
    api_key = '8ONjCXJ1ogsqG1UxQzKxYn7tz'
    project_name = 'LargeScaleStructure'
    workspace = 'dfaroughy'
    save_dir = '/pscratch/sd/d/dfarough/LargeScaleStructure/results'
    datafile = '/pscratch/sd/d/dfarough/LargeScaleStructure/data/Uchuu1000-Pl18_z0p00_hlist_4.h5'
    radius = 36
    batch_size = 1024
    max_epochs = 50000
    train_split = 1.0
    lr = 0.001
    lr_final = 0.0001
    warmup_epochs = 5
    flow = 'uniform'
    dim = 3
    dim_fourier = 256
    n_embd = 256
    num_blocks = 8
    dropout = 0.1
    sigma = 1e-5 # cfm hyperparameter
    gamma = 1.0  # fourier feature hyperparameter
    mass_reg = 1.0
    use_mass_reg = True
    use_OT = True
    tag = 'cfm_uchuu'
#==================

config = Config()
source = SphericalUniform(radius=1)
lss = UchuuCentered(datafile=config.datafile, radius=config.radius)

train_dataloader, _ = build_dataloaders(source=source, 
                                        target=lss, 
                                        batch_size=config.batch_size, 
                                        train_split=config.train_split)

callback = L.callbacks.ModelCheckpoint(dirpath=None,
                                       monitor="train_loss",
                                       filename="best",
                                       save_top_k=1,
                                       mode="min",
                                       save_last=True,
                                        )

logger = set_comet_logger(config)

trainer = L.Trainer(max_epochs=config.max_epochs, 
                    accelerator='gpu', 
                    devices='auto',
                    num_nodes=config.num_nodes,
                    strategy='ddp',
                    callbacks=[callback],
                    logger=logger,
                    sync_batchnorm=True,
                    gradient_clip_val=1.0,
                    )

cfm = ConditionalFlowMatching(config)
trainer.fit(cfm, train_dataloader)
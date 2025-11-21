import pytorch_lightning as L
from cfm import ConditionalFlowMatching
from pytorch_lightning.loggers import CometLogger
from dataclasses import dataclass

from datasets import SphericalUniform, UchuuCentered
from utils import build_dataloaders


#==================

@dataclass
class Config:
    api_key = '8ONjCXJ1ogsqG1UxQzKxYn7tz'
    project_name = 'LargeScaleStructure'
    workspace = 'dfaroughy'
    offline_directory = '/pscratch/sd/d/dfarough/LargeScaleStructure/Results'
    datafile = '/pscratch/sd/d/dfarough/LargeScaleStructure/data/Uchuu1000-Pl18_z0p00_hlist_4.h5'
    batch_size = 1024
    max_epochs = 10000
    train_split = 0.8
    lr = 0.0001
    dim = 3
    dim_fourier = 2048
    n_embd = 2048
    num_blocks = 10
    dropout = 0.1
    sigma = 1e-4 # cfm hyperparameter
    gamma = 1.0  # fourier feature hyperparameter
    mass_reg = 0.1
    use_mass_reg = True
    use_OT = True

#==================

config = Config()
uniform = SphericalUniform(radius=1)
lss = UchuuCentered(datafile=config.datafile, radius=36)

train_dataloader, val_dataloader = build_dataloaders(source=uniform, 
                                                     target=lss, 
                                                     batch_size=config.batch_size, 
                                                     train_split=config.train_split)

callback = L.callbacks.ModelCheckpoint(dirpath=None,
                                       monitor="val_loss",
                                       filename="best",
                                       save_top_k=1,
                                       mode="min",
                                       save_last=True,
                                        )

logger = CometLogger(api_key=config.api_key,
                     project=config.project_name,
                     workspace=config.workspace,
                     offline_directory=config.offline_directory
                    )

trainer = L.Trainer(max_epochs=config.max_epochs, 
                    accelerator='gpu', 
                    devices='auto',
                    strategy='ddp_find_unused_parameters_true',
                    callbacks=[callback],
                    logger=logger,
                    sync_batchnorm=True,
                    gradient_clip_val=1.0,
                    )

cfm = ConditionalFlowMatching(config)
trainer.fit(cfm, train_dataloader, val_dataloader)
import yaml
import os
import torch
import pytorch_lightning as L
from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

from utils import BASE_DIR, COMET_API_KEY, WORKSPACE, PROJECT
from data.datasets import DataCoupling
from generative_models.flow_matching import ConditionalFlowMatching
from generative_models.action_matching import ActionMatching


@dataclass
class Configs:
    """General configuration class that can load
       from existing experiment directory given exp_key.
    """
    base_dir = BASE_DIR
    api_key = COMET_API_KEY
    workspace = WORKSPACE
    project_name = PROJECT
    experiment_id: str = None
    
    def __init__(self, load_config: Path=None, **kwargs):

        if load_config is not None:
            loaded_config = yaml.safe_load(open(load_config))
            for key, value in loaded_config.items():
                setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class RunExperiment:
    def __init__(self, 
                 config: Configs, 
                 target=None,
                 source=None
                 ):

        self.config = config
        self.target = target
        self.source = source

        if config.gen_model == 'ConditionalFlowMatching':
            self.gen_model = ConditionalFlowMatching

        elif config.gen_model == 'ActionMatching':
            self.gen_model = ActionMatching

    def train(self):
        model = self.gen_model(config=self.config)
        dataloader, _ = self.build_dataloaders()
        logger = self.set_comet_logger()

        callback = L.callbacks.ModelCheckpoint(dirpath=None,
                                                monitor="train_loss",
                                                filename="best",
                                                save_top_k=1,
                                                mode="min",
                                                save_last=True
                                                )
        trainer = L.Trainer(max_epochs=self.config.max_epochs, 
                            accelerator='gpu', 
                            devices='auto',
                            num_nodes=self.config.num_nodes,
                            strategy='ddp',
                            callbacks=[callback],
                            logger=logger,
                            sync_batchnorm=True,
                            gradient_clip_val=1.0,
                            )
        trainer.fit(model, dataloader)

    def resume(self, experiment_id: str):
        ckpt_path = Path(BASE_DIR, PROJECT, experiment_id, 'checkpoints', 'last.ckpt')
        model = self.gen_model.load_from_checkpoint(ckpt_path,
                                                    config=self.config,
                                                    map_location="cpu"
                                                    )

        dataloader, _ = self.build_dataloaders()
        logger = self.set_comet_logger()

        callback = L.callbacks.ModelCheckpoint(dirpath=None,
                                            monitor="train_loss",
                                            filename="best",
                                            save_top_k=1,
                                            mode="min",
                                            save_last=True
                                            )
        trainer = L.Trainer(max_epochs=self.config.max_epochs, 
                            accelerator='gpu', 
                            devices='auto',
                            num_nodes=self.config.num_nodes,
                            strategy='ddp',
                            callbacks=[callback],
                            logger=logger,
                            sync_batchnorm=True,
                            gradient_clip_val=1.0,
                            )
        trainer.fit(model, dataloader, ckpt_path=ckpt_path)

    def generate(self, experiment_id: str):
        path = Path(BASE_DIR, PROJECT, experiment_id)
        model = self.gen_model.load_from_checkpoint(path / f'checkpoints/{self.config.ckpt}.ckpt',
                                                    config=self.config,
                                                    map_location="cpu"
                                                    )

        dataloader, _ = self.build_dataloaders()
        generator = L.Trainer(accelerator='gpu', 
                              devices=[0], 
                              num_nodes=1,
                              inference_mode=False)

        sample_batched = generator.predict(model, dataloader)
        trajectories = torch.cat(sample_batched, dim=0)  
        gen_sample = trajectories.permute(1,0,2)[-1]  
        torch.save(gen_sample, path / 'gen_sample.pt')

    def build_dataloaders(self):
        dataset = DataCoupling(target=self.target, source=self.source)
        train_size = int(self.config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        if val_size == 0:
            return train, None
        val = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train, val

    @rank_zero_only
    def set_comet_logger(self):

        logger = CometLogger(api_key=COMET_API_KEY,
                             project_name=PROJECT,
                             workspace=WORKSPACE,
                             save_dir=BASE_DIR,
                             experiment_key=self.config.experiment_id
                             )

        # create new experiment
        if self.config.experiment_id is None:

            tags = self.config.tags + [self.config.data_name, 
                                       self.config.gen_model, 
                                       self.config.network]

            if hasattr(self.config, 'flow'):
                tags += [self.config.flow]

            logger.experiment.log_parameters(vars(self.config))
            logger.experiment.add_tags(tags)
            self.config.experiment_id = logger.experiment.get_key()
            path = Path(BASE_DIR, PROJECT, self.config.experiment_id)
            os.makedirs(path, exist_ok=True)
            
            # save config file
            with open(path / "config.yaml", "w") as f:
                yaml.safe_dump(vars(self.config), f, sort_keys=False, default_flow_style=False)

        return logger




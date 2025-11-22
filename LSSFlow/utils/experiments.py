import yaml
import os
import numpy as np
import torch
from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, random_split

from data.datasets import DataCoupling

def build_dataloaders(source=None, target=None, batch_size=1024, train_split=1.0):

    dataset = DataCoupling(target=target, source=source)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_size == 0:
        return train, None

    val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train, val


@rank_zero_only
def set_comet_logger(config):
    logger = CometLogger(api_key=config.api_key,
                         project_name=config.project_name,
                         workspace=config.workspace,
                         save_dir=config.save_dir
                        )
    if not hasattr(config, "experiment_id"): # if new experiment

        config_dict = {k: v for k, v in vars(config.__class__).items() if not k.startswith("__")}
        logger.experiment.log_parameters(config_dict)
        logger.experiment.add_tags(config.tag)

        if logger.experiment.get_key() is not None:

            config.experiment_id = logger.experiment.get_key()
            path = f"{config.save_dir}/{config.experiment_id}"
            os.makedirs(path, exist_ok=True)

            with open(os.path.join(path, "config.yaml"), "w") as f:
                yaml.safe_dump(config_dict, f, sort_keys=False, default_flow_style=False)

    return logger


class FlowGeneratorCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experiment_dir = Path(f'{config.save_dir}/{config.experiment_id}')
        self.tag = f"_{config.tag}" if config.tag else ""

    def on_predict_start(self, trainer, pl_module):
        self.batched_data = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.batched_data.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank
        self._save_results_local(rank)
        trainer.strategy.barrier()  # wait for all ranks to finish

        if trainer.is_global_zero:
            self._gather_results_global(trainer)
            self._clean_temp_files()

    def _save_results_local(self, rank):
        randint = np.random.randint(0, 1000000)
        data = torch.cat(self.batched_data, dim=0)
        data.save_to(f"{self.experiment_dir}/temp_data{self.tag}_{rank}_{randint}.h5")

    @rank_zero_only
    def _gather_results_global(self, trainer):
    
        os.mkdir(f'{self.experiment_dir}/generation_results{self.tag}')

        with open(f'{self.experiment_dir}/generation_results{self.tag}/configs.yaml' , 'w' ) as outfile:
            yaml.dump( self.config.__dict__, outfile, sort_keys=False)

        temp_files = self.experiment_dir.glob(f"temp_data{self.tag}_*.h5")
        sample = torch.cat([torch.load(str(f)) for f in temp_files], dim=0)

        if sample.has_continuous: # post-process continuous features
            mu = torch.tensor(self.config.metadata['mean'])
            sig = torch.tensor(self.config.metadata['std'])
            sample.continuous = (sample.continuous * sig) + mu

        sample.apply_mask()  # ensure mask is applied to pads
        sample.save_to(f'{self.experiment_dir}/generation_results{self.tag}/generated_sample.h5')

    def _clean_temp_files(self):
        for f in self.experiment_dir.glob(f"temp_data{self.tag}_*.h5"):
            f.unlink()

import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass

from data.datasets import Uniform3D
from generative_models.architectures import ResNet, LearnableFourierEmbedding
from generative_models.dynamics import UniformFlow

@dataclass
class Config:
    dim = 3
    flow = 'uniform'
    gamma = 1.0   # fourier feature hyperparameter
    n_embd = 128
    dim_fourier = 64
    mass_reg = 1e-4
    use_OT = False
    num_blocks = 3
    dropout = 0.1
    lr = 0.001
    lr_final = 1e-5
    max_epochs = 100
    warmup_epochs = 5
    num_steps = 100


class ActionMatching(L.LightningModule):

    """ MLP-based Dynamical generative model 
    """    

    def __init__(self, config: Config):
        super().__init__()
        
        self.model = nn.ModuleDict(dict(fourier_feats=LearnableFourierEmbedding(x_dim=config.dim + 1,   # time + state
                                                                                fourier_dim=config.dim_fourier,
                                                                                gamma=config.gamma, 
                                                                                group_sizes=[1,3], 
                                                                                group_fourier_dims=[config.dim_fourier//2, config.dim_fourier//2],
                                                                                ), 
                                        wf=nn.Sequential(nn.Linear(config.dim_fourier, config.n_embd),
                                                         nn.LeakyReLU(0.2),
                                                         nn.Linear(config.n_embd, config.n_embd),
                                                         ),
                                        wx=nn.Sequential(nn.Linear(config.dim, config.n_embd),
                                                         nn.LeakyReLU(0.2),
                                                         nn.Linear(config.n_embd, config.n_embd),
                                                         ),
                                        resnet=ResNet(dim_input=2*config.n_embd, 
                                                      dim_embd=2*config.n_embd, 
                                                      dim_out=1, 
                                                      n_blocks=config.num_blocks, 
                                                      dropout=config.dropout,
                                                      activation=config.activation)
                                     ))
        self.config = config
        self.conditional_dynamics = UniformFlow(sigma=0.0, use_OT=config.use_OT)
        
        self.weight_scheduler = nn.Sequential(nn.Linear(1, config.n_embd),
                                              nn.LeakyReLU(0.2),
                                              nn.Linear(config.n_embd, 1),
                                              nn.Sigmoid()
                                              )
                                                
        nn.init.constant_(self.weight_scheduler[2].bias, 1.0) 

        if config.use_mass_reg:
            self.mass_reg_phi = nn.Parameter(torch.tensor(config.mass_reg))
            self.mass_reg_wt = nn.Parameter(torch.tensor(config.mass_reg))
            
        self.source = Uniform3D(support='cube')  # source distribution


    # ...Lightning methods 

    def forward(self, x, t) -> torch.Tensor:
        """ Parametrization of the potential field phi
        """
        t = t.unsqueeze(-1)
        tx = torch.cat([t, x], dim=-1)
        f = self.model.fourier_feats(tx)
        x_emb = self.model.wx(x)
        f_emb = self.model.wf(f)
        h = torch.cat([x_emb, f_emb], dim=-1)
        return self.model.resnet(h), self.weight_scheduler(t)

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('train_loss', 
                  loss,
                  on_epoch=True,
                  on_step=True,
                  prog_bar=True,
                  logger=True,
                  sync_dist=True,
                  batch_size=len(batch))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('val_loss', 
                  loss,                 
                  on_epoch=True,
                  on_step=True,
                  prog_bar=True,
                  logger=True,
                  sync_dist=True,
                  batch_size=len(batch))
        return {"val_loss": loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ''' for sample generation
        '''
        trajectories = self.simulate_dynamics(batch)
        return trajectories.detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        cosine_epochs = max(self.config.max_epochs - self.config.warmup_epochs, 1)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.config.lr_final,
            last_epoch=-1
        )

        # linear warmup over the first `warmup_epochs` 
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",   # step per epoch
            "frequency": 1,
            "strict": True,
            },
        }

    # ...Model functions

    def loss(self, batch):

        x1 = batch.target
        x0 = self.source.sample(num_points=len(x1), device=self.device)
        t = torch.rand(len(x1), device=self.device)
        xt = self.conditional_dynamics.sample(x0, x1, t)  

        with torch.set_grad_enabled(True):
            t = t.detach().requires_grad_(True)
            xt = xt.detach().requires_grad_(True)  
            phi, wt = self.forward(xt, t)
            phi0, w0 = self.forward(x0, torch.zeros_like(t))
            phi1, w1 = self.forward(x1, torch.ones_like(t))
            nabla_phi, dot_phi = self.grad_phi(phi, xt, t)
            dot_wt = self.dwdt(wt, t)         

        kin_energy = 0.5 * (nabla_phi**2).sum(1, keepdims=True)
        loss = phi0 * w0 - phi1 * w1 + (kin_energy + dot_phi) * wt + phi * dot_wt
        loss = loss.mean()

        if self.config.use_mass_reg:
            loss += torch.abs(self.mass_reg_phi) * torch.mean(phi**2)  # mass term regularization
            loss += torch.abs(self.mass_reg_wt) * torch.mean(wt**2)    # mass term regularization
            
        return loss

    def simulate_dynamics(self, batch):
        """ Generate target data from source input by solving the ODE with 
            Euler's method.
            Returns: full time-dependent paths from x0 to x1.
        """
        self.eval()
        device = self.device
        time_steps = torch.linspace(0.0, 1.0, self.config.num_steps, device=device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        x = batch.source.clone().to(device).requires_grad_(True) 
        paths = [x.clone()]  # append t=0 source

        for t in time_steps:
            time = torch.full((len(x),), t.item(), device=device).requires_grad_(True)

            with torch.set_grad_enabled(True):
                phi, _ = self.forward(x, time)
                nabla_phi, _ = self.grad_phi(phi, x, time)

            with torch.no_grad():
                x += nabla_phi * delta_t

            paths.append(x.clone())

        return torch.stack(paths, dim=1)  # (B, num_steps, dim)

    def grad_phi(self, phi, x, t):
        nabla_phi, dot_phi = torch.autograd.grad(phi,
                                                 (x,t),
                                                 grad_outputs=torch.ones_like(phi),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 )
        return nabla_phi, dot_phi

    def dwdt(self, wt, t):
        return torch.autograd.grad(wt,
                                    t,
                                    grad_outputs=torch.ones_like(wt),
                                    create_graph=True,
                                    retain_graph=True,
                                    )[0]
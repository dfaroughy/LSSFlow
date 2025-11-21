import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
import ot as pot
from functools import partial
from models import ResNet, MLP, LearnableFourierEmbedding

@dataclass
class Config:
    dim = 3
    n_embd = 128
    sigma = 0.01  # cfm hyperparameter
    gamma = 1.0   # fourier feature hyperparameter
    mass_reg = 1e-4
    use_OT = False
    num_blocks = 3
    dropout = 0.1
    lr = 0.001
    lr_final = 1e-5
    max_epochs = 100
    warmup_epochs = 5
    num_steps = 100


class ConditionalFlowMatching(L.LightningModule):

    """ MLP-based Dynamical generative model 
    """    

    def __init__(self, config: Config):
        super().__init__()
        
        self.model = nn.ModuleDict(dict(fourier_feats=LearnableFourierEmbedding(
                                                        x_dim=config.dim + 1,   # time + state
                                                        fourier_dim=config.dim_fourier,
                                                        gamma=config.gamma, 
                                                        group_sizes=[1,3], 
                                                        group_fourier_dims=[config.dim_fourier//2, config.dim_fourier//2],
                                                        ), 
                                        wf = nn.Sequential(nn.Linear(config.dim_fourier, config.n_embd),
                                                           nn.GELU(),
                                                           nn.Linear(config.n_embd, config.n_embd),
                                                           ),
                                        wx=nn.Sequential(nn.Linear(config.dim, config.n_embd),
                                                         nn.GELU(),
                                                         nn.Linear(config.n_embd, config.n_embd),
                                                        ),
                                        resnet=ResNet(dim_input=2*config.n_embd, 
                                                      dim_embd=2*config.n_embd, 
                                                      dim_out=1, 
                                                      n_blocks=config.num_blocks, 
                                                      dropout=config.dropout),
                                     ))
        self.conditional_dynamics = UniformFlow(sigma=config.sigma, use_OT=config.use_OT)
        self.config = config

        if config.use_mass_reg:
            self.mass_reg = nn.Parameter(torch.tensor(config.mass_reg))

    # ...Lightning methods 

    def forward(self, x, t) -> (torch.Tensor, torch.Tensor):
        """ Parametrization of the potential field phi
        """
        tx = torch.cat([t.unsqueeze(-1), x], dim=-1)
        f = self.model.fourier_feats(tx)
        x_emb = self.model.wx(x)
        f_emb = self.model.wf(f)
        h = torch.cat([x_emb, f_emb], dim=-1)
        return self.model.resnet(h)

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('train_loss', 
                  loss,
                  on_epoch=True,
                  on_step=False,
                  prog_bar=True,
                  logger=True,
                  sync_dist=True,
                  batch_size=len(batch))
        self.log('train_loss_step', 
                    loss,
                    on_epoch=False,
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=len(batch))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # with torch.set_grad_enabled(True):
        loss = self.loss(batch)
        self.log('val_loss', 
                  loss,                 
                  on_epoch=True,
                  on_step=False,
                  prog_bar=True,
                  logger=True,
                  sync_dist=True,
                  batch_size=len(batch))
        self.log('val_loss_step', 
                    loss,
                    on_epoch=False,
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=len(batch))
        return {"val_loss": loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ''' for sample generation
        '''
        # with torch.set_grad_enabled(True):
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

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=self.config.lr)

    # ...Model functions

    def loss(self, batch):
        x0, x1 = batch.source, batch.target
        t = torch.rand(len(x0), device=self.device)
        xt = self.conditional_dynamics.sample(x0, x1, t)
        with torch.set_grad_enabled(True):   
            xt = xt.detach().requires_grad_(True)  
            phi = self.forward(xt, t)
            vt = -self.grad_phi(phi, xt)
        ut = self.conditional_dynamics.vector_field()
        loss =  F.mse_loss(vt, ut, reduction='mean')

        if self.config.use_mass_reg:
            loss += torch.abs(self.mass_reg) * torch.mean(phi**2)  # mass term regularization

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
        with torch.set_grad_enabled(True):
            x = batch.source.clone().to(device).requires_grad_(True) 
        paths = [x.clone()]  # append t=0 source

        for t in time_steps:
            time = torch.full((len(x),), t.item(), device=device)
            phi = self.forward(x, time)
            with torch.set_grad_enabled(True):
                vt = -self.grad_phi(phi, x)
            with torch.no_grad():
                x += vt * delta_t
            paths.append(x.clone())

        return torch.stack(paths, dim=1)  # (B, num_steps, dim)

    def grad_phi(self, phi, x):
        grad_phi = torch.autograd.grad(
            phi,
            x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad_phi


# conditional dynamics:


class UniformFlow:
    """Conditional dynamics linear interpolation between 
       boundaries states at t=0 and t=1.
    notation:
      - t: time
      - x0: (B, dim) continuous source state at t=0
      - x1: (B, dim) continuous target state at t=1
      - xt: (B, dim) continuous state at time t
      - z: (B, dim) delta function regularizer
    """

    def __init__(self, sigma, use_OT=False):
        self.sigma = sigma
        self.use_OT = use_OT

    def sample(self, x0, x1, t) -> torch.Tensor:
        ''' sample xt ~ p_t(x|x0,x1)
        '''

        if self.use_OT:
            OT = OTSampler(batch_size=x0.shape[0], replace=False)	
            x0, x1 = OT.sample_plan(x0, x1)
            
        self.x0 = x0
        self.x1 = x1

        t = self.reshape_time_dim_like(t, x1)    # (B,) -> (B, 1)
        xt = t * x1 + (1.0 - t) * x0             # time-interpolated state
        z = torch.randn_like(xt)                 # noise
        xt += self.sigma * z                     # Dirac -> Gauss smear
        return xt

    def vector_field(self) -> torch.Tensor:
        ''' u_t(x|x0,x1)
        '''
        return self.x1 - self.x0

    def reshape_time_dim_like(self, t, state):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (state.ndim - 1)))


# OT

class OTSampler:
    def __init__(self, batch_size, replace=False):
        self.ot_fn = partial(pot.emd, numThreads=1)
        self.batch_size = batch_size
        self.replace = replace
        
    def get_map(self, x0, x1):
        a = pot.unif(x0.shape[0])
        b = pot.unif(x1.shape[0])
        M = torch.cdist(x0, x1)**2 # euclidean distance
        p = self.ot_fn(a, b, M.detach().cpu().numpy())

        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            print("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=self.batch_size, replace=self.replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi)
        return x0[i], x1[j]
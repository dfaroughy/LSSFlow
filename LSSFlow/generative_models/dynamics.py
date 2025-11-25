import numpy as np
import torch
import ot as pot
from functools import partial


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



class FreeFallFlow:
    """Conditional dynamics with quadratic time interpolation between 
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
        self.t = self.reshape_time_dim_like(t, x1)    # (B,) -> (B, 1)
        
        xt = self.t**2 * x1 + (1.0 - self.t**2) * x0    # time-interpolated state
        z = torch.randn_like(xt)                        # noise
        xt += self.sigma * z                            # Dirac -> Gauss smear
        return xt

    def vector_field(self) -> torch.Tensor:
        ''' u_t(x|x0,x1)
        '''
        return 2 * self.t * (self.x1 - self.x0)

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
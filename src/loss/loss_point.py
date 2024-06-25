from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


def remove_outlier(a):
    mean = a.mean()  
    std_dev = a.std()  
    z_scores = (a - mean) / std_dev  
    threshold = 2
    outlier_indices = torch.abs(z_scores) > threshold  
    a_without_outliers = a[~outlier_indices]  
    # print("Outlier values:", a[outlier_indices])  
    return a_without_outliers


@dataclass
class LossPointCfg:
    weight: float = 1.0

@dataclass
class LossPointCfgWrapper:
    point: LossPointCfg

class LossPoint(Loss[LossPointCfg, LossPointCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        use_conf = True
        # use_conf = False

        depth_mk_pc0s, depth_mk_pc1s, mkpt_pcs, mconfs= batch['depth_cost_triangulation'] # [b, n, 3]
       
        loss = 0.0
        for bi in range(len(depth_mk_pc0s)):
            # l0 = torch.norm(mkpt_pcs[bi] - depth_mk_pc0s[bi], dim=-1)
            # l1 = torch.norm(mkpt_pcs[bi] - depth_mk_pc1s[bi], dim=-1)

            l0 = torch.sum(torch.sqrt(torch.pow((mkpt_pcs[bi] - depth_mk_pc0s[bi]), 2)), dim=1)
            l1 = torch.sum(torch.sqrt(torch.pow((mkpt_pcs[bi] - depth_mk_pc1s[bi]), 2)), dim=1)

            # l0 = torch.sum(torch.log(1 + torch.abs(mkpt_pcs[bi] - depth_mk_pc0s[bi])), dim=1)
            # l1 = torch.sum(torch.log(1 + torch.abs(mkpt_pcs[bi] - depth_mk_pc1s[bi])), dim=1)

            if use_conf == True:
                l0 *= mconfs[bi] 
                l1 *= mconfs[bi] 

            # l_bi = 0.5 * (l0.mean() + l1.mean())
            l_bi = 0.5 * (remove_outlier(l0).mean() + remove_outlier(l1).mean())
            loss += l_bi

        return self.cfg.weight * loss
    

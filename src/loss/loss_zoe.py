from dataclasses import dataclass
from jaxtyping import Float

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

from src.visualization.vis_depth import viz_depth_tensor
from PIL import Image
import numpy as np

'''
zoe depth Loss
'''
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths from monodepth2
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


@dataclass
class LossZoeCfg:
    weight: float = 1.0
    

@dataclass
class LossZoeCfgWrapper:
    zoe: LossZoeCfg

class LossZoe(Loss[LossZoeCfg, LossZoeCfgWrapper]):
    def forward(
            self,
            prediction: DecoderOutput,
            batch: BatchedExample,
            gaussians: Gaussians,
            global_step: int,
        ) -> Float[Tensor, ""]:
        near, far = 0.0, 100.0
        zoe_depth = batch['zoe_depth'].squeeze(-1).squeeze(-1)
        pred_depth = batch["context"]["est_depth"].squeeze(-1).squeeze(-1) # b v h w srf s -> b v h w
        pred_depth = torch.clamp(pred_depth, near, far) # np.clip

        delta = pred_depth - zoe_depth
        return self.cfg.weight * (delta**2).mean()
    
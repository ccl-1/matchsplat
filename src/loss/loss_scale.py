from dataclasses import dataclass
from jaxtyping import Float

import torch
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


'''
scale Loss
'''

@dataclass
class LossScaleCfg:
    weight: float = 1.0
    

@dataclass
class LossScaleCfgWrapper:
    scale: LossScaleCfg

class LossScale(Loss[LossScaleCfg, LossScaleCfgWrapper]):
    def forward(
            self,
            prediction: DecoderOutput,
            batch: BatchedExample,
            gaussians: Gaussians,
            global_step: int,
        ) -> Float[Tensor, ""]:

        extrinsics = batch["context"]["extrinsics"]
        b, v, _, _ = extrinsics.shape  
        if v == 2:
            pose_ref = extrinsics[:, 0].clone().detach()
            pose_tgt = extrinsics[:, 1].clone().detach()
            pose = pose_tgt.inverse() @ pose_ref  # torch.Size([1, 4, 4])
        return self.cfg.weight * (torch.norm(batch['pred_scale']- pose[:, :3, 3], dim=-1).mean())
    
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
from src.loss.PWN_planes import PWNPlanesLoss

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

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)

def mse_loss(prediction, target, mask, reduction=reduction_image_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))
    return reduction(image_loss, 2 * M)

def logl1_loss(prediction, target, mask, reduction=reduction_image_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(torch.log(1 + torch.abs(res)), (1, 2))
    
    return reduction(image_loss, M)

def depth_loss(prediction, target, mask):
    """ 
    prediction  [Tensor, "B H W"]
    target      [Tensor, "B H W"]
    mask        [Tensor, "B H W"]
    """
    scale, shift = compute_scale_and_shift(prediction, target, mask)
    prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    return logl1_loss(prediction_ssi, target, mask, reduction_image_based)


def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    pred_metric = a * pred + b
    return pred_metric


@dataclass
class LossZoeCfg:
    weight: float = 1.0
    
# https://github.com/aim-uofa/AdelaiDepth/blob/main/LeReS/Train/lib/models/PWN_planes.py
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

        self.use_vnl = False
        near, far = 0.0, 100
        intri = batch["context"]["intrinsics"] # b v 3 3
        zoe_depth = batch["context"]['zoe_depth'].squeeze(-1).squeeze(-1)
        pred_depth = batch["context"]["est_depth"].squeeze(-1).squeeze(-1) # b v h w srf s -> b v h w
        # pred_depth = torch.clamp(pred_depth, near, far)
        
        mask = (pred_depth > near) & (pred_depth < far)
        zoe_depth = zoe_depth * mask
        pred_depth = pred_depth * mask


        # delta = pred_depth - zoe_depth
        # return self.cfg.weight * (delta**2).mean()

        zoe_depth_0, pred_depth_0 = pred_depth[:,0,:,:], zoe_depth[:,0,:,:]
        zoe_depth_1, pred_depth_1 = pred_depth[:,1,:,:], zoe_depth[:,1,:,:]
        mask_0, mask_1 = torch.ones_like(pred_depth_0), torch.ones_like(pred_depth_1)
        loss = (depth_loss(mask_0, zoe_depth_0, mask_0) + depth_loss(mask_1, zoe_depth_1, mask_1)) / 2.
    

        if self.use_vnl:
            pred_depth_0, zoe_depth_0 = pred_depth[:,0,:,:].unsqueeze(1).cuda(), zoe_depth[:,0,:,:].unsqueeze(1).cuda() # b 1 h w
            pred_depth_1, zoe_depth_1 = pred_depth[:,1,:,:].unsqueeze(1).cuda(), zoe_depth[:,1,:,:].unsqueeze(1).cuda()
            mask_0, mask_1 = torch.ones_like(pred_depth_0), torch.ones_like(pred_depth_1)

            vnl_loss = PWNPlanesLoss()
            vnl = (vnl_loss(pred_depth_0, zoe_depth_0, mask_0, intri[:,0,:,:]) + \
                vnl_loss(pred_depth_1, zoe_depth_1, mask_1, intri[:,1,:,:])) / 2.
            loss +=vnl
        
        return self.cfg.weight * loss
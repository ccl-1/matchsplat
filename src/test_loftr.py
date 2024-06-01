import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch
from PIL import Image
from einops import einsum, rearrange, repeat
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np
import math


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.image_io import save_image
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.model.encoder.loftr.utils.full_config import full_default_cfg
    from src.model.encoder.loftr.utils.opt_config import opt_default_cfg
    from src.model.encoder.encoder_eloftr import EncoderELoFTR, reparameter
    from src.model.decoder import get_decoder
    from src.visualization.vis_depth import viz_depth_tensor
    from src.geometry.projection import (homogenize_points, project, 
                                         sample_image_grid, get_world_rays,
                                         calculate_distance_to_image_plane)






model_type = 'full' 
precision = 'fp32' 
if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)
    
if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True




def modify_ckpt(encoder, checkpoint_path):
    ckpt_new = {}
    model_dict = encoder.state_dict()
    ckpt = torch.load(checkpoint_path)['state_dict']
    for k, v in ckpt.items() :
        k = k.split('encoder.')[1]
        if k in model_dict and (v.shape == model_dict[k].shape):
            ckpt_new[k] = v 
    return ckpt_new



def scaled_K(batch, downscale):
    b, v, _, h, w = batch["context"]["image"].shape
    h, w = int(h / downscale), int(w / downscale)
    
    # unnormalized camera intrinsic
    intrinsics = batch["context"]["intrinsics"].clone().detach() 
    intrinsics[:, :, 0, :] *= float(w)
    intrinsics[:, :, 1, :] *= float(h)
    return intrinsics


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main_1",
)
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0)
    dataset = iter(data_module.train_dataloader())
    batch = next(dataset, 0)
    print ("context:", batch["scene"])
    
    # load model
    encoder = EncoderELoFTR(cfg=cfg.model.encoder, backbone_cfg =_default_cfg )
    encoder = encoder.to(device)
    encoder = encoder.eval()
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)

    # intr = batch["context"]["intrinsics"][0][0].unsqueeze(0)
    # extr = batch["context"]["extrinsics"][0][0].unsqueeze(0)

    # ckpt_path = "checkpoints/re10k.ckpt"
    depth_ckpt_path = "checkpoints/wo_fpn.ckpt"
    encoder.load_state_dict(torch.load(depth_ckpt_path), strict=False) # only load weight of depth_predictor

    visualization_dump = {}
    gaussians = encoder(batch, #["context"], 
                        0, False, visualization_dump=visualization_dump, scene_names=batch["scene"])
    
    # decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    # decoder = decoder.to(device)
    # print(visualization_dump.keys())

    # save context views
    # save_image(batch["context"]["image"][0, 0], f"outputs/out/input_0.png")
    # save_image(batch["context"]["image"][0, 1], f"outputs/out/input_1.png")

    # save encoder depth map
    # depth_vis = ((visualization_dump['P2']["depth"].squeeze(-1).squeeze(-1)).cpu().detach())
    # for v_idx in range(depth_vis.shape[1]):
    #     vis_depth = viz_depth_tensor(1.0 / depth_vis[0, v_idx], return_numpy=True)  # inverse depth
    #     Image.fromarray(vis_depth).save(f"outputs/out/depth_P2_{v_idx}.png")
    
    # Render depth.
    # ds =2
    # batch["context"]["intrinsics"] = scaled_K(batch, ds)
    # *_, h, w = batch["context"]["image"].shape
    # h, w = int(h/ds), int(w / ds)

"""
    rendered = decoder.forward( gaussians,
        batch["context"]["extrinsics"], batch["context"]["intrinsics"],
        batch["context"]["near"],batch["context"]["far"], (h, w), "depth",) # here depth mode = depth not null
    
    result = rendered.depth.cpu().detach()
    print("i get depth")
    for v_idx in range(result.shape[1]):
        vis_depth = viz_depth_tensor(1.0 / result[0, v_idx], return_numpy=True)  # inverse depth
        Image.fromarray(vis_depth).save(f"./outputs/out/depth_P2_{v_idx}_gs.png")
"""
run()


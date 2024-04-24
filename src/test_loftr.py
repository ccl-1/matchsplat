import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch
from PIL import Image
from einops import einsum, rearrange, repeat
from scipy.spatial.transform import Rotation as R


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
    from src.geometry.projection import homogenize_points, project





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


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main_1",
)
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    dataset = iter(data_module.train_dataloader())
    batch = next(dataset, 0)

    # load model
    encoder = EncoderELoFTR(cfg=cfg.model.encoder, backbone_cfg =_default_cfg )
    encoder = encoder.eval()
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)

    ckpt_path = "checkpoints/re10k.ckpt"
    new_ckpt_path = "checkpoints/re10k_loftr.ckpt"
    depth_ckpt_path = "checkpoints/depth_predictor.ckpt"
    print("==> Load depth_predictor checkpoint: %s" % depth_ckpt_path)
    encoder.load_state_dict(torch.load(depth_ckpt_path), strict=False) # only load weight of depth_predictor

    # ckpt_new = modify_ckpt(encoder, ckpt_path)
    # torch.save(ckpt_new, depth_ckpt_path)
    # encoder.load_state_dict(ckpt_new, strict=False) 
    # torch.save(encoder.state_dict(), new_ckpt_path)

    visualization_dump = {}
    gaussians = encoder(batch["context"], 0, False, visualization_dump=visualization_dump, scene_names=batch["scene"])
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    print(visualization_dump.keys())

    # save context views
    save_image(batch["context"]["image"][0, 0], f"outputs/out/input_0.png")
    save_image(batch["context"]["image"][0, 1], f"outputs/out/input_1.png")

    # save encoder depth map
    depth_vis = ((visualization_dump["depth"].squeeze(-1).squeeze(-1)).cpu().detach())
    for v_idx in range(depth_vis.shape[1]):
        vis_depth = viz_depth_tensor(1.0 / depth_vis[0, v_idx], return_numpy=True)  # inverse depth
        Image.fromarray(vis_depth).save(f"outputs/out/depth_{v_idx}.png")
    

    # TODO  decode bugs  ..... 
""" 
    # Render depth.
    *_, h, w = batch["context"]["image"].shape
    rendered = decoder.forward( gaussians,
        batch["context"]["extrinsics"], batch["context"]["intrinsics"],
        batch["context"]["near"],batch["context"]["far"], (h, w), "depth",)
    
    result = rendered.depth.cpu().detach()
    print(rendered)
    for v_idx in range(result.shape[1]):
        vis_depth = viz_depth_tensor(1.0 / result[0, v_idx], return_numpy=True)  # inverse depth
        Image.fromarray(vis_depth).save(f"/outputs/out/depth_{v_idx}_gs.png")
"""
run()


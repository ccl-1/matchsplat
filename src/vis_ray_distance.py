import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch
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
    
    from src.model.encoder.loftr.utils.full_config import full_default_cfg
    from src.model.encoder.loftr.utils.opt_config import opt_default_cfg
    from src.model.encoder.encoder_eloftr import EncoderELoFTR, reparameter
    from src.model.decoder import get_decoder
    from src.visualization.vis_depth import viz_depth_tensor
    from src.geometry.epipolar_lines import lift_to_3d
    from src.geometry.projection import (homogenize_points, project, 
                                         sample_image_grid, get_world_rays)






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
    intr = batch["context"]["intrinsics"][0][0].unsqueeze(0)
    extr = batch["context"]["extrinsics"][0][0].unsqueeze(0)

    # ht, wd = 256, 256
    ht, wd = 10, 10
    xy_ray, indices = sample_image_grid(((ht, wd)), intr.device) # H W 2 
    origins, directions = get_world_rays(xy_ray.reshape(1, -1, 2) , extr, intr) # [1, hw,3] [1.2272, -0.2104,  1.3145]
    
    coordinates_np, origins_np, directions_np = \
            xy_ray.numpy(), origins.numpy(), directions.numpy() # [1, hw, 3], [h, w, 2]
    
    depth = 10 * torch.ones((ht,wd))
    depth = depth.reshape(1, -1) 
    point_3d = origins + directions * depth[..., None]
    d_costvolume = torch.norm(point_3d - origins, dim=-1).squeeze(0)
    d_triangulation = (point_3d - origins).squeeze(0)[:, 2]

    #  ----- from image plane to ray depth -----
    t = directions.squeeze(0)
    dz = d_triangulation - origins[..., 2].squeeze(0)
    dx = dz * t[:, 0] / t[:, 2]  
    dy = dz * t[:, 1] / t[:, 2]  
    xyz = torch.stack((dx, dy, dz), dim=1)  
    #  ----- ------------------------------------


    plt.figure(figsize=(10, 5))
    # Plot grid points
    plt.subplot(2, 2, 1)
    plt.scatter(coordinates_np[..., 0], coordinates_np[..., 1] , c='red', marker='o')
    plt.title("Normalized Coordinates of Image Grid")
    plt.xlabel("Normalized x")
    plt.ylabel("Normalized y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    xx = list(range(d_triangulation.shape[0]))
    plt.plot(xx, d_costvolume.cpu().numpy(),  label='ray distance')
    plt.plot(xx, d_triangulation.cpu().numpy(), label='distance to image plane')
    plt.plot(xx, xyz[:,2].cpu().numpy(), label='tri_ray')

    plt.xlabel('H*W')
    plt.ylabel('Depth')
    plt.title('Comparasion of ray distance and distance to image plane ')
    plt.legend()
    plt.grid(True)
    
    # Plot origins and directions
    plt.subplot(2, 2, 3)
    plt.quiver(origins_np[..., 0], origins_np[..., 1], 
               directions_np[..., 0], directions_np[..., 1], 
               angles='xy', scale_units='xy', scale=1, color='blue')
    plt.scatter(origins_np[..., 0], origins_np[..., 1], c='red', marker='o')
    plt.title("Ray Origins and Directions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/tmp/grid.jpg')
    plt.show()


    # Plot origins and directions ON 3D
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(origins_np[..., 0], origins_np[..., 1], origins_np[..., 2], c='red', marker='o')
    ax.quiver(origins_np[..., 0], origins_np[..., 1], origins_np[..., 2], 
              directions_np[..., 0], directions_np[..., 1], directions_np[..., 2], 
             color='blue')
    plt.title("Ray Origins and Directions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig('outputs/tmp/ray.jpg')
    plt.show()

run()


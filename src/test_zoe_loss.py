import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch
from torchvision.utils import save_image
from PIL import Image

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
    from src.loss.loss_zoe import *
    from src.loss.PWN_planes import PWNPlanesLoss


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

    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    dataset = iter(data_module.train_dataloader())
    batch = next(dataset, 0)
    extri= batch["context"]["extrinsics"].cuda()
    intri = batch["context"]["intrinsics"].cuda() # torch.Size([1, 2, 3, 3])
    near = batch["context"]["near"].cuda()
    far = batch["context"]["far"].cuda()

    # ------------- test loss -------------------------------------


    loss_mode = 'zoe'
    # loss_mode = 'surface normal'

    zoe_depth = torch.rand(2, 2, 256,256) *10
    pred_depth = torch.rand(2, 2, 256,256)*20

    if loss_mode == 'zoe':  # for zoe depth
        print("==> test zoe depth loss ")
        pred_depth_0, zoe_depth_0 = pred_depth[:,0,:,:].cuda(), zoe_depth[:,0,:,:].cuda()
        pred_depth_1, zoe_depth_1 = pred_depth[:,1,:,:].cuda(), zoe_depth[:,1,:,:].cuda()
        mask_0, mask_1 = torch.ones_like(pred_depth_0), torch.ones_like(pred_depth_1)
        loss =  depth_loss(pred_depth_0, zoe_depth_0, mask_0) + depth_loss(pred_depth_1, zoe_depth_1, mask_1)

    elif loss_mode == 'surface normal':
        print("==> test surface normal loss ")
        pred_depth_0, zoe_depth_0 = pred_depth[:,0,:,:].unsqueeze(1).cuda(), zoe_depth[:,0,:,:].unsqueeze(1).cuda()
        pred_depth_1, zoe_depth_1 = pred_depth[:,1,:,:].unsqueeze(1).cuda(), zoe_depth[:,1,:,:].unsqueeze(1).cuda()
        mask_0, mask_1 = torch.ones_like(pred_depth_0), torch.ones_like(pred_depth_1)
        # mask_0, mask_1 = torch.zeros_like(pred_depth_0), torch.zeros_like(pred_depth_1)

        vnl_loss = PWNPlanesLoss()
        loss = vnl_loss(pred_depth_0, zoe_depth_0, mask_0, intri.squeeze(0)) + \
            vnl_loss(pred_depth_1, zoe_depth_1, mask_1, intri.squeeze(0))
    else:
        loss = 100
    print(loss)

run()


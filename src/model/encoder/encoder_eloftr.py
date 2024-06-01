import torch
from torch import Tensor, nn
import torchvision.transforms as tf
import torch.nn.functional as F
from torchvision.utils import save_image

from einops.einops import rearrange, repeat
from jaxtyping import Float
from collections import OrderedDict
import matplotlib.cm as cm
from PIL import Image


from dataclasses import dataclass
from typing import Literal, Optional, List

# loftr
from .loftr.loftr import LoFTR, reparameter
from.loftr.utils.plotting import make_matching_figure

# dataset
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians


from .encoder import Encoder
from ...global_cfg import get_cfg
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .costvolume.get_depth import DepthPredictorMultiView

from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from src.visualization.vis_depth import viz_depth_tensor


# model load once no all the time 。。 
def get_zoe_depth(zoe, imgs, vis= False):
    # repo = "isl-org/ZoeDepth"
    # zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True).cuda()
    b, v, c, h, w = imgs.size()
    depths = []
    for v_idx in range(v):
        img = imgs[:, v_idx, :, :, :].cuda()
        depth = zoe.infer(img)  # b 1 h w 

        if vis:
            vis_depth = viz_depth_tensor(depth[0][0].detach().cpu(), return_numpy=True)
            Image.fromarray(vis_depth).save(f"outputs/out/zoe_depth_{v_idx}.png")
       
        depths.append(depth.unsqueeze(1))
    depths = torch.cat(depths, dim=1) # b v c h w
    depths = repeat(depths, "b v dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1,)
    return depths


def save_match(context, mkpts0, mkpts1, mconf, path='./match.png'):
    # this can only be used in val/test mode
    color = cm.jet(mconf.cpu())
    text = ['LoFTR', 'Matches: {}'.format(len(mkpts0))]
    img1 = context["image"][0,0].permute(1,2,0).detach().numpy()
    img2 = context["image"][0,1].permute(1,2,0).detach().numpy()
    mkpts0, mkpts1 = mkpts0.detach().numpy(), mkpts1.detach().numpy()                               
    make_matching_figure(img1, img2,  mkpts0, mkpts1, color, text=text, path=path)
            
@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderELoFTRCfg:
    name: Literal["eloftr"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg

    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int

    eloftr_weights_path: str | None
    downscale_factor: int

    shim_patch_size: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    wo_fpn_depth: bool



class EncoderELoFTR(Encoder[EncoderELoFTRCfg]):
    backbone: LoFTR
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderELoFTRCfg, backbone_cfg) -> None:
        super().__init__(cfg)
        self.config = backbone_cfg
        self.return_cnn_features = True
        self.profiler = None

        # print("==> Load ZoeDepth model ")
        # repo = "isl-org/ZoeDepth"
        # self.zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True).cuda()
        
        self.matcher = LoFTR(backbone_cfg, profiler=self.profiler)   
        ckpt_path = cfg.eloftr_weights_path        
        if cfg.eloftr_weights_path is None:
            print("==> Init E-loFTR backbone from scratch")
        else:
            print("==> Load E-loFTR backbone checkpoint: %s" % ckpt_path)
            self.matcher.load_state_dict(torch.load(ckpt_path)['state_dict'], False)
            self.matcher = reparameter(self.matcher) # no reparameterization will lead to low performance
             
        self.cnn_64 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.cnn_128 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.cnn_256 = nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())

        self.trans_64 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.trans_128 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.trans_256 = nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())

        self.translation_regressor = nn.Sequential(
            nn.Linear(128*64*64 * 2, 128 ),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        if cfg.wo_fpn_depth:
            self.depth_predictor = DepthPredictorMultiView(
                feature_channels=cfg.d_feature,
                upscale_factor=cfg.downscale_factor,
                num_depth_candidates=cfg.num_depth_candidates,
                costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim, # input channels
                costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
                costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
                gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
                gaussians_per_pixel=cfg.gaussians_per_pixel,
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                depth_unet_feat_dim=cfg.depth_unet_feat_dim,
                depth_unet_attn_res=cfg.depth_unet_attn_res,
                depth_unet_channel_mult=cfg.depth_unet_channel_mult,
                wo_depth_refine=cfg.wo_depth_refine,
                wo_cost_volume=cfg.wo_cost_volume,
                wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            )  
        else:
            self.d_feature = [64, 128, 256]
            self.downscale_factor = [2, 4, 8] 
            self.fpn_depth_predictor = nn.ModuleList([
                DepthPredictorMultiView(
                    feature_channels=df,
                    upscale_factor=ds,
                    num_depth_candidates=cfg.num_depth_candidates,
                    costvolume_unet_feat_dim=df, 
                    costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
                    costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
                    gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
                    gaussians_per_pixel=cfg.gaussians_per_pixel,
                    num_views=get_cfg().dataset.view_sampler.num_context_views,
                    depth_unet_feat_dim=cfg.depth_unet_feat_dim,
                    depth_unet_attn_res=cfg.depth_unet_attn_res,
                    depth_unet_channel_mult=cfg.depth_unet_channel_mult,
                    wo_depth_refine=cfg.wo_depth_refine,
                    wo_cost_volume=cfg.wo_cost_volume,
                    wo_cost_volume_refine=cfg.wo_cost_volume_refine,)  
                for df, ds in zip(self.d_feature, self.downscale_factor)
            ])

    def data_process(self, images): 
        """  b v c h w -> b, 1, h, w,  range [0, 1] """
        assert images.shape[1] == 2   # 2 VIEWS
        img0, img1 = images[:,0], images[:,1]

        to_gary = tf.Grayscale()
        img0_gray, img_gray = to_gary(img0), to_gary(img1)  # b 1 h w      
        data = {'image0': img0_gray, 'image1': img_gray}
        return data

    def get_trans_cnn_feature(self, x, y):
        # trans
        b, v, _, _, _ = x[0].shape
        x = [rearrange(i, "b v c h w -> (b v) c h w", b=b, v=v) for i in x]
        x_d1, x_d2, x_d4 = x 
        x2 = F.interpolate(x_d1, scale_factor=0.5, mode='bilinear', align_corners=False)
        x2 = self.trans_64(x2)          # bv 64 d2
        x4 = F.interpolate(x_d2, scale_factor=0.5, mode='bilinear', align_corners=False)
        x4 = self.trans_128(x4)         # bv 128 d4
        x8 = F.interpolate(x_d4, scale_factor=0.5, mode='bilinear', align_corners=False)
        x8 = self.trans_256(x8)         # bv 256 d8
        x = [x2, x4, x8]
        x = [rearrange(i, "(b v) c h w -> b v c h w", b=b, v=v) for i in x]

        # cnn
        y = [rearrange(i, "b v c h w -> (b v) c h w", b=b, v=v) for i in y]
        y_d2, y_d4, y_d8 = y 
        y2 = self.cnn_64(y_d2) 
        y4 = self.cnn_128(y_d4) 
        y8 = self.cnn_256(y_d8) 
        y = [y2, y4, y8]
        y = [rearrange(i, "(b v) c h w -> b v c h w", b=b, v=v) for i in y]
        
        return x, y

    def get_scale(self, x):
        b, v, _, _, _ = x.shape
        x = rearrange(x, "b v c h w -> b (v c h w)", b=b, v=v) 
        scale = self.translation_regressor(x)
        return scale

    def map_pdf_to_opacity(
            self,
            pdf: Float[Tensor, " *batch"],
            global_step: int,
        ) -> Float[Tensor, " *batch"]:
            # https://www.desmos.com/calculator/opvwti3ba9

            # Figure out the exponent.
            cfg = self.cfg.opacity_mapping
            x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
            exponent = 2**x

            # Map the probability density to an opacity.
            return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def convert_fd_to_gaussians(self, h,w, context, depths, 
                raw_gaussians, densities,
                global_step, device
                ):
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",),(h, w),)
        return gaussians

    def forward(
        self,
        batch: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        ) :
        context = batch["context"]
        b, v, _, h, w = context["image"].shape      # 224, 320
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k in context:
            if context[k].device != device:
                context[k] = context[k].to(device)

        data = self.data_process(context["image"])  # input size must be divides by 32
        fpn_features, cnn_features = self.matcher(data, self.return_cnn_features, self.cfg.wo_fpn_depth)
        trans_features, cnn_features = self.get_trans_cnn_feature(fpn_features, cnn_features)
        # d2/d4/d8
        # torch.Size([1, 2, 64, 128, 128]) torch.Size([1, 2, 128, 64, 64]) torch.Size([1, 2, 256, 32, 32])


        conf_mask = data['mconf'] >= 0.5
        batch["mkpts0"], batch["mkpts1"], batch["mconf"], batch['mbids'] = \
            data['mkpts0_f'][conf_mask], data['mkpts1_f'][conf_mask], data['mconf'][conf_mask], data['m_bids'][conf_mask]

        # batch["mkpts0"], batch["mkpts1"], batch["mconf"], batch['mbids'] = \
        #     data["mkpts0_f"], data["mkpts1_f"], data["mconf"], data['m_bids']
        
        pred_scales = self.get_scale(trans_features[1]) # b,3
        batch['pred_scale'] = pred_scales

        # Sample depths from the resulting features.
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel

        if self.cfg.wo_fpn_depth: # single layer, [b,v,128,56,60]
            in_feats = trans_features[1] 
            depths, densities, raw_gaussians = self.depth_predictor(
                in_feats,
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                gaussians_per_pixel=gpp,
                deterministic=deterministic,
                extra_info=extra_info,
                cnn_features=cnn_features[1],
                wo_fpn_depth=self.cfg.wo_fpn_depth,
                batch=batch,
            )
            batch["context"]["est_depth"] = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)

            gaussians = self.convert_fd_to_gaussians(h,w, context, depths, 
                    raw_gaussians, densities, global_step, device)
            
            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump["depth"] = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                )
                visualization_dump["scales"] = rearrange(
                    gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
                )
                visualization_dump["rotations"] = rearrange(
                    gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                )

            # Optionally apply a per-pixel opacity.
            opacity_multiplier = 1

            return Gaussians(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    opacity_multiplier * gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
    
        else: # input d2/d4/d8, output d1/d2/d4   depth need constraint
            fpn_gaussians= []
            for in_feat, cnn_feat, depth_predictor, downscale_factor \
                in zip(trans_features, cnn_features, self.fpn_depth_predictor, self.downscale_factor):
                depths, densities, raw_gaussians = depth_predictor(
                    in_feat,
                    context["intrinsics"],
                    context["extrinsics"],
                    context["near"],
                    context["far"],
                    gaussians_per_pixel=gpp,
                    deterministic=deterministic,
                    extra_info=extra_info,
                    cnn_features=cnn_feat,
                    wo_fpn_depth=self.cfg.wo_fpn_depth)
                b, v, _, h, w = context["image"].shape 
                h, w = int(h / downscale_factor *2),  int(w / downscale_factor *2)
                
                # here depth can be fusioned.
                gaussians = self.convert_fd_to_gaussians(h,w, context, depths, raw_gaussians, densities, global_step, device)
            
                # Dump visualizations if needed.
                if visualization_dump is not None:
                    idx = 'P' + str(int(downscale_factor))
                    visualization_dump[idx] = {}
                    visualization_dump[idx]["depth"] = rearrange(
                        depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                    )
                    visualization_dump[idx]["scales"] = rearrange(
                        gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
                    )
                    visualization_dump[idx]["rotations"] = rearrange(
                        gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                    )
                # Optionally apply a per-pixel opacity.
                opacity_multiplier = 1

                fpn_gaussians.append(Gaussians(
                    rearrange(
                        gaussians.means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    rearrange(
                        gaussians.covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    rearrange(
                        gaussians.harmonics,
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                    ),
                    rearrange(
                        opacity_multiplier * gaussians.opacities,
                        "b v r srf spp -> b (v r srf spp)",
                    ),
                ))
            return fpn_gaussians


    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None



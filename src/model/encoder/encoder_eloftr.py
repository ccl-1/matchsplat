import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.transforms as tf
from einops.einops import rearrange, repeat
from jaxtyping import Float
from collections import OrderedDict

from dataclasses import dataclass
from typing import Literal, Optional, List

# loftr
from .loftr.loftr import LoFTR, reparameter
from .loftr.utils.plotting import make_matching_figure

# dataset
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians

from .encoder import Encoder
from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding

from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
# from .costvolume.get_depth_fpn import DepthPredictorMultiView
from .costvolume.get_depth import DepthPredictorMultiView


from .visualization.encoder_visualizer_eloftr_cfg import EncoderVisualizerELoftrCfg

from src.visualization.vis_depth import viz_depth_tensor
from PIL import Image
import numpy as np
from .pose.pose_estimation import run_weighted_8_point

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int

def get_zoe_depth(zoe, imgs, vis= False):
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



def MLP(channels: list, do_bn=True, do_leaky=False, last_layer=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1 if last_layer else n):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            if do_leaky:
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ConfidenceMLP(nn.Module):
    def __init__(self, feature_dim, in_dim, out_dim=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.layers_f = MLP([feature_dim * 2, feature_dim * 2, feature_dim], last_layer=False)
        self.layers_c = MLP([in_dim, feature_dim, feature_dim], last_layer=False)
        self.layers = MLP([feature_dim, out_dim])

        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(self, desc0, desc1, mconf):
        """
            desc: [b, c, n]
            mconf: [b, 1, n]
        """
        inputs = torch.cat([desc0, desc1], dim=-2)
        out_f = self.layers_f(inputs)
        out_c = self.layers_c(mconf)
        return torch.sigmoid(self.layers(out_f + out_c))
        # inputs = torch.cat(desc[:-1], dim=-2)
        # out_f = self.layers_f(inputs)
        # out_c = self.layers_c(desc[-1])
        # return torch.sigmoid(self.layers(out_f + out_c))


class PosePrediction(nn.Module):
    def __init__(self):
        super(PosePrediction, self).__init__()
        
        self.pose_cnn = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU(),
        )
        
        self.pose_regressor = nn.Sequential(
            nn.Linear(32*64*64, 512 ),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 2),
            nn.ReLU(),
        )
        self.rotation_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
        self.translation_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    
    def r6d2mat(self,d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalisation per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
                first two rows of the rotation matrix. 
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row
    
    def forward(self, x):
        b, v, c, h, w = x.size()
        x = rearrange(x, "b v c h w -> (b v) c h w", b=b, v=v)
        x = self.pose_cnn(x)
        x = rearrange(x, "(b v) c h w -> (b v) (c h w)", b=b, v=v)
        pose_latent_ctxt = self.pose_regressor(x)[:,:128]
        rot_ctxt, tran_ctxt = self.rotation_regressor(pose_latent_ctxt), self.translation_regressor(pose_latent_ctxt) # B x n_views x 9, B x n_views x 3 
        R_ctxt = self.r6d2mat(rot_ctxt)[:, :3, :3] 
        pose_mlp = torch.cat((torch.cat((R_ctxt, tran_ctxt.unsqueeze(-1)), dim=-1), 
                torch.FloatTensor([0,0,0,1]).expand(b*v, 1,-1).to(tran_ctxt.device)), dim=1)
        pose_mlp = rearrange(pose_mlp, "(b v) h w -> b v h w", b=b, v=v)
        return pose_mlp

        

@dataclass
class EncoderELoFTRCfg:
    name: Literal["eloftr"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerELoftrCfg

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
    est_pose: str | None
    use_scale: str | None
    use_fpn:bool



class EncoderELoFTR(Encoder[EncoderELoFTRCfg]):
    backbone: LoFTR
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderELoFTRCfg, backbone_cfg) -> None:
        super().__init__(cfg)
        self.config = backbone_cfg
        self.return_cnn_features = True
        # self.use_zoe_loss = False

        # if self.use_zoe_loss:
        #     print("==> Load ZoeDepth model ")
        #     repo = "isl-org/ZoeDepth"
        #     self.zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True).cuda()
        self.get_pose = PosePrediction()
        self.matcher = LoFTR(backbone_cfg)
        ckpt_path = cfg.eloftr_weights_path
        if cfg.eloftr_weights_path is None:
            print("==> Init E-loFTR backbone from scratch")
        else:
            print("==> Load E-loFTR backbone checkpoint: %s" % ckpt_path)
            self.matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
            self.matcher = reparameter(self.matcher) # no reparameterization will lead to low performance
            # if precision == 'fp16':
            #     encoder = self.matcher.half()

        self.conv_1x1_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_1x1_2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_1x1_3 = nn.Conv2d(128, 64, kernel_size=1)

        self.conv_3x3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_3x3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_3x3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.scale_conv = nn.Sequential(nn.Conv2d(128, 32, kernel_size=1, stride=1), nn.ReLU())
        self.scale_regressor = nn.Sequential( # input(B, N)
            nn.Linear(32*64*64*2, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        # confidence prediction for pose estimation
        if self.cfg.est_pose == 'w8-point':
            conf_feat_dim = 256
            conf_dim = 1
            self.conf_mlp = ConfidenceMLP(conf_feat_dim, conf_dim)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # TODO BA based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=[256, 128, 64] if cfg.use_fpn else cfg.d_feature, 
            upscale_factor=[8, 4, 2] if cfg.use_fpn else cfg.downscale_factor , 
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=[256, 128, 64] if cfg.use_fpn else cfg.costvolume_unet_feat_dim, 

            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2), # 1 * ((7 + 3 * self.d_sh) + 2) d_sh=(4+1)**2
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )


    def data_process(self, images): 
        """  b v c h w -> b, 1, h, w,  range [0, 1]? """
        assert images.shape[1] == 2   # 2 VIEWS
        img0, img1 = images[:,0], images[:,1]

        to_gary = tf.Grayscale()
        img0_gray, img_gray = to_gary(img0), to_gary(img1)  # b 1 h w      
        data = {'image0': img0_gray, 'image1': img_gray}
        return data

    def get_scale(self, x):
        b, v, _, _, _ = x.shape
        x = rearrange(x, "b v c h w -> (b v) c h w", b=b, v=v)
        x = self.scale_conv(x)
        x = rearrange(x, "(b v) c h w -> b (v c h w)", b=b, v=v)
        scale = self.scale_regressor(x)
        return scale

    def get_fpn_trans_features(self, trans_features): 
        b, v, _, _, _ = trans_features.shape
        trans_features = rearrange(trans_features, "b v c h w -> (b v) c h w", b=b, v=v)
        t1 = F.relu(self.conv_1x1_1(trans_features)) #[(bv), 256, h/8, w/8]
        t1 = F.relu(self.conv_3x3_1(t1))             #[(bv), 256, h/8, w/8]

        t2 = F.interpolate(t1, scale_factor=2., mode='bilinear', align_corners=False)
        t2 = F.relu(self.conv_1x1_2(t2))             #[(bv), 128, h/4, w/4]
        t2 = F.relu(self.conv_3x3_2(t2))             #[(bv), 128, h/4, w/4]

        t3 = F.interpolate(t2, scale_factor=2., mode='bilinear', align_corners=False)
        t3 = F.relu(self.conv_1x1_3(t3))             #[(bv), 64, h/2, w/2]
        t3 = F.relu(self.conv_3x3_3(t3))             #[(bv), 64, h/2, w/2]

        t1 = rearrange(t1, "(b v) c h w -> b v c h w", b=b, v=v)
        t2 = rearrange(t2, "(b v) c h w -> b v c h w", b=b, v=v)
        t3 = rearrange(t3, "(b v) c h w -> b v c h w", b=b, v=v)
        return [t1, t2, t3]


    def sample_descriptors(self, keypoints, descriptors, s: int = 8):
        """Interpolate descriptors at keypoint locations
            keypoints: [1, n, 2]
            descriptors: [1, c, h, w]
            return : [1, c, n]
        """
        b, c, h, w = descriptors.shape
        # print(keypoints[0,0])
        keypoints = keypoints - s / 2.0 + 0.5
        keypoints /= torch.tensor(
            [(w * s - s / 2. - 0.5), (h * s - s / 2. - 0.5)],
        ).to(
            keypoints
        )[None]
        # print(keypoints[0,0] * 256)
        # input()

        # keypoints /= 256
        keypoints = keypoints * 2. - 1.  # normalize to (-1, 1)
        args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
        )
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1
        )
        return descriptors


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

    def test_loftr(
        self, 
        context: dict,
        global_step: int,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape      # 224, 320
        data = self.data_process(context["image"])  # input size must be divides by 32

        print ("input image shape", data['image0'].shape)
        trans_features, cnn_features = self.matcher(data, self.return_cnn_features)
        mkpts0, mkpts1, mconf = data['mkpts0_f'], data['mkpts1_f'], data['mconf']

        print ("mkpts0.shape", mkpts0.shape)

    def forward(
        self,
        batch: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        context = batch["context"]
        device = context["image"].device
        b, v, _, h, w = context["image"].shape      # 224, 320
        data = self.data_process(context["image"])  # input size must be divides by 32
        
        """
            trans_feature: [b, v, 256, h/8, w/8]
            cnn_features: [b, v, 256, h/8, w/8], [b, v, 128, h/4, w/4], [b, v, 64, h/2, w/2]
        """
        trans_features, cnn_features_list = self.matcher(data, self.return_cnn_features)    
        trans_features_list = self.get_fpn_trans_features(trans_features)

        if not self.cfg.use_fpn:
            in_feats = trans_features_list[1]
            cnn_feats = cnn_features_list[1]
        else:
            cnn_feats = cnn_features_list
            in_feats = trans_features_list
        
        pred_scales = self.get_scale(trans_features_list[1]) # b,1
        batch['pred_scale'] = pred_scales
   
        # conf 训练动态变化，所以不能直接用来过滤， conf_mask 过滤会导致最终没有点用， render depth 出现 nan ...  
        batch["mkpts0"], batch["mkpts1"], batch["mconf"], batch['mbids'] = \
            data["mkpts0_f"], data["mkpts1_f"], data["mconf"], data['m_bids']

        if self.cfg.est_pose == 'mlp':
            extrinsics_est = self.get_pose(trans_features_list[1])
            batch['pose_mlp'] = extrinsics_est
            
        else:
            #  Weights decoder for weighted 8-point algorithm
            # only affects the training phase, maybe ... 
            if self.cfg.est_pose == 'w8-point':
                weights_list = []
                for i in range(b):
                    b_mask = batch['mbids'] == i
                    mkpts0 = batch["mkpts0"][b_mask][None] # [1, n, 2]
                    mkpts1 = batch["mkpts1"][b_mask][None] # [1, n, 2]
                    mconf = batch["mconf"][b_mask].view(1, 1, -1) # [1, 1, n]

                    # print ("mconf before:", batch["mconf"][b_mask])
                    feat0 = trans_features[i][0][None] # [1, 256, h/8, w/8]
                    feat1 = trans_features[i][1][None] # [1, 256, h/8, w/8]
                    # print(feat0.shape)

                    # sample descriptors from trans features
                    desc0 = self.sample_descriptors(mkpts0, feat0) # [1, 256, n]
                    desc1 = self.sample_descriptors(mkpts1, feat1) # [1, 256, n]

                    # go through conf mlp for decoding the final weights
                    # weights = self.conf_mlp(desc0, desc1, mconf) # [1, 1, n]

                    mconf = self.conf_mlp(desc0, desc1, mconf).squeeze()
                    # print ("mconf after:", mconf)
                    weights_list.append(mconf)

                weights = torch.cat(weights_list, dim=0)
                batch["mconf"] = weights

            # Pose estimation from the sparse matched keypoints
            extrinsics_est, pose_eval_dict = run_weighted_8_point(batch, use_scale=self.cfg.use_scale)
            batch.update(pose_eval_dict)

        # print('after:', batch["mconf"][0])
        # batch['rel_pose'] =  rearrange(context["extrinsics"], "b v x y -> (b v) x y")
        # batch['gt_rel_pose'] = rearrange(extrinsics_est, "b v x y -> (b v) x y")
        
        # Sample depths from the resulting features.
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel

        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"] if not self.cfg.est_pose  else extrinsics_est,  
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_feats,
            batch=batch,
        )
        """
        # depths (b, v, 65536, 1, 1)  [srf, dpt]
        # densities (b, v, 65536, 1, 1)
        # raw_gaussians (b, v, 65536, 84)  1 * ((7 + 3 * self.d_sh) + 2) d_sh=(4+1)**2 4->sh_degree
        """
        batch["context"]["est_depth"] = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)
        
        # if self.use_zoe_loss:
        #     zoe_depths = get_zoe_depth(self.zoe, context["image"], vis=False).to(densities.device) # b v 1 h w        
        #     batch["context"]['zoe_depth'] =  rearrange(zoe_depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)

        # save depth result to compare
        # vis_depth = False
        # if vis_depth:
        #     near, far = 0.0, 100.0
        #     depth_vis = batch["context"]["est_depth"].squeeze(-1).squeeze(-1).cpu().detach()
        #     zoe_depth_vis = batch["context"]['zoe_depth'].squeeze(-1).squeeze(-1).cpu().detach()
        #     for v_idx in range(depth_vis.shape[1]):
        #         depth_vis = np.clip(depth_vis, near, far)
        #         vis_depth = viz_depth_tensor(1.0 / depth_vis[0, v_idx], return_numpy=True)  # inverse depth
        #         Image.fromarray(vis_depth).save(f"outputs/tmp/depth/pred_{v_idx}.png")
        #         vis_depth = viz_depth_tensor(1.0 / zoe_depth_vis[0, v_idx], return_numpy=True)  # inverse depth
        #         Image.fromarray(vis_depth).save(f"outputs/tmp/depth/zoe_{v_idx}.png")
        #         print(depth_vis[0, v_idx])
        #         print(zoe_depth_vis[0, v_idx]) 
        #     input()

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy") # (65536, 1, 2)
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        ) # (b, v, 65536, 1, 84)
        offset_xy = gaussians[..., :2].sigmoid() #[b, v, 65536, 1, 2]
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size #[b, v, 65536, 1, 2]
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

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



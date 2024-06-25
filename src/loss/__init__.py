from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_pose import LossPose, LossPoseCfgWrapper
from .loss_ssim import LossSSIM, LossSSIMCfgWrapper
from .loss_corres import LossCorres, LossCorresCfgWrapper
from .loss_scale import LossScale, LossScaleCfgWrapper
from .loss_zoe import LossZoe, LossZoeCfgWrapper
from .loss_point import LossPoint, LossPointCfgWrapper


LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossPoseCfgWrapper: LossPose,
    LossSSIMCfgWrapper: LossSSIM,
    LossCorresCfgWrapper: LossCorres,
    LossScaleCfgWrapper: LossScale,
    LossZoeCfgWrapper: LossZoe,
    LossPointCfgWrapper: LossPoint,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper \
                | LossPoseCfgWrapper | LossSSIMCfgWrapper | LossCorresCfgWrapper \
                |LossScaleCfgWrapper | LossZoeCfgWrapper | LossPointCfgWrapper \


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]

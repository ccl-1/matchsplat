
import warnings
import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from src.model.encoder.encoder_costvolume import EncoderCostVolume


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.model.decoder import get_decoder
    from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA,DecoderSplattingCUDACfg
    


def ckpt_load(encoder, checkpoint_path):
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
    config_name="main",
)
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    dataset = iter(data_module.train_dataloader())
    batch = next(dataset, 0)

    # load model
    encoder = EncoderCostVolume(cfg.model.encoder)
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)

    # checkpoint_path = "checkpoints/re10k.ckpt"
    # new_checkpoint_path = "checkpoints/re10k_new.ckpt"

    # ckpt_new = ckpt_load(encoder, checkpoint_path)
    # encoder.load_state_dict(ckpt_new,strict=False) 
    # torch.save(encoder.state_dict(), new_checkpoint_path)
    
    # run model
    # gaussians = encoder(batch["context"], 0, False, scene_names=batch["scene"])

    visualization_dump = {}
    Gaussians = encoder(batch["context"], 0, False, visualization_dump=visualization_dump, scene_names=batch["scene"])
    # decoder = get_decoder(cfg.model.decoder, cfg.dataset)

 


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    run()
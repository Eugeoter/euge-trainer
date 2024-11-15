from absl import app, flags
from ml_collections import config_flags


def main(argv):
    config = flags.FLAGS.config
    trainer = flags.FLAGS.trainer
    if trainer == "sd15":
        from modules.trainers.sd15_trainer import SD15Trainer
        trainer_class = SD15Trainer
    elif trainer == "sdxl":
        from modules.trainers.sdxl_trainer import SDXLTrainer
        trainer_class = SDXLTrainer
    elif trainer == 'sd3':
        from modules.trainers.sd3_trainer import SD3Trainer
        trainer_class = SD3Trainer
    elif trainer == 'flux':
        from modules.trainers.flux_trainer import FluxTrainer
        trainer_class = FluxTrainer
    elif trainer == 'hunyuan':
        from modules.trainers.hunyuan_trainer import HunyuanTrainer
        trainer_class = HunyuanTrainer
    elif trainer == 'sd15_controlnet':
        from modules.trainers.sd15_controlnet_trainer import SD15ControlNetTrainer
        trainer_class = SD15ControlNetTrainer
    elif trainer == 'sd15_controlnext':
        from modules.trainers.sd15_controlnext_trainer import SD15ControlNeXtTrainer
        trainer_class = SD15ControlNeXtTrainer
    elif trainer == 'sd15_lora_adapter':
        from modules.trainers.sd15_lora_adapter_trainer import SD15LoraAdapterTrainer
        trainer_class = SD15LoraAdapterTrainer
    elif trainer == 'sdxl_controlnet':
        from modules.trainers.sdxl_controlnet_trainer import SDXLControlNetTrainer
        trainer_class = SDXLControlNetTrainer
    elif trainer == 'sdxl_controlnext':
        from modules.trainers.sdxl_controlnext_trainer import SDXLControlNeXtTrainer
        trainer_class = SDXLControlNeXtTrainer
    elif trainer == 'sdxl_distill':
        from modules.trainers.sdxl_distill_trainer import SDXLDistillTrainer
        trainer_class = SDXLDistillTrainer
    elif trainer == 'hunyuan_controlnext':
        from modules.trainers.hunyuan_controlnext_trainer import HunyuanControlNeXtTrainer
        trainer_class = HunyuanControlNeXtTrainer
    elif trainer == 'flux_controlnext':
        from modules.trainers.flux_controlnext_trainer import FluxControlNeXtTrainer
        trainer_class = FluxControlNeXtTrainer
    elif trainer == 'vae':
        from modules.trainers.vae_trainer import VAETrainer
        trainer_class = VAETrainer
    elif trainer == 'ws':
        from modules.trainers.ws_trainer import WaifuScorerTrainer
        trainer_class = WaifuScorerTrainer
    else:
        raise ValueError(f"Invalid trainer: {trainer}")

    trainer = trainer_class.from_config(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
    flags.mark_flags_as_required(["config"])
    flags.DEFINE_string("trainer", "sd15", "Trainer class name.")
    app.run(main)

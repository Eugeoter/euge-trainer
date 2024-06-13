from absl import app, flags
from ml_collections import config_flags
from modules.trainers.sdxl_trainer import SDXLT2ITrainer


def main(argv):
    config = flags.FLAGS.config
    trainer = SDXLT2ITrainer.from_config(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
    flags.mark_flags_as_required(["config"])
    app.run(main)

from pytorch_lightning import seed_everything
from pytorch_pfn_extras.config import Config
from src.policy.pointernet.datamodule import TSPDataModule
from src.policy.pointernet.modelmodule import ActorCriticModule
from src.utils import load_config
import wandb
import click
from pathlib import Path


def train(cfg: Config):
    wandb.login()
    seed_everything(cfg["/seed"])
    output_dir = Path(cfg["/output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticModule(cfg)
    datamodule = TSPDataModule(cfg)
    trainer = cfg["/trainer"]
    trainer.fit(model, datamodule)


@click.command()
@click.option("--config_path", "-c", default="yml/pointernet/exp001.yml")
@click.option("--default_config_path", "-d", default="yml/pointernet/default.yml")
def main(config_path, default_config_path):
    cfg = load_config(config_path, default_config_path)
    train(cfg)


if __name__ == "__main__":
    main()

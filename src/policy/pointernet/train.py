from pytorch_lightning import LightningModule, seed_everything
from pytorch_pfn_extras.config import Config
import torch.nn as nn
import torch
from src.policy.pointernet.datamodule import TSPDataModule
from src.config import load_config
import wandb
import click


class ActorCriticModule(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.actor = cfg["/actor_model"]
        self.critic = cfg["/critic_model"]
        self.mse = nn.MSELoss()
        self.automatic_optimization = False

    def forward(self, x):
        return self.actor(x)

    def training_step(self, batch, batch_idx):
        self.__share_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.__share_step(batch, "val")

    def __share_step(self, batch, mode):
        x = batch
        pred_tour, log_prob = self(x)

        # critic loss
        target_dist = self.__calc_tour_dist(x, pred_tour)
        pred_dist = self.critic(x).squeeze()
        critic_loss = self.mse(pred_dist, target_dist.detach())

        # actor loss
        adv = target_dist.detach() - pred_dist.detach()
        actor_loss = (log_prob * adv).mean()

        # optimization
        if mode == "train":
            actor_opt, critic_opt = self.optimizers()
            actor_scheduler, critic_scheduler = self.lr_schedulers()
            # actor
            actor_opt.zero_grad()
            self.manual_backward(actor_loss)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["/clip_grad_norm"])
            actor_opt.step()
            actor_scheduler.step()
            # critic
            critic_opt.zero_grad()
            self.manual_backward(critic_loss)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["/clip_grad_norm"])
            critic_opt.step()
            critic_scheduler.step()

        # logging
        self.log_dict(
            {
                f"{mode}_actor_loss": actor_loss,
                f"{mode}_critic_loss": critic_loss,
                f"{mode}_tour_dist": target_dist.mean(),
            },
            prog_bar=True if mode == "val" else False,
            on_step=False if mode == "val" else True,
            on_epoch=True if mode == "val" else False,
        )

    def __calc_tour_dist(self, x, tour):
        d = torch.gather(x, dim=1, index=tour[:, :, None].repeat(1, 1, 2))
        d1 = torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)  # first node to last node
        d2 = (d[:, 0] - d[:, -1]).norm(p=2, dim=1)  # last node to first node
        return (d1 + d2).squeeze()

    def configure_optimizers(self):
        actor_opt = self.cfg["/actor_optimizer"]
        critic_opt = self.cfg["/critic_optimizer"]
        actor_scheduler = self.cfg["/actor_scheduler"]
        critic_scheduler = self.cfg["/critic_scheduler"]

        optimizers = [actor_opt, critic_opt]
        schedulers = [actor_scheduler, critic_scheduler]

        return optimizers, schedulers


def train(cfg):
    wandb.login()
    seed_everything(cfg["/seed"])
    model = ActorCriticModule(cfg)
    datamodule = TSPDataModule(cfg)
    trainer = cfg["/trainer"]
    trainer.fit(model, datamodule)


@click.command()
@click.option("--config_path", "-c", default="yml/test.yml")
@click.option("--default_config_path", "-d", default="yml/test_default.yml")
def main(config_path, default_config_path):
    cfg = load_config(config_path, default_config_path)
    train(cfg)


if __name__ == "__main__":
    main()

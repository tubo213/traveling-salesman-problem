from pytorch_lightning import LightningModule
from pytorch_pfn_extras.config import Config
import torch.nn as nn
import torch


class ActorCriticModule(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
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
        pred_tour, log_ll = self(x)

        # critic loss
        target_dist = self.__calc_tour_dist(x, pred_tour)
        pred_dist = self.critic(x).squeeze()
        critic_loss = self.mse(pred_dist, target_dist.detach())

        # critic optimization
        if mode == "train":
            critic_opt = self.optimizers()[1]
            critic_scheduler = self.lr_schedulers()[1]
            critic_opt.zero_grad()
            self.manual_backward(critic_loss)
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg["/clip_grad_norm"], norm_type=2
            )
            critic_opt.step()
            critic_scheduler.step()

        # actor loss
        adv = (target_dist - pred_dist).squeeze().detach()
        actor_loss = (log_ll * adv).mean()

        # actor optimization
        if mode == "train":
            actor_opt = self.optimizers()[0]
            actor_scheduler = self.lr_schedulers()[0]
            # actor
            actor_opt.zero_grad()
            self.manual_backward(actor_loss)
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg["/clip_grad_norm"], norm_type=2
            )
            actor_opt.step()
            actor_scheduler.step()

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
        idx = torch.arange(x.shape[0]).to(x.device)
        x_left = x[idx[:, None], tour[:, :-1]]
        x_right = x[idx[:, None], tour[:, 1:]]
        x_start = x[idx, tour[:, 0]]
        x_end = x[idx, tour[:, -1]]

        d_start_end = torch.norm(x_left - x_right, dim=-1, p=2).sum(
            dim=-1
        )  # start to end distance
        d_end_start = torch.norm(x_end - x_start, dim=-1, p=2)  # end to start distance

        return d_start_end + d_end_start

    def configure_optimizers(self):
        actor_opt = self.cfg["/actor_optimizer"]
        critic_opt = self.cfg["/critic_optimizer"]
        actor_scheduler = self.cfg["/actor_scheduler"]
        critic_scheduler = self.cfg["/critic_scheduler"]
        optimizers = [actor_opt, critic_opt]
        schedulers = [actor_scheduler, critic_scheduler]

        return optimizers, schedulers

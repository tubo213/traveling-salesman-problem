from pytorch_lightning import LightningDataModule
from pytorch_pfn_extras.config import Config


class TSPDataModule(LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        return self.cfg["/train_dataloader"]

    def val_dataloader(self):
        return self.cfg["/val_dataloader"]

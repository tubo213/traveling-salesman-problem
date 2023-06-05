from pytorch_lightning import LightningDataModule
from pytorch_pfn_extras.config import Config
from src.generator import Generator
from src.dataset import TSPDataset
from torch.utils.data import DataLoader


class TSPDataModule(LightningDataModule):
    def __init__(self, cfg: Config, generator: Generator):
        super().__init__()
        self.cfg = cfg
        self.generator = generator

    def train_dataloader(self):
        data = self.generator.get_data(
            num_samples=self.cfg["/train_num_samples"],
            seed=self.cfg["/seed"] + 1,
        )
        train_ds = TSPDataset(data)
        return DataLoader(
            train_ds,
            batch_size=self.cfg["/batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg["/num_workers"],
        )

    def val_dataloader(self):
        data = self.generator.get_data(
            num_samples=self.cfg["/val_num_samples"],
            seed=self.cfg["/seed"] + 2,
        )
        val_ds = TSPDataset(data)
        return DataLoader(
            val_ds,
            batch_size=self.cfg["/batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg["/num_workers"],
        )

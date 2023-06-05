from generator import Generator
from src.policy.pointernet.dataset import TSPDataset
from src.policy.pointernet.datamodule import TSPDataModule
from src.policy.pointernet.model import PointerNet, Critic
import torch.optim as optim
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    Adafactor,
)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

CONFIG_TYPES = {
    # utils
    "__len__": lambda obj: len(obj),
    "multiply": lambda a, b: a * b,
    "method_call": lambda obj, method: getattr(obj, method)(),
    # Generator
    "Generator": Generator,
    # Dataset
    "TSPDataset": TSPDataset,
    # DataModule
    "TSPDataModule": TSPDataModule,
    # Optimizer
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "Adafactor": Adafactor,
    # Model
    "PointerNet": PointerNet,
    "Critic": Critic,
    # Scheduler
    "CosineWithWarmup": get_cosine_schedule_with_warmup,
    "LinearWithWarmup": get_linear_schedule_with_warmup,
    # logger
    "WandbLogger": WandbLogger,
    # Callback
    "ModelCheckpoint": ModelCheckpoint,
    "LearningRateMonitor": LearningRateMonitor,
    # Trainer
    "Trainer": Trainer,
}

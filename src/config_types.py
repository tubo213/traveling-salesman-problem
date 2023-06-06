from src.generator import Generator
from src.policy.pointernet.dataset import TSPDataset
from src.policy.pointernet.datamodule import TSPDataModule
from src.policy.pointernet.model import (
    PointerNet,
    PointerNetCritic,
    TransformerCritic,
)
import torch.optim as optim
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    Adafactor,
)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from src.policy import (
    TwoOptPolicy,
    ThreeOptPolicy,
    RandomPolicy,
    PointerNetPolicy,
    GreedyPolicy,
)


CONFIG_TYPES = {
    # utils
    "__len__": lambda obj: len(obj),
    "multiply": lambda a, b: a * b,
    "add": lambda a, b: a + b,
    "method_call": lambda obj, method: getattr(obj, method)(),
    # Policies
    "RandomPolicy": RandomPolicy,
    "GreedyPolicy": GreedyPolicy,
    "TwoOptPolicy": TwoOptPolicy,
    "ThreeOptPolicy": ThreeOptPolicy,
    "PointerNetPolicy": PointerNetPolicy,
    # For PointerNet
    # Generator
    "Generator": Generator,
    # Dataset
    "TSPDataset": TSPDataset,
    # DataLoader
    "DataLoader": DataLoader,
    # DataModule
    "TSPDataModule": TSPDataModule,
    # Optimizer
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "Adafactor": Adafactor,
    # Model
    "PointerNet": PointerNet,
    "PointerNetCritic": PointerNetCritic,
    "TransformerCritic": TransformerCritic,
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

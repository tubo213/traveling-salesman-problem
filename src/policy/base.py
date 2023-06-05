from abc import ABCMeta, abstractmethod
import numpy as np
from pytorch_lightning import seed_everything


class BasePolicy(metaclass=ABCMeta):
    def __init__(self, seed=0):
        seed_everything(seed)

    @abstractmethod
    def solve(self, x: np.ndarray):
        """
        x: [N, seq_len, input_size]
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

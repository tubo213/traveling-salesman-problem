from src.policy.base import BasePolicy
import numpy as np


class SolverPolicy(BasePolicy):
    def __init__(self, seed):
        super().__init__(seed)

    def solve(self, x: np.ndarray):
        pass

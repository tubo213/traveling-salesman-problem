from src.policy.base import BasePolicy
from src.policy.greedy import GreedyPolicy
import numpy as np
from abc import abstractmethod
from joblib import Parallel, delayed
from typing import Optional


class BaseAnnealing(BasePolicy):
    def __init__(
        self, timelimit: float, init_policy: Optional[BasePolicy] = GreedyPolicy(), seed=0
    ):
        super().__init__(seed)
        self.timelimit = timelimit
        self.init_policy = init_policy

    def solve(self, x):
        init_tour = self.init_policy.solve(x)
        # solve in parallel
        results = Parallel(n_jobs=-1)(
            [
                delayed(self.solve_sample)(x_i, init_tour_i)
                for x_i, init_tour_i in zip(x, init_tour(x))
            ]
        )
        return np.array(results)

    def solve_sample(self, x_i, init_tour_i):
        tour = init_tour_i
        for _ in range(self.num_iteration):
            tour = self.solve_sample_once(x_i, tour)

        return tour

    @abstractmethod
    def update(self, x_i, tour_i):
        raise NotImplementedError

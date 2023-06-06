from src.policy.base import BasePolicy
from src.policy.greedy import GreedyPolicy
import numpy as np
from abc import abstractmethod
from joblib import Parallel, delayed
from typing import Optional
import time
from src.utils import calc_score


def transition_prob(crr_score, next_score, tmp):
    delta = crr_score - next_score
    return np.exp(min(0, (delta / tmp)))


def calc_score_i(tour_i, x_i):
    score = calc_score(tour_i[None, :], x_i[None, :])

    return score


class BaseAnnealing(BasePolicy):
    def __init__(
        self,
        start_tmp,
        end_tmp,
        timelimit: float,
        init_policy: Optional[BasePolicy] = GreedyPolicy(),
        seed=0,
    ):
        super().__init__(seed)
        self.start_tmp = start_tmp
        self.end_tmp = end_tmp
        self.timelimit = timelimit
        self.init_policy = init_policy

    def solve(self, x):
        init_tour = self.init_policy.solve(x)
        # solve in parallel
        results = Parallel(n_jobs=-1)(
            [
                delayed(self.solve_sample)(x_i, init_tour_i)
                for x_i, init_tour_i in zip(x, init_tour)
            ]
        )
        return np.array(results)

    def solve_sample(self, x_i, init_tour_i):
        start_time = time.perf_counter()
        tour = init_tour_i
        crr_score = calc_score_i(tour, x_i)
        while True:
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > self.timelimit:
                break

            next_tour = self.update(x_i, tour)
            next_score = calc_score_i(next_tour, x_i)
            tmp = self.get_tmp(elapsed_time)
            prob = transition_prob(crr_score, next_score, tmp)

            if np.random.rand() < prob:
                tour = next_tour
                crr_score = next_score
        return tour

    @abstractmethod
    def update(self, x_i, tour_i):
        raise NotImplementedError

    def get_tmp(self, elapsed_time):
        return self.start_tmp + (self.end_tmp - self.start_tmp) * elapsed_time / self.timelimit

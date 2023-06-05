from src.policy.base import BasePolicy

import numpy as np


class RandomPolicy(BasePolicy):
    def __init__(self, seed=0):
        super().__init__(seed)

    def solve(self, x: np.ndarray):
        num_samples = x.shape[0]
        num_nodes = x.shape[1]
        tours = np.concatenate(
            [np.random.permutation(num_nodes)[None, :] for _ in range(num_samples)], axis=0
        )

        return tours


if __name__ == "__main__":
    random = RandomPolicy()
    x = np.random.rand(3, 25, 2)
    print(random.solve(x))

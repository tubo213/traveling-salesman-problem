import numpy as np

from src.policy.base import BasePolicy


class GreedyPolicy(BasePolicy):
    def __init__(self, seed=0):
        super().__init__(seed)

    def solve(self, x: np.ndarray):
        """
        x: [N, seq_len, input_size]
        """
        d1 = np.expand_dims(x, 1)  # [N, 1, seq_len, input_size]
        d2 = np.expand_dims(x, 2)  # [N, seq_len, 1, input_size]
        dist_mat = np.linalg.norm(d1 - d2, axis=3)  # [N, seq_len, seq_len]
        neighbors = np.argsort(dist_mat, axis=2)  # [N, seq_len, seq_len]

        tours = []
        for i in range(x.shape[0]):
            tour = [np.random.randint(x.shape[1])]
            for _ in range(x.shape[1] - 1):
                prev = tour[-1]
                for j in range(x.shape[1]):
                    if neighbors[i, prev, j] not in set(tour):
                        tour.append(neighbors[i, prev, j])
                        break
            tours.append(tour)
        return np.array(tours)


if __name__ == "__main__":
    greedy = GreedyPolicy()
    x = np.random.rand(3, 25, 2)
    print(greedy.solve(x))

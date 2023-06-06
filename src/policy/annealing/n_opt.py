from src.policy.annealing.base import BaseAnnealing
import numpy as np


class TwoOptPolicy(BaseAnnealing):
    def update(self, _, tour):
        next_tour = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        next_tour[i:j] = next_tour[i:j][::-1]
        return next_tour


class ThreeOptPolicy(BaseAnnealing):
    def update(self, _, tour):
        next_tour = tour.copy()
        i, j, k = np.random.choice(len(tour), 3, replace=False)
        next_tour[i:j] = next_tour[i:j][::-1]
        next_tour[j:k] = next_tour[j:k][::-1]
        return next_tour


if __name__ == "__main__":
    x = np.random.rand(10, 10, 2)
    policy = ThreeOptPolicy(start_tmp=1, end_tmp=0.1, timelimit=1)
    tour = policy.solve(x)
    from src.utils import calc_score
    print(calc_score(tour, x).mean())

    policy = TwoOptPolicy(start_tmp=1, end_tmp=0.1, timelimit=1)
    tour = policy.solve(x)
    print(calc_score(tour, x).mean())

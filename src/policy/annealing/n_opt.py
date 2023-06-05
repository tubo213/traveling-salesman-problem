from src.policy.annealing.base import BaseAnnealing


class TwoOptPolicy(BaseAnnealing):
    def update(self, x_i, tour_i):
        """
        x_i: [seq_len, input_size]
        tour_i: [seq_len]
        """
        return None

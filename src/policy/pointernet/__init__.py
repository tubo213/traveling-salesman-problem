from src.policy.pointernet import PointerNet
from src.policy.base import BasePolicy
import torch
import numpy as np


class PointerNetPolicy(BasePolicy):
    def __init__(self, model: PointerNet, ckpt_path=None):
        self.model = model
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))

    @torch.no_grad()
    def solve(self, x: np.ndarray):
        """
        x: [N, seq_len, input_size]
        """
        inputs = torch.FloatTensor(x).to(self.model.device)
        pred_tour, _ = self.model(inputs)
        pred_tour = pred_tour.cpu().numpy()
        return pred_tour

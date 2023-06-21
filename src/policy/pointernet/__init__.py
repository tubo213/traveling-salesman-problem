from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.policy.base import BasePolicy
from src.policy.pointernet.model import PointerNet


class PointerNetPolicy(BasePolicy):
    def __init__(self, model: PointerNet, ckpt_path: Optional[str], device: str = "cpu"):
        super().__init__(seed=0)
        self.model = model
        if ckpt_path is not None:
            self.model.load_state_dict(self._parse_state_dict(ckpt_path))
            self.ckpt_name = Path(ckpt_path).stem
        else:
            self.ckpt_name = ""
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def solve(self, x: np.ndarray):
        """
        x: [N, seq_len, input_size]
        """
        inputs = torch.FloatTensor(x).to(self.device)
        pred_tour, _ = self.model(inputs)
        pred_tour = pred_tour.cpu().numpy()
        return pred_tour

    def _parse_state_dict(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path)["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if "actor." in k:
                new_state_dict[k.replace("actor.", "")] = v
        return new_state_dict

    def __str__(self):
        class_name = self.__class__.__name__
        ckpt_name = self.ckpt_name
        return f"{class_name}_{ckpt_name}"

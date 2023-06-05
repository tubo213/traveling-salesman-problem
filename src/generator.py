import numpy as np
from pytorch_lightning import seed_everything


class Generator:
    def __init__(self, num_nodes, dim_node):
        self.num_nodes = num_nodes
        self.dim_node = dim_node

    def get_data(self, num_samples, seed=0):
        seed_everything(seed)
        return np.random.rand(num_samples, self.num_nodes, self.dim_node)

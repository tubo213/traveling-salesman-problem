import torch
from torch.utils.data import Dataset

from src.generator import Generator


class TSPDataset(Dataset):
    def __init__(self, num_nodes, dim_node, num_samples, seed=0):
        generator = Generator(num_nodes, dim_node)
        self.data = generator.get_data(num_samples, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])

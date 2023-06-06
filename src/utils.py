from dataclasses import dataclass
import numpy as np
from typing import List
from pytorch_pfn_extras.config import Config
import yaml
from typing import Optional
from src.config_types import CONFIG_TYPES
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(path: str, default_path: Optional[str] = None) -> Config:
    with open(path) as f:
        cfg: dict = yaml.safe_load(f)

    if default_path is not None:
        with open(default_path) as f:
            default_cfg: dict = yaml.safe_load(f)
        # merge default config
        for k, v in default_cfg.items():
            if k not in cfg:
                print(f"used default {k}: {v}")
                cfg[k] = v

    return Config(cfg, types=CONFIG_TYPES)  # type: ignore


@dataclass(frozen=True)
class ExpResult:
    policy_name: str
    tour: np.ndarray
    score: np.ndarray


def calc_score(tour: np.ndarray, x: np.ndarray):
    """
    x: [batch_size, num_nodes, 2]
    """
    # [batch size, num_nodes] -> [batch_size, num_nodes-1, 2]
    idx = np.arange(len(tour))
    x_left = x[idx[:, None], tour[:, :-1]]
    x_right = x[idx[:, None], tour[:, 1:]]
    x_start = x[idx, tour[:, 0]]
    x_end = x[idx, tour[:, -1]]

    d_left_right = np.sqrt((x_left - x_right) ** 2).sum(axis=2).sum(axis=1)
    d_start_end = np.sqrt((x_start - x_end) ** 2).sum(axis=1)

    return (d_left_right + d_start_end).squeeze()


def plot_results(
    results: List[ExpResult],
    x: np.ndarray,
    num_samples: int = 3,
    save_dir: Optional[Path] = None,
    plot_tour: bool = False,
):
    # plot samples
    fig, axes = plt.subplots(
        num_samples, len(results), figsize=(10 * len(results), 7 * num_samples)
    )
    sample_idx = np.random.randint(len(x), size=num_samples)
    for i, sample_id in enumerate(sample_idx):
        for j, result in enumerate(results):
            sample = x[sample_id]
            sample_tour = result.tour[sample_id]
            sample_score = result.score[sample_id]
            title = f"{result.policy_name} score={sample_score:.2f}"
            if plot_tour:
                sample_tour_str = ";".join([str(x) for x in sample_tour])
                title += f"\n tour: {sample_tour_str}"
            row = np.append(sample[sample_tour, 0], sample[sample_tour, 0][0])
            col = np.append(sample[sample_tour, 1], sample[sample_tour, 1][0])
            axes[i, j].plot(row, col, marker=".", markersize=10)
            axes[i, j].set_title(title, fontsize=30)

    if save_dir is not None:
        save_path = save_dir / "samples.png"
        fig.savefig(save_path, bbox_inches="tight")

    # plot score
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharey=True, sharex=True)
    for i, result in enumerate(results):
        label = f"{result.policy_name} score={result.score.mean():.2f}"
        ax.hist(result.score, label=label, alpha=0.7)
        ax.set_title("Score distribution", fontsize=30)
        ax.set_ylabel("Frequency", fontsize=20)
        ax.set_xlabel("Length of tour", fontsize=20)
        ax.legend()

    if save_dir is not None:
        save_path = save_dir / "score.png"
        fig.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    print(np.arange(10)[1:])
    print(np.arange(10)[:-1])

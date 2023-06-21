from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.policy import GreedyPolicy, RandomPolicy, TwoOptPolicy


def visualize_logs(x, logs, path_to_dir: Path, log_step=10):
    for step, (tour_i, score) in enumerate(tqdm(logs)):
        if step % log_step != 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            x_cood = x[tour_i, 0]
            y_cood = x[tour_i, 1]
            x_cood = np.append(x_cood, x_cood[0])
            y_cood = np.append(y_cood, y_cood[0])

            ax.plot(x_cood, y_cood, marker=".", markersize=7, label="city")
            ax.set_title(f"step: {step+1} 総距離: {score:.2f}", fontsize=15)
            ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])

            fig.savefig(
                path_to_dir / f"{step+1}.png",
            )
            plt.close()


def make_fig(paths: List[Path], save_path: Path, max_step=np.inf):
    imgs = []
    for i, path in enumerate(tqdm(paths)):
        imgs.append(Image.open(path))
        if i > max_step:
            break

    imgs[0].save(
        save_path,
        save_all=True,
        append_images=imgs[1:],
        duration=30,
        loop=0,
    )


@click.command()
@click.option("--path_to_dir", type=Path, default="logs/annealing")
def main(path_to_dir: Path):
    # policy = TwoOptPolicy(
    #     start_tmp=100,
    #     end_tmp=0.01,
    #     timelimit=1.0,
    # )
    # x = np.random.rand(100, 2)
    # init_tour = policy.init_policy.solve(x[None, :])[0]
    # best_tour, logs = policy.solve(x, init_tour, return_log=True)
    print(path_to_dir)


if __name__ == "__main__":
    main()

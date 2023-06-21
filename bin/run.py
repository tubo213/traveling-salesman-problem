import logging
from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt
from pytorch_pfn_extras.config import Config
from ttimer import get_timer

from src.config import load_config
from src.generator import Generator
from src.policy.base import BasePolicy
from src.utils import ExpResult, calc_score, plot_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger("run")

plt.style.use("ggplot")


def run(cfg: Config):
    policies: List[BasePolicy] = cfg["/policies"]
    generator: Generator = cfg["/generator"]
    test_data = generator.get_data(cfg["/num_samples"], cfg["/seed"])
    results = []
    timer = get_timer("timer")
    with timer("Experiment"):
        for policy in policies:
            policy_name = policy.__str__()
            with timer(policy_name):
                tour = policy.solve(test_data)
                score = calc_score(tour, test_data)
            LOGGER.info(f"{policy_name} score: {score.mean():.2f}")
            results.append(ExpResult(policy_name, tour, score))
    LOGGER.info(timer.render())
    save_dir = Path(cfg["/output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_results(results, test_data, save_dir=save_dir)


@click.command()
@click.option("--config_path", "-c", default="yml/exp001.yml")
@click.option("--default_config_path", "-d", default="yml/default.yml")
def main(config_path, default_config_path):
    cfg = load_config(config_path, default_config_path)
    run(cfg)


if __name__ == "__main__":
    main()

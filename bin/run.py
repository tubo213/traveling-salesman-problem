from src.utils import plot_results, calc_score, ExpResult
from src.utils import load_config
from pytorch_pfn_extras.config import Config
import click
from src.generator import Generator
from src.policy.base import BasePolicy
from typing import List
from pathlib import Path
import logging
import seaborn as sns
import matplotlib.pyplot as plt

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
    for policy in policies:
        policy_name = policy.__str__()
        LOGGER.info(f"Running {policy_name}")
        tour = policy.solve(test_data)
        score = calc_score(tour, test_data)
        LOGGER.info(f"Finished {policy_name} with score {score.mean():.2f}")
        results.append(ExpResult(policy_name, tour, score))

    save_dir = Path(cfg["/output_dir"])
    plot_results(results, test_data, save_dir=save_dir)


@click.command()
@click.option("--config_path", "-c", default="yml/exp001.yml")
@click.option("--default_config_path", "-d", default="yml/default.yml")
def main(config_path, default_config_path):
    cfg = load_config(config_path, default_config_path)
    run(cfg)


if __name__ == "__main__":
    main()

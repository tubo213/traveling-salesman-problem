"""
- 焼きなましの温度をOptunaでチューニングする
- 探索範囲はハードコーディング
"""
from src.utils import calc_score
from src.config import load_config
from pytorch_pfn_extras.config import Config
import optuna
import click


def tune(cfg: Config):
    generator = cfg["/generator"]
    test_data = generator.get_data(cfg["/num_samples"], cfg["/seed"])
    policy = cfg["/policy"]

    def objective(trial: optuna.trial.Trial):
        start_tmp = trial.suggest_float("start_tmp", 0.01, 100.0, log=False)
        end_tmp_rate = trial.suggest_float("end_tmp_rate", 0, 1)
        end_tmp = start_tmp * end_tmp_rate
        policy.start_tmp = start_tmp
        policy.end_tmp = end_tmp
        tour = policy.solve(test_data)
        score = calc_score(tour, test_data)

        return score.mean()

    study = optuna.create_study(
        direction="minimize",
    )
    study.optimize(objective, callbacks=[optuna.study.MaxTrialsCallback(cfg["/num_trials"])])
    print(study.best_params)


@click.command()
@click.option("--config_path", "-c", default="yml/tune/exp001.yml")
def main(config_path):
    cfg = load_config(config_path)
    tune(cfg)


if __name__ == "__main__":
    main()

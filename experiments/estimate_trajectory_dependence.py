import warnings
from pathlib import Path
from typing import Any

import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.algos.dqn import DQN
from src.utils.mixing_estimation import estimate_tau_mixing


warnings.filterwarnings("ignore")


def make_env(env_name: str, render_mode: str | None = None):
    env_kwargs = {}

    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    if "Pong" in env_name or "Breakout" in env_name:
        return gym.make(f"ALE/{env_name}", obs_type="ram", **env_kwargs)

    return gym.make(env_name, **env_kwargs)


def rollout(env, agent: DQN) -> np.ndarray:
    obs, _ = env.reset()

    trajectory = []
    finished = False

    while not finished:
        old_obs = obs

        processed_obs = agent.preprocess_state(obs)
        obs_np = (
            np.array(processed_obs)
            if not isinstance(processed_obs, np.ndarray)
            else processed_obs
        )

        with torch.no_grad():
            action = (
                agent.Q(torch.from_numpy(obs_np).float().unsqueeze(0))
                .argmax()
                .item()
            )

        obs, _, done, trunc, _ = env.step(action)

        experience = np.hstack([old_obs, action])
        trajectory.append(experience)

        if done or trunc:
            finished = True

    return np.array(trajectory)


def single_rollout_worker(cfg: dict[str, Any], seed: int) -> np.ndarray:
    """Run one policy rollout and estimate tau-mixing coefficients."""
    warnings.filterwarnings("ignore")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(cfg["env_name"])
    env.reset(seed=seed)

    agent = DQN(
        env=env,
        load_weights_path=cfg["load_model_path"],
        **cfg["dqn"],
    )

    trajectory = rollout(env, agent)

    tau = estimate_tau_mixing(
        trajectory,
        max_lag=cfg["max_lag"],
        n_neighbours=cfg["knn"],
        show_progress=False,
    )

    env.close()

    return tau


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="mixing_estimation/trajectories/config",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    worker_cfg = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )

    worker_cfg["load_model_path"] = to_absolute_path(worker_cfg["load_model_path"])

    output_path = Path(to_absolute_path(cfg.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg.seed)

    tau_list = Parallel(n_jobs=cfg.num_cores)(
        delayed(single_rollout_worker)(
            worker_cfg,
            seed=base_seed + i,
        )
        for i in tqdm(range(cfg.n_rollouts), desc="Rollouts")
    )

    all_tau_estimates = np.array(tau_list)

    np.save(output_path, all_tau_estimates)
    print(f"Tau estimates saved to {output_path}")


if __name__ == "__main__":
    main()
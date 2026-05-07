import gymnasium as gym
import ale_py  # noqa: F401
from pathlib import Path
import torch

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.algos.dqn import DQN
from src.utils.buffers import save_replay_buffer_states_actions


def make_env(env_name: str, render_mode: str | None = None):
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    if "Pong" in env_name or "Breakout" in env_name:
        return gym.make(f"ALE/{env_name}", obs_type="ram", **env_kwargs)

    return gym.make(env_name, **env_kwargs)


def create_dqn(cfg: DictConfig, sampler=None):
    dqn_kwargs = OmegaConf.to_container(cfg.dqn, resolve=True)

    return DQN(
        env=make_env(cfg.env_name),
        load_weights_path=cfg.load_model_path,
        minibatch_log_path=cfg.minibatch_log_path,
        sampler=sampler,
        **dqn_kwargs,
    )


@hydra.main(version_base=None, config_path="conf/train", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    sampler = instantiate(cfg.sampler) if cfg.get("sampler") is not None else None

    dqn = create_dqn(cfg, sampler=sampler)
    dqn.run()

    if cfg.buffer_path is None:
        buffer_dir = Path.home() / "tmp" / "saved_buffers"
    else:
        buffer_dir = Path(cfg.buffer_path)

    buffer_dir.mkdir(parents=True, exist_ok=True)
    buffer_path = buffer_dir / f"{cfg.env_name}.npz"

    save_replay_buffer_states_actions(dqn.buffer, buffer_path)
    print(f"Training completed and buffer saved to {buffer_path}")

    if cfg.save_model_path is not None:
        save_path = Path(cfg.save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dqn.Q.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    eval_env = make_env(
        cfg.env_name,
        render_mode="human" if cfg.render else None,
    )
    r = dqn.evaluate_model(env=eval_env, render=cfg.render)
    print(f"Evaluation reward: {r}")


if __name__ == "__main__":
    main()

import gymnasium as gym
import torch
import numpy as np
from typing import Union, Optional
from tqdm import tqdm

from src.buffers.replay_buffer import ReplayBuffer, BaseSampler, UniformSampler
from src.networks.q_network import QNetwork
from src.annealing_schedules.annealing import AbstractAnnealingSchedule, LinearSchedule


class DQN:
    def __init__(
        self,
        env: gym.Env,
        discount: float,
        buffer_size: int,
        max_episodes: int,
        minibatch_size: int,
        eps_schedule: Union[float, AbstractAnnealingSchedule],
        eps_min: float,
        lr: float,
        t_target: int,
        start_time: int,
        update_freq: int,
        n_updates: int,
        q_net_kwargs: dict,
        exploration_fraction: float = 1.0,
        decay: Optional[float] = None,
        tau: float = 1.0,
        load_weights_path: Optional[str] = None,
        minibatch_log_path: Optional[str] = None,
        sampler: Optional[BaseSampler] = None,
        buffer: Optional[ReplayBuffer] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Deep Q-Network agent for training and evaluating value-based policies.

        Supports discrete or Box observations, epsilon-greedy exploration with an
        annealing schedule, replay-buffer sampling, target-network updates, optional
        soft updates, and loading pretrained Q-network weights.

        Args:
            env: Gymnasium environment with discrete actions.
            discount: Discount factor for future rewards.
            buffer_size: Maximum number of transitions stored in replay buffer.
            max_episodes: Number of training episodes.
            minibatch_size: Number of samples per optimization step.
            eps_schedule: Initial epsilon value or custom annealing schedule.
            eps_min: Final/minimum epsilon when using a generated schedule.
            lr: Adam optimizer initial learning rate.
            t_target: Step interval for updating the target network.
            start_time: Number of transitions to collect before training starts.
            update_freq: Step interval between training updates.
            n_updates: Number of gradient updates per update step.
            q_net_kwargs: Keyword arguments used to construct the Q-networks.
            exploration_fraction: Fraction of training over which epsilon decays.
            decay: Optional decay parameter for the generated epsilon schedule.
            tau: Soft-update coefficient for the target network.
            load_weights_path: Optional path to pretrained Q-network weights.
            minibatch_log_path: Optional path for saving sampled minibatches.
            sampler: Optional sampler used by a newly created replay buffer.
            buffer: Optional preconstructed replay buffer.
            rng: Optional NumPy random generator for exploration decisions.
        """
        self.env = env
        self.discount = discount
        self.max_episodes = max_episodes
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.t_target = t_target
        self.update_freq = update_freq
        self.n_updates = n_updates
        self.tau = tau
        self.start_time = start_time
        self.minibatch_log_path = minibatch_log_path

        self.rng = rng if rng is not None else np.random.default_rng()

        if buffer is not None and sampler is not None:
            raise ValueError("Pass either `buffer` or `sampler`, not both.")

        if buffer is not None:
            self.buffer = buffer
        else:
            if sampler is None:
                sampler = UniformSampler(
                    replace=True
                )  # Default to uniform sampling with replacement
                self.buffer = ReplayBuffer(size=buffer_size, sampler=sampler)
            else:
                sampler = sampler
                self.buffer = ReplayBuffer(size=buffer_size, sampler=sampler)
        print(f"Sampler: {sampler}")
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Discrete):
            self.obs_type = "discrete"
            input_dim = obs_space.n
        elif isinstance(obs_space, gym.spaces.Box):
            self.obs_type = "box"
            input_dim = obs_space.shape[0]
        else:
            raise NotImplementedError(f"Unsupported observation space: {obs_space}")

        q_net_kwargs["input_dim"] = input_dim
        q_net_kwargs["output_dim"] = env.action_space.n

        self.Q = QNetwork(**q_net_kwargs)
        self.Q_target = QNetwork(**q_net_kwargs)
        self.Q_target.load_state_dict(self.Q.state_dict())

        if load_weights_path is not None:
            self.Q.load_state_dict(torch.load(load_weights_path))
            self.Q_target.load_state_dict(self.Q.state_dict())
            print(f"Loaded weights from {load_weights_path}")

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)

        total_steps = exploration_fraction * max_episodes
        if isinstance(eps_schedule, float):
            self.eps_schedule = LinearSchedule(
                initial_value=eps_schedule,
                final_value=eps_min,
                total_steps=total_steps,
                decay=decay,
            )
        else:
            self.eps_schedule = eps_schedule

        self.eps = self.eps_schedule.temp
        self.steps_taken = 0
        self.episode_rewards = []

    def preprocess_state(self, state):
        if self.obs_type == "discrete":
            one_hot = torch.zeros(self.env.observation_space.n, dtype=torch.float32)
            one_hot[state] = 1.0
            return one_hot
        else:
            return torch.tensor(state, dtype=torch.float32)

    def select_action(self, state):
        if self.rng.random() < self.eps:
            return self.env.action_space.sample()

        with torch.no_grad():
            q_vals = self.Q(state.unsqueeze(0))
            return torch.argmax(q_vals, dim=1).item()

    def _step(self, last_state):
        self.steps_taken += 1

        action = self.select_action(last_state)
        next_state, reward, done, trunc, _ = self.env.step(action)
        next_state = self.preprocess_state(next_state)

        experience = (last_state, action, reward, next_state, done or trunc)
        self.buffer.update(experience)

        return next_state, done or trunc

    def _get_loss(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.minibatch_size
        )

        if self.minibatch_log_path is not None:
            batch = (
                torch.cat([states, actions.unsqueeze(1).float()], dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            with open(self.minibatch_log_path, "ab") as f:
                np.save(f, batch)

        current_q = self.Q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            max_next_q = self.Q_target(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.discount * max_next_q

        loss = torch.nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        return loss

    def _soft_update(self):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _update(self):
        if (
            self.steps_taken % self.update_freq == 0
            and len(self.buffer) >= self.minibatch_size
        ):
            for _ in range(self.n_updates):
                loss = self._get_loss()
                self.optimizer.step()

        if self.steps_taken % self.t_target == 0:
            self._soft_update()

    def run(self):
        state, _ = self.env.reset()
        state = self.preprocess_state(state)

        while len(self.buffer) < self.start_time:
            done = False
            while not done:
                state, done = self._step(state)
            state, _ = self.env.reset()
            state = self.preprocess_state(state)

        for _ in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)

            done = False
            episode_reward = 0

            while not done:
                state, done = self._step(state)
                episode_reward += 1
                self._update()

            self.eps = self.eps_schedule()
            self.episode_rewards.append(episode_reward)

    def evaluate_model(self, env: gym.Env, render: bool = False):
        state, _ = env.reset()
        state = self.preprocess_state(state)

        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            with torch.no_grad():
                action = torch.argmax(self.Q(state.unsqueeze(0)), dim=1).item()

            state, reward, done, trunc, _ = env.step(action)
            state = self.preprocess_state(state)
            total_reward += reward
            done = done or trunc

        print("Evaluation reward:", total_reward)
        return total_reward


if __name__ == "__main__":
    pass

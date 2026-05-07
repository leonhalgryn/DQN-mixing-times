import numpy as np
from src.buffers.samplers import BaseSampler, UniformSampler
from typing import Optional, Sequence
import torch


class ReplayBuffer:
    def __init__(self, size: int, sampler: Optional[BaseSampler] = None):
        if size <= 0:
            raise ValueError("`size` must be positive.")

        self.size = size
        self.ptr = 0
        self.current_size = 0

        self._initialized = False

        self.sampler = sampler if sampler is not None else UniformSampler(replace=True)

        self.buffer = self

    def __len__(self) -> int:
        return self.current_size

    def set_sampler(self, sampler: BaseSampler) -> None:
        self.sampler = sampler

    def _initialize_arrays(self, state: np.ndarray) -> None:
        obs_dim = state.shape[0]

        self.states = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)

        self._initialized = True

    def update(self, experience: Sequence) -> None:
        state, action, reward, next_state, done = experience

        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy()

        if not self._initialized:
            self._initialize_arrays(state)

        i = self.ptr

        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done

        self.ptr = (self.ptr + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if not self._initialized or self.current_size == 0:
            raise ValueError("Cannot sample because the replay buffer is empty.")

        idx = self.sampler.sample_indices(
            self.current_size,
            batch_size,
            buffer=self,
        )

        idx = np.asarray(idx, dtype=np.int64)

        if idx.shape != (batch_size,):
            raise ValueError(
                f"Sampler must return shape ({batch_size},), got {idx.shape}."
            )
        if np.any(idx < 0) or np.any(idx >= self.current_size):
            raise ValueError("Sampler returned out-of-range indices.")

        return idx

    def sample(self, batch_size: int):
        idx = self.sample_indices(batch_size)

        states = torch.from_numpy(self.states[idx])
        actions = torch.from_numpy(self.actions[idx])
        rewards = torch.from_numpy(self.rewards[idx])
        next_states = torch.from_numpy(self.next_states[idx])
        dones = torch.from_numpy(self.dones[idx])

        return states, actions, rewards, next_states, dones


if __name__ == "__main__":
    pass

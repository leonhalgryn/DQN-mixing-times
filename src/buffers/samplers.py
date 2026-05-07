from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseSampler(ABC):
    """
    Base class for replay-buffer samplers.

    All samplers should implement `sample_indices`, which returns an array
    of indices into the replay buffer.
    """

    @abstractmethod
    def sample_indices(
        self,
        current_size: int,
        batch_size: int,
        *,
        buffer: Optional["ReplayBuffer"] = None,
    ) -> np.ndarray:
        """
        Return sampled indices in [0, current_size).

        Args:
            current_size: Number of elements currently stored.
            batch_size: Number of samples requested.

        Returns:
            A NumPy array of shape (batch_size,) containing integer indices.
        """
        raise NotImplementedError


class UniformSampler(BaseSampler):
    """
    Default sampler: uniform random sampling with replacement by default.

    If `replace=False` and `batch_size > current_size`, it returns each existing
    index once, then fills the remainder by sampling uniformly with replacement.
    """

    def __init__(self, replace: bool = True, rng: Optional[np.random.Generator] = None):
        self.replace = replace
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_indices(
        self,
        current_size: int,
        batch_size: int,
        *,
        buffer: Optional["ReplayBuffer"] = None,
    ) -> np.ndarray:
        if current_size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        if self.replace:
            return self.rng.choice(current_size, size=batch_size, replace=True)

        if batch_size <= current_size:
            return self.rng.choice(current_size, size=batch_size, replace=False)

        unique_part = np.arange(current_size, dtype=np.int64)
        remaining_part = self.rng.integers(
            low=0,
            high=current_size,
            size=batch_size - current_size,
            dtype=np.int64,
        )
        return np.concatenate([unique_part, remaining_part])


class OrderedUniformSampler(UniformSampler):
    """
    Uniform random sampling with replacement, but returned in ascending
    buffer-index order.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None, replace: bool = True):
        super().__init__(replace=replace, rng=rng)

    def sample_indices(
        self,
        current_size: int,
        batch_size: int,
        *,
        buffer: Optional["ReplayBuffer"] = None,
    ) -> np.ndarray:
        idx = super().sample_indices(current_size, batch_size, buffer=buffer)
        return np.sort(idx).astype(np.int64)

    def __init__(
        self,
        block_length: int,
        rng: Optional[np.random.Generator] = None,
        replace: bool = True,
        seed: Optional[int] = None,
        sort: bool = False,
    ):
        super().__init__()
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.block_length = block_length
        self.replace = replace
        self.sort = sort

    def sample_indices(self, current_size: int, batch_size: int, *, buffer=None):
        if current_size <= 0:
            raise ValueError("current_size must be positive")
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")

        if not self.replace and batch_size > current_size:
            raise ValueError(
                "Cannot sample more unique indices than current_size when replace=False"
            )

        if self.replace:
            sample = self._sample_with_replacement(current_size, batch_size)
        else:
            sample = self._sample_without_replacement(current_size, batch_size)
        if self.sort:
            sample.sort()
        return sample

    def _sample_with_replacement(self, current_size: int, batch_size: int):
        n_full_blocks, remainder = divmod(batch_size, self.block_length)

        parts = []

        if n_full_blocks:
            starts = self.rng.integers(0, current_size, size=n_full_blocks)
            offsets = np.arange(self.block_length)
            blocks = (starts[:, None] + offsets[None, :]) % current_size
            parts.append(blocks.ravel())

        if remainder:
            start = self.rng.integers(0, current_size)
            tail = (start + np.arange(remainder)) % current_size
            parts.append(tail)

        if not parts:
            return []

        return np.concatenate(parts).tolist()

    def _sample_without_replacement(self, current_size: int, batch_size: int):
        available = np.arange(current_size, dtype=np.int64)
        out = np.empty(batch_size, dtype=np.int64)
        write_pos = 0
        remaining = batch_size

        while remaining > 0:
            take = min(self.block_length, remaining)
            m = len(available)

            start = self.rng.integers(0, m)
            pos = (start + np.arange(take)) % m
            block = available[pos]

            out[write_pos : write_pos + take] = block
            write_pos += take
            remaining -= take

            mask = np.ones(m, dtype=bool)
            mask[pos] = False
            available = available[mask]

        return out.tolist()


class ContiguousBlockSampler(BaseSampler):
    """
    Contiguous block sampler with circular wraparound in replay-time order.

    Blocks are formed in logical chronological order. If the replay buffer
    has wrapped around, the oldest element is at `buffer.ptr`, so logical
    position k maps to physical index:

        physical = (buffer.ptr + k) % current_size     if buffer is full
        physical = k                                   otherwise

    Parameters
    ----------
    block_length : int
        number of observations per block
    gap_length : int, default=0
        gap length between blocks
    replace : bool, default=True
        If False, indices already sampled (in blockes + gaps) are removed
        from the currently available indices before the next block is drawn.
    sort : bool, default=False
        If True, sort the final returned physical indices.
    """

    def __init__(
        self,
        block_length: int,
        rng: Optional[np.random.Generator] = None,
        replace: bool = True,
        seed: Optional[int] = None,
        sort: bool = False,
        gap_length: int = 0,
    ):
        super().__init__()
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.block_length = block_length
        self.gap_length = gap_length
        self.replace = replace
        self.sort = sort

    def sample_indices(self, current_size: int, batch_size: int, *, buffer=None):
        if current_size <= 0:
            raise ValueError("current_size must be positive")
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        if self.gap_length < 0:
            raise ValueError("gap_length must be non-negative")

        if batch_size == 0:
            return []

        n_blocks = (batch_size + self.block_length - 1) // self.block_length
        total_consumed = batch_size + self.gap_length * max(0, n_blocks - 1)

        if not self.replace and total_consumed > current_size:
            raise ValueError(
                "Cannot sample requested gapped blocks without replacement: "
                f"need {total_consumed} logical positions but current_size={current_size}."
            )

        if self.replace:
            logical_idx = self._sample_with_replacement(current_size, batch_size)
        else:
            logical_idx = self._sample_without_replacement(current_size, batch_size)

        physical_idx = self._logical_to_physical(
            np.asarray(logical_idx, dtype=np.int64),
            current_size=current_size,
            buffer=buffer,
        )

        if self.sort:
            physical_idx.sort()

        return physical_idx.tolist()

    def _sample_with_replacement(self, current_size: int, batch_size: int):
        """
        Sample in logical index space [0, current_size).
        Always returns exactly batch_size logical indices.
        Gaps are internal only.
        """
        out = np.empty(batch_size, dtype=np.int64)
        write_pos = 0
        remaining = batch_size

        while remaining > 0:
            take = min(self.block_length, remaining)

            start = self.rng.integers(0, current_size)
            kept = (start + np.arange(take)) % current_size

            out[write_pos : write_pos + take] = kept
            write_pos += take
            remaining -= take

        return out.tolist()

    def _sample_without_replacement(self, current_size: int, batch_size: int):
        """
        Sample in logical index space [0, current_size) without replacement,
        removing both kept block positions and internal gap positions from the
        currently available logical pool.

        Always returns exactly batch_size logical indices.
        """
        available = np.arange(current_size, dtype=np.int64)
        out = np.empty(batch_size, dtype=np.int64)
        write_pos = 0
        remaining = batch_size

        while remaining > 0:
            take = min(self.block_length, remaining)
            consume = take + (self.gap_length if remaining > take else 0)

            m = len(available)
            start = self.rng.integers(0, m)
            pos = (start + np.arange(consume)) % m

            kept_pos = pos[:take]
            out[write_pos : write_pos + take] = available[kept_pos]
            write_pos += take
            remaining -= take

            mask = np.ones(m, dtype=bool)
            mask[pos] = False
            available = available[mask]

        return out.tolist()

    @staticmethod
    def _logical_to_physical(logical_idx: np.ndarray, current_size: int, buffer=None):
        """
        Convert logical replay-time indices to physical buffer array indices.

        If the buffer is not yet full, logical and physical order coincide.

        If the buffer is full, the oldest element is at buffer.ptr, so:
            logical 0 -> physical buffer.ptr
            logical 1 -> physical buffer.ptr + 1
            ...
        """
        if buffer is None:
            return logical_idx

        if (
            getattr(buffer, "current_size", None) is None
            or getattr(buffer, "ptr", None) is None
        ):
            return logical_idx

        if buffer.current_size < buffer.size:
            return logical_idx


class YuContiguousBlockSampler(BaseSampler):
    """
    Sampler that exactly matches the theoretical block-gap construction.
    See Section B.4 of the paper for the theoretical details of this sampler

    Construction
    ------------
    Choose one random logical starting index t0. Then retain observations in
    blocks of length `block_length`, separated by gaps of length `gap_length`:

        block 1: t0, ..., t0 + b - 1
        gap 1:   t0 + b, ..., t0 + b + a - 1
        block 2: t0 + b + a, ..., t0 + 2b + a - 1
        gap 2:   ...
        ...

    until exactly `batch_size` retained observations have been collected.
    The final retained block may be truncated.

    Indices are generated in logical replay-time order. If `wraparound=True`,
    logical positions are interpreted modulo `current_size`, which is natural
    for a full circular replay buffer. If `wraparound=False`, the sampler
    requires enough room from the chosen start index onward and will resample
    t0 if needed.

    Parameters
    ----------
    block_length : int
        Number of retained observations per full block.
    gap_length : int, default=0
        Number of skipped observations between retained blocks.
    rng : np.random.Generator, optional
        Random number generator.
    seed : int, optional
        Seed used if `rng` is not provided.
    sort : bool, default=False
        If True, sort the returned physical indices at the end.
        Usually this should remain False if you want the returned sample
        to preserve the theoretical retained order.
    wraparound : bool, default=True
        If True, logical indices are taken modulo `current_size`.
        If False, the whole sampled pattern must fit inside
        [0, current_size - 1] without wraparound.
    """

    def __init__(
        self,
        block_length: int,
        gap_length: int = 0,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        sort: bool = False,
        wraparound: bool = True,
    ):
        super().__init__()
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.block_length = block_length
        self.gap_length = gap_length
        self.sort = sort
        self.wraparound = wraparound

    def sample_indices(self, current_size: int, batch_size: int, *, buffer=None):
        if current_size <= 0:
            raise ValueError("current_size must be positive")
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        if self.gap_length < 0:
            raise ValueError("gap_length must be non-negative")

        if batch_size == 0:
            return []

        logical_idx = self._sample_logical_indices(
            current_size=current_size,
            batch_size=batch_size,
        )

        physical_idx = self._logical_to_physical(
            np.asarray(logical_idx, dtype=np.int64),
            current_size=current_size,
            buffer=buffer,
        )

        if self.sort:
            physical_idx.sort()

        return physical_idx.tolist()

    def _sample_logical_indices(self, current_size: int, batch_size: int):
        """
        Generate exactly `batch_size` retained logical indices according to
        the theoretical block-gap construction.

        If wraparound=True, indices are produced modulo current_size.
        If wraparound=False, the sampled pattern must fit without exceeding
        current_size - 1.
        """
        b = self.block_length
        a = self.gap_length

        n_full_blocks = batch_size // b
        remainder = batch_size % b
        n_blocks = n_full_blocks + (1 if remainder > 0 else 0)
        if batch_size <= b:
            total_span = batch_size
        else:
            if remainder == 0:
                total_span = (n_full_blocks - 1) * (b + a) + b
            else:
                total_span = n_full_blocks * (b + a) + remainder

        if not self.wraparound and total_span > current_size:
            raise ValueError(
                "Requested block-gap pattern does not fit without wraparound: "
                f"need span {total_span}, but current_size={current_size}."
            )

        if self.wraparound:
            t0 = int(self.rng.integers(0, current_size))
        else:
            max_start = current_size - total_span
            t0 = int(self.rng.integers(0, max_start + 1))

        out = np.empty(batch_size, dtype=np.int64)
        write_pos = 0
        block_id = 0

        while write_pos < batch_size:
            take = min(b, batch_size - write_pos)
            start = t0 + block_id * (b + a)
            block = start + np.arange(take, dtype=np.int64)

            if self.wraparound:
                block %= current_size

            out[write_pos : write_pos + take] = block
            write_pos += take
            block_id += 1

        return out.tolist()

    @staticmethod
    def _logical_to_physical(logical_idx: np.ndarray, current_size: int, buffer=None):
        """
        Convert logical replay-time indices to physical buffer array indices.

        If the buffer is not yet full, logical and physical order coincide.

        If the buffer is full, the oldest element is at buffer.ptr, so:
            logical 0 -> physical buffer.ptr
            logical 1 -> physical buffer.ptr + 1
            ...
        """
        if buffer is None:
            return logical_idx

        if (
            getattr(buffer, "current_size", None) is None
            or getattr(buffer, "ptr", None) is None
        ):
            return logical_idx

        if buffer.current_size < buffer.size:
            return logical_idx

        return (buffer.ptr + logical_idx) % current_size

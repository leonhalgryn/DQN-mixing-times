import numpy as np
from typing import Optional
import abc
from matplotlib import pyplot as plt


class AbstractAnnealingSchedule(abc.ABC):
    """
    Abstract base class for implementing annealing schedules for temperature-type parameters
    Atributes:
        :initial_value: (float):
        :final_value: (float):
        :total_steps: (int): total number of timesteps over which initial_value should be decayed to final_value
    """

    def __init__(self, initial_value: float, final_value: float, total_steps: int):
        self.initial_value = initial_value
        self.temp = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.steps = 0

    @abc.abstractmethod
    def update(self):
        """
        Decay the temperature parameter.
        """
        raise NotImplementedError

    def _set_steps(self, t: Optional[int] = None):
        if t is not None:
            self.steps = t

    def _reset(self):
        self.temp = self.initial_value
        self.steps = 0

    def plot_schedule(self, max_steps=None):
        if max_steps is None:
            max_steps = int(self.total_steps * 1.5)
        y = [self.__call__() for i in range(max_steps)]
        plt.plot(y)
        plt.show()
        self._reset()

    def __call__(self, t: Optional[int] = None):
        """
        Get the current value of the temperature parameter.
        """
        self._set_steps(t=t)
        if not self.total_steps:
            return self.temp
        if self.steps < self.total_steps:
            self.update()
        return self.temp


class ConstantSchedule(AbstractAnnealingSchedule):
    def __init__(
        self, initial_value: float, final_value: float = None, total_steps: int = None
    ):
        super().__init__(
            initial_value=initial_value,
            final_value=final_value,
            total_steps=total_steps,
        )

    def update(self):
        pass


class LinearSchedule(AbstractAnnealingSchedule):
    """
    Implements linear annealing from initial_value to final_value.
    """

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        decay: Optional[float] = None,
    ):
        super().__init__(initial_value, final_value, total_steps)
        self.decay = decay or (self.initial_value - self.final_value) / self.total_steps

    def update(self):
        self.steps += 1
        self.temp -= self.decay
        self.temp = max(self.final_value, self.temp)


class SquareRootSchedule(AbstractAnnealingSchedule):
    """ """

    def __init__(self, initial_value, final_value, total_steps):
        super().__init__(
            initial_value=initial_value,
            final_value=final_value,
            total_steps=total_steps,
        )

    def update(self):
        self.steps += 1
        self.temp = max(
            self.final_value,
            self.initial_value
            - (self.initial_value - self.final_value)
            * np.sqrt(min(1.0, self.steps / self.total_steps)),
        )


class ExponentialSchedule(AbstractAnnealingSchedule):
    """
    Implements exponential annealing from initial_value to final_value.
    """

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        decay: Optional[float] = None,
    ):
        super().__init__(initial_value, final_value, total_steps)
        self.decay = decay or np.log(final_value / initial_value) / self.total_steps

    def update(self):
        self.steps += 1
        self.temp = max(
            self.final_value, self.initial_value * np.exp(self.decay * self.steps)
        )


class CyclicalSchedule(AbstractAnnealingSchedule):
    """ """

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: float,
        cycles: int,
        exponent: float,
    ):
        super().__init__(
            initial_value=initial_value,
            final_value=final_value,
            total_steps=total_steps,
        )
        self.cycles = cycles
        self.p = exponent

    def update(self):
        self.steps += 1
        tmp = self.total_steps / self.cycles
        temp = ((self.steps % tmp) / tmp) ** self.p
        if self.steps >= self.total_steps:
            self.temp = self.final_value
        else:
            self.temp = max(self.final_value, self.initial_value - temp)


if __name__ == "__main__":
    pass

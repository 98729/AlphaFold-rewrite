import numpy as np


class RandomNormal:
    """Initializes by sampling from a normal distribution."""

    def __init__(self, stddev=1., mean=0.):
        """Constructs a :class:`RandomNormal` initializer.
        Args:
            stddev: The standard deviation of the normal distribution to sample from.
            mean: The mean of the normal distribution to sample from.
        """
        self.stddev = stddev
        self.mean = mean

    def __call__(self, shape):
        return self.mean + self.stddev * np.random.normal(shape)

import numpy as np


class RandomUniform:
    """Initializes by sampling from a uniform distribution."""

    def __init__(self, shape, minval=0., maxval=1.):
        """Constructs a :class:`RandomUniform` initializer.
    Args:
      minval: The lower limit of the uniform distribution.
      maxval: The upper limit of the uniform distribution.
    """
        self.shape = shape
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape):
        return np.random.uniform(self.minval, self.maxval, shape)

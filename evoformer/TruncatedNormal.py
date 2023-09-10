import numpy as np


class TruncatedNormal:
    """Initializes by sampling from a truncated normal distribution."""

    def __init__(self,
                 stddev=1.,
                 mean=0.):
        """Constructs a :class:`TruncatedNormal` initializer.
        Args:
            stddev: The standard deviation parameter of the truncated normal distribution.
            mean: The mean of the truncated normal distribution.
        """
        self.stddev = stddev
        self.mean = mean
        self.dtype = int or float

    def __call__(self, shape):
        real_dtype = np.finfo(self.dtype).dtype
        is_complex = np.issubdtype(self.dtype, np.complexfloating)
        if is_complex:
            shape = [2, *shape]
        unscaled = jax.random.truncated_normal(hk.next_rng_key(), -2., 2., shape, real_dtype)
        if is_complex:
            unscaled = unscaled[0] + 1j * unscaled[1]
        return self.stddev * unscaled + self.mean

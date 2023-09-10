import numpy as np
from TruncatedNormal import TruncatedNormal
from RandomNormal import RandomNormal
from RandomUniform import RandomUniform


def glorot_uniform(shape):
    return VarianceScaling(shape, scale=1.0, mode='fan_avg', distribution='uniform')


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        # Assuming convolution kernels (2D, 3D, or more.)
        # kernel_shape: (..., input_depth, depth)
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


class VarianceScaling:
    """Initializer which adapts its scale to the shape of the initialized array.
    The initializer first computes the scaling factor ``s = scale / n``, where n
    is:
        - Number of input units in the weight tensor, if ``mode = fan_in``.
        - Number of output units, if ``mode = fan_out``.
        - Average of the numbers of input and output units, if ``mode = fan_avg``.
    Then, with ``distribution="truncated_normal"`` or ``"normal"``,
    samples are drawn from a distribution with a mean of zero and a standard
    deviation (after truncation, if used) ``stddev = sqrt(s)``.
    With ``distribution=uniform``, samples are drawn from a uniform distribution
    within ``[-limit, limit]``, with ``limit = sqrt(3 * s)``.
    The variance scaling initializer can be configured to generate other standard
    initializers using the scale, mode and distribution arguments. Here are some
    example configurations:
    ==============  ==============================================================
    Name            Parameters
    ==============  ==============================================================
    glorot_uniform  VarianceScaling(1.0, "fan_avg", "uniform")
    glorot_normal   VarianceScaling(1.0, "fan_avg", "truncated_normal")
    lecun_uniform   VarianceScaling(1.0, "fan_in",  "uniform")
    lecun_normal    VarianceScaling(1.0, "fan_in",  "truncated_normal")
    he_uniform      VarianceScaling(2.0, "fan_in",  "uniform")
    he_normal       VarianceScaling(2.0, "fan_in",  "truncated_normal")
    ==============  ==============================================================
    """

    def __init__(self, shape, scale=1.0, mode='fan_in', distribution='truncated_normal'):
        """Constructs the :class:`VarianceScaling` initializer.
        Args:
            scale: Scale to multiply the variance by.
            mode: One of ``fan_in``, ``fan_out``, ``fan_avg``
            distribution: Random distribution to use. One of ``truncated_normal``,
            ``normal`` or ``uniform``.
        """
        if scale < 0.0:
            raise ValueError('`scale` must be a positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument:', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'truncated_normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument:', distribution)
        self.shape = shape
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def __call__(self, shape):
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == 'fan_in':
            scale /= max(1.0, fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)

        if self.distribution == 'truncated_normal':
            stddev = np.sqrt(scale)
            # Adjust stddev for truncation.
            # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            distribution_stddev = np.asarray(.87962566103423978)
            stddev = stddev / distribution_stddev
            return TruncatedNormal(stddev=stddev)(shape)
        elif self.distribution == 'normal':
            stddev = np.sqrt(scale)
            return RandomNormal(stddev=stddev)(shape)
        else:
            limit = np.sqrt(3.0 * scale)
            return RandomUniform(shape, minval=-limit, maxval=limit)

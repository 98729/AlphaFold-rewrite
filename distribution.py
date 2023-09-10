#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:04:01 2022

@author: sam
"""

import numpy as np

from erf_utils import erf_inv, erf


def to_abs_axes(axis: int or tuple[int], ndim: int):
    if isinstance(axis, int):
        return (axis,)
    else:
        return tuple(sorted({a % ndim for a in axis}))


class LayerNorm():

    def __init__(self,
                 axis: int or tuple,
                 create_scale: bool,
                 create_offset: bool,
                 eps: float = 1e-5,
                 scale_init: np.ndarray = None,
                 offset_init: np.ndarray = None,
                 use_fast_variance: bool = False,
                 name: str = None,
                 *,
                 param_axis: int or tuple = None
                 ):
        self.axis = axis
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = scale_init
        self.offset_init = offset_init
        self.use_fast_variance = use_fast_variance
        self.param_axis = ([-1] if param_axis is None else param_axis)
        self.name = name

    def __call__(self,
                 inputs: np.ndarray,
                 scale: np.ndarray = None,
                 offset: np.ndarray = None):
        if self.create_scale and scale is not None:
            raise ValueError(
                "Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`.")

        axis = self.axis
        mean = np.mean(inputs, axis=axis, keepdims=True)

        if self.use_fast_variance:
            mean_of_squares = np.mean(np.square(inputs), axis=axis, keepdims=True)
            variance = mean_of_squares - np.square(mean)
        else:
            variance = np.var(inputs, axis=axis, keepdims=True)

        param_axis = to_abs_axes(self.param_axis, inputs.ndim)

        if param_axis == (inputs.ndim - 1,):
            param_shape = (inputs.shape[-1],)

        else:
            param_shape = tuple((inputs.shape[i] if i in param_axis else 1)
                                for i in range(inputs.ndim))

        if self.create_scale:
            self.scale_init = np.array(1., dtype=inputs.dtype)
            scale = self.scale_init

        elif scale is None:
            scale = np.array(1., dtype=inputs.dtype)

        if self.create_offset:
            self.offset_init = np.array(0., dtype=inputs.dtype)
            offset = self.offset_init

        elif offset is None:
            offset = np.array(0., dtype=inputs.dtype)

        scale = np.broadcast_to(scale, inputs.shape)
        offset = np.broadcast_to(offset, inputs.shape)

        mean = np.broadcast_to(mean, inputs.shape)

        eps = np.asarray(self.eps, dtype=variance.dtype)

        inv = np.reciprocal(np.sqrt(variance + eps))

        result = inv * (inputs - mean) + offset
        return result


class Constant():

    def __init__(self, constant: int or float or np.ndarray):
        self.constant = constant

    def __call__(self, shape: tuple, dtype: np.float64):
        return np.broadcast_to(np.asarray(self.constant), shape).astype(dtype)


class RandomUniform():

    def __init__(self, minval=0., maxval=1.):
        """Constructs a :class:`RandomUniform` initializer.
        Args:
          minval: The lower limit of the uniform distribution.
          maxval: The upper limit of the uniform distribution.
        """
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape: int or tuple, dtype: np.float64):
        return np.random.uniform(low=self.minval, high=self.maxval, size=shape).astype(dtype)


class RandomNormal():

    def __init__(self, stddev=1., mean=0.):
        """Constructs a :class:`RandomNormal` initializer.
        Args:
          stddev: The standard deviation of the normal distribution to sample from.
          mean: The mean of the normal distribution to sample from.
        """
        self.stddev = stddev
        self.mean = mean

    def __call__(self, shape: int or tuple, dtype: np.float64):
        m = self.mean
        s = self.stddev

        return (m + s * np.random.standard_normal(size=shape)).astype(dtype)


def truncated_normal(lower: float or np.ndarray, upper: float or np.ndarray, dtype: np.float64, shape: tuple):
    if shape is None and isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray):
        shape = np.broadcast_shapes(np.shape(lower), np.shape(upper))
    elif shape is None and not isinstance(lower, np.ndarray):
        raise ValueError('shape must be setted if lower and upper are not np.ndarray')

    sqrt2 = np.array(np.sqrt(2), dtype)
    if isinstance(lower, np.ndarray):
        lower = lower.astype(dtype)
    if isinstance(upper, np.ndarray):
        upper = upper.astype(dtype)
    # print(lower)
    a = erf(lower / sqrt2)
    # print(a)
    b = erf(upper / sqrt2)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("truncated_normal only accepts floating point dtypes.")

    u = np.random.uniform(low=a, high=b, size=shape).astype(dtype)
    out = sqrt2 * erf_inv(u)

    return np.clip(out,
                   np.nextafter(lower, np.array(np.inf, dtype=dtype)),
                   np.nextafter(upper, np.array(-np.inf, dtype=dtype)))


class TruncatedNormal():

    def __init__(self, stddev: float or np.ndarray = 1.0, mean: float or np.ndarray = 0.0):

        """Constructs a :class:`TruncatedNormal` initializer.
        Args:
          stddev: The standard deviation parameter of the truncated
            normal distribution.
          mean: The mean of the truncated normal distribution.
        """
        self.stddev = stddev
        self.mean = mean

    def __call__(self, shape: tuple, dtype: np.float64):

        real_dtype = np.finfo(dtype).dtype
        if isinstance(self.mean, np.ndarray):
            m = self.mean.astype(dtype)

        elif not isinstance(self.mean, np.ndarray):
            m = np.array(self.mean).astype(dtype)

        if isinstance(self.stddev, np.ndarray):
            s = self.stddev.astype(real_dtype)

        elif not isinstance(self.stddev, np.ndarray):
            s = np.array(self.stddev).astype(dtype)

        is_complex = np.issubdtype(dtype, np.complexfloating)

        if is_complex:
            shape = [2, *shape]

        unscaled = truncated_normal(-2., 2., dtype=real_dtype, shape=shape)

        if is_complex:
            unscaled = unscaled[0] + 1j * unscaled[1]

        return s * unscaled + m


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


class VarianceScaling():

    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal'):

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

        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def __call__(self, shape: tuple, dtype: np.float64):

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
            distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
            stddev = stddev / distribution_stddev

            return TruncatedNormal(stddev=stddev)(shape, dtype)
        elif self.distribution == 'normal':
            stddev = np.sqrt(scale)
            return RandomNormal(stddev=stddev)(shape, dtype)
        else:
            limit = np.sqrt(3.0 * scale)
            return RandomUniform(minval=-limit, maxval=limit)(shape, dtype)

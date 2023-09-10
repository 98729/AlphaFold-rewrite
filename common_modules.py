#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:47:18 2022

@author: sam
"""

import numpy as np
from distribution import LayerNorm, Constant, RandomUniform, RandomNormal, truncated_normal, TruncatedNormal, \
    VarianceScaling

TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978,
                                            dtype=np.float32)


def get_initializer_scale(initializer_name, input_shape):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == 'zeros':
        w_init = Constant(0.0)

    else:
        # fan-in scaling
        scale = 1.

        for channel_dim in input_shape:
            scale /= channel_dim

        if initializer_name == 'relu':
            scale *= 2

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = TruncatedNormal(mean=0.0, stddev=stddev)

    return w_init


class Linear():

    def __init__(self,
                 num_output: int or tuple,
                 initializer: str = 'linear',
                 num_input_dims: int = 1,
                 use_bias: bool = True,
                 bias_init: float = 0.,
                 precision=None,
                 name: str = 'linear'):

        if isinstance(num_output, int):
            self.output_shape = (num_output,)

        if not isinstance(num_output, int) and not isinstance(num_output, tuple):
            raise ValueError('num_output must be integer or tuple of int')

        if isinstance(num_output, tuple):
            self.output_shape = num_output

        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.num_input_dims = num_input_dims
        self.num_output_dims = len(self.output_shape)
        self.precision = precision
        self.name = name

    def __call__(self, inputs):

        """Connects Module.
        Args:
          inputs: Tensor with at least num_input_dims dimensions.
        Returns:
          output of shape [...] + num_output.
        """

        num_input_dims = self.num_input_dims

        if self.num_input_dims > 0:
            in_shape = inputs.shape[-self.num_input_dims:]
        else:
            in_shape = ()

        weight_init = get_initializer_scale(self.initializer, in_shape)

        in_letters = 'abcde'[:self.num_input_dims]
        out_letters = 'hijkl'[:self.num_output_dims]

        weight_shape = in_shape + self.output_shape
        weights = weight_init(shape=weight_shape, dtype=inputs.dtype)

        equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

        output = np.einsum(equation, inputs, weights)

        if self.use_bias:
            bias = Constant(self.bias_init)(self.output_shape, dtype=inputs.dtype)

        output += bias

        return output

# x = np.random.normal(size = (8,8,64))

# linear = Linear(num_output=256,initializer='relu',name='transition1')(x)
# print(linear)

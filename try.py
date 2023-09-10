#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:34:44 2022

@author: sam
"""

import collections.abc
import types
from typing import Optional, Sequence, Tuple, Union
import numpy as np
#import jax
#import jax.numpy as jnp

AxisOrAxes = Union[int, Sequence[int], slice]
AxesOrSlice = Union[Tuple[int, ...], slice]

# TODO(tomhennigan): Update users to `param_axis=-1` and flip + remove this.
ERROR_IF_PARAM_AXIS_NOT_EXPLICIT = False


def to_axes_or_slice(axis: AxisOrAxes) -> AxesOrSlice:
    if isinstance(axis, slice):
      return axis
    elif isinstance(axis, int):
      return (axis,)
    elif (isinstance(axis, collections.abc.Iterable) and
          all(isinstance(ax, int) for ax in axis)):
      return tuple(axis)
    else:
      raise ValueError(
          f"`axis` should be an int, slice or iterable of ints. Got: {axis}")


def to_abs_axes(axis: AxesOrSlice, ndim: int) -> Tuple[int, ...]:
    if isinstance(axis, slice):
      return tuple(range(ndim)[axis])
    else:
      return tuple(sorted({a % ndim for a in axis}))
  
    
axis = [-1]
axis = to_axes_or_slice(axis)
print(axis)

#x = jnp.ones([1,2,3])
#print(x)

L = ([-1])
print(L)

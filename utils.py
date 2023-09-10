#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:11:19 2022

@author: sam
"""

import numpy as np


def final_init(x: bool):
    if x:
        return 'zeros'

    else:
        return 'linear'


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""

    if drop_mask_channel:
        mask = mask[..., 0]

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, int):

        axis = [axis]

    elif axis is None:

        axis = list(range(len(mask_shape)))

    broadcast_factor = 1.

    for axis_ in axis:

        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]

        if mask_size == 1:
            broadcast_factor *= value_size

        else:
            assert mask_size == value_size

    return (np.sum(mask * value, axis=tuple(axis)) / (np.sum(mask, axis=tuple(axis)) * broadcast_factor + eps))

# msa_act = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
#                     26,27,28,29,30,31,32])
# msa_act = np.reshape(msa_act,(2,2,8))

# pair_act = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
#                     26,27,28,29,30,31,32])
# pair_act = np.reshape(pair_act,(2,2,8))

# q_avg = mask_mean(msa_act, pair_act, axis=1)
# print(q_avg)

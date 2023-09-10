#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:03:54 2022

@author: sam
"""

import numpy as np

def sharded_apply(fun:callable, shard_size = None, in_axes:int = 0, out_axes:int = 0, new_out_axes:bool = False):
    
    if new_out_axes:
        raise NotImplementedError('New output axes not yet implemented.')

    # shard size None denotes no sharding
    if shard_size is None:
        return fun

def inference_subbatch(module:callable, subbatch_size:int, 
                       batched_args, nonbatched_args, low_memory:bool = False,
                       input_subbatch_dim: int = 0, output_subbatch_dim = None):
    
    assert len(batched_args) > 0
    
    if not low_memory:
    
        args = list(batched_args) + list(nonbatched_args)
        
        return module(*args)
    
    if output_subbatch_dim is None:
        output_subbatch_dim = input_subbatch_dim
    
    def run_module(*batched_args):
        
        args = list(batched_args) + list(nonbatched_args)
        
        return module(*args)
    
    sharded_module = sharded_apply(run_module,
                                 shard_size=subbatch_size,
                                 in_axes=input_subbatch_dim,
                                 out_axes=output_subbatch_dim)
    #print(sharded_module)
    return sharded_module(*batched_args)
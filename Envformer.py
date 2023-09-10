#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:41:58 2022

@author: sam
"""

import numpy as np
from erf_utils import erf_inv, erf
import mapping
from distribution import LayerNorm, Constant, RandomUniform, RandomNormal, truncated_normal, TruncatedNormal,VarianceScaling
import common_modules
import utils
import sys
import pickle
import residue_constants
import quat_affine


def to_abs_axes(axis:int or tuple[int], ndim:int):
    if isinstance(axis, int):
        return (axis,)
    else:
        return tuple(sorted({a % ndim for a in axis}))
    
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

 
    
        
def glorot_uniform():
    
    return VarianceScaling(scale=1.0,
                           mode='fan_avg',
                           distribution='uniform')

def softmax(x:int or float or np.ndarray):
    
    x_max = np.amax(x, axis = -1,keepdims=True)
    unnormalized = np.exp(x - x_max)
    
    return unnormalized / np.sum(unnormalized, axis = -1, keepdims=True)

def sigmoid(x:int or float or np.ndarray):
    
    return 1.0 / np.add(1.0, np.exp(np.negative(x)))

def relu(x:int or float or np.ndarray):
    
    return(np.maximum(0, x))

def one_hot(x:np.ndarray, num_classes:int, dtype = np.float64):
    
    return np.eye(num_classes)[x]


def apply_dropout(*, tensor, safe_key, rate, is_training, broadcast_dim=None):
    
    """Applies dropout to a tensor."""
    np.random.seed(safe_key)
    if is_training and rate != 0.0:
        
        shape = list(tensor.shape)
    
        if broadcast_dim is not None:
            
            shape[broadcast_dim] = 1
        
        keep_rate = 1.0 - rate
        
        keep = np.random.binomial(n = 1, p = keep_rate, size = tuple(shape))
        
        return keep * tensor / keep_rate
    
    else:
        return tensor
    
def dropout_wrapper(module,
                    input_act,
                    mask,
                    safe_key,
                    output_act=None,
                    is_training=True,
                    **kwargs):
    """Applies module + dropout + residual update."""
    if output_act is None:
        output_act = input_act
    
    residual = module(input_act, mask, key = safe_key, is_training=is_training, **kwargs)
    
    dropout_rate = 0.0 if module.__dict__['deterministic'] else module.__dict__['dropout_rate']
    
    if module.__dict__['shared_dropout']:
        
        if module.__dict__['orientation'] == 'per_row':
            
            broadcast_dim = 0
        
        else:
            
            broadcast_dim = 1
    
    else:
        
        broadcast_dim = None
    
    residual = apply_dropout(tensor=residual,
                           safe_key=safe_key,
                           rate=dropout_rate,
                           is_training=is_training,
                           broadcast_dim=broadcast_dim)
    
    new_act = output_act + residual

    return new_act

class Attention():
    
    def __init__(self,output_dim,dropout_rate = 0.15, gating = True, num_head = 8, 
                 orientation = 'per_row', shared_dropout = True, deterministic = False,
                 multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False,
                 name='attention'):
        
        self.dropout_rate = dropout_rate
        self.gating = gating
        self.num_head = num_head
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.output_dim = output_dim
        self.name = name
        
    def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
        
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        
        key_dim = int(q_data.shape[-1])
        value_dim = int(m_data.shape[-1])
        num_head = self.num_head
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head
        #print(q_data.shape)
        q_weights = glorot_uniform()(shape = (q_data.shape[-1], num_head, key_dim), dtype = np.float32)
        k_weights = glorot_uniform()(shape = (m_data.shape[-1], num_head, key_dim), dtype = np.float32)
        v_weights = glorot_uniform()(shape = (m_data.shape[-1], num_head, value_dim), dtype = np.float32)
        #print(q_weights.shape)
        q = np.einsum('bqa,ahc->bqhc', q_data, q_weights,optimize='optimal') * key_dim**(-0.5)
        #print('q: ',q.shape)
        k = np.einsum('bka,ahc->bkhc', m_data, k_weights,optimize='optimal')
        #print('k: ',k.shape)
        v = np.einsum('bka,ahc->bkhc', m_data, v_weights,optimize='optimal')
        #print('v: ',v.shape)
        logits = np.einsum('bqhc,bkhc->bhqk', q, k,optimize='optimal') + bias
        
        if nonbatched_bias is not None:
            logits += np.expand_dims(nonbatched_bias, axis=0)
        #print('logits: ',logits)
        weights = softmax(logits)
        #print(weights)
        weighted_avg = np.einsum('bhqk,bkhc->bqhc', weights, v,optimize='optimal')
        
        if self.zero_init:
            init = Constant(0.0)
        
        else:
            init = glorot_uniform()
        
        if self.gating:
            gating_weights = Constant(0.0)(shape = (q_data.shape[-1], num_head, value_dim),dtype = np.float32)
            
            gating_bias = Constant(1.0)(shape = (num_head, value_dim), dtype = np.float32)
            #print(gating_bias.shape)
            
            gate_values = np.einsum('bqc, chv->bqhv', q_data, gating_weights,optimize='optimal') + gating_bias
            #print(gate_values.shape)
            gate_values = sigmoid(gate_values)
            
            weighted_avg *= gate_values
        
        o_weights = init(shape = (num_head, value_dim, self.output_dim), dtype = np.float32)
        
        o_bias = Constant(0.0)(shape = (self.output_dim,), dtype = np.float32)
        
        output = np.einsum('bqhc,hco->bqo', weighted_avg, o_weights,optimize='optimal') + o_bias
        
        return output



class GlobalAttention():
    
    def __init__(self,output_dim,dropout_rate = 0.0, gating = True, num_head = 8, 
                 orientation = 'per_column', shared_dropout = True, deterministic = False,
                 multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False,
                 name='attention'):
        
        self.dropout_rate = dropout_rate
        self.gating = gating
        self.num_head = num_head
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.output_dim = output_dim
        self.name = name
    
    def __call__(self, q_data, m_data, q_mask):
        
        """Builds GlobalAttention module.
        Arguments:
          q_data: A tensor of queries with size [batch_size, N_queries,
            q_channels]
          m_data: A tensor of memories from which the keys and values
            projected. Size [batch_size, N_keys, m_channels]
          q_mask: A binary mask for q_data with zeros in the padded sequence
            elements and ones otherwise. Size [batch_size, N_queries, q_channels]
            (or broadcastable to this shape).
        Returns:
          A float32 tensor of size [batch_size, N_queries, output_dim].
        """
        
        key_dim = int(q_data.shape[-1])
        value_dim = int(m_data.shape[-1])
        num_head = self.num_head
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head
        
        q_weights = glorot_uniform()(shape = (q_data.shape[-1], num_head, key_dim),dtype=np.float32)
        k_weights = glorot_uniform()(shape = (m_data.shape[-1], key_dim),dtype=np.float32)
        v_weights = glorot_uniform()(shape = (m_data.shape[-1], value_dim),dtype=np.float32)
        
        v = np.einsum('bka,ac->bkc', m_data, v_weights,optimize='optimal')
        
        q_avg = utils.mask_mean(q_mask, q_data, axis=1)
        
        q = np.einsum('ba,ahc->bhc', q_avg, q_weights,optimize='optimal') * key_dim**(-0.5)
        k = np.einsum('bka,ac->bkc', m_data, k_weights,optimize='optimal')
        bias = (1e9 * (q_mask[:, None, :, 0] - 1.))
        logits = np.einsum('bhc,bkc->bhk', q, k,optimize='optimal') + bias
        weights = softmax(logits)
        weighted_avg = np.einsum('bhk,bkc->bhc', weights, v,optimize='optimal')
        
        if self.zero_init:
            init = Constant(0.0)
        
        else:
            init = glorot_uniform()
        
        o_weights = init(shape = (num_head, value_dim, self.output_dim), dtype = np.float32)
        
        o_bias = Constant(0.0)(shape = (self.output_dim,), dtype = np.float32)
        
        if self.gating:
            gating_weights = Constant(0.0)(shape = (q_data.shape[-1], num_head, value_dim),dtype = np.float32)
        
            gating_bias = Constant(1.0)(shape = (num_head, value_dim),dtype = np.float32)
            
            gate_values = np.einsum('bqc, chv->bqhv', q_data,gating_weights,optimize='optimal')
            
            gate_values = sigmoid(gate_values + gating_bias)
            
            weighted_avg = weighted_avg[:, None] * gate_values
            
            output = np.einsum('bqhc,hco->bqo', weighted_avg, o_weights,optimize='optimal') + o_bias
            
        else:
            output = np.einsum('bhc,hco->bo', weighted_avg, o_weights,optimize='optimal') + o_bias
            output = output[:, None]
        
        return output
            
        
        

class MSARowAttentionWithPairBias():
    
    def __init__(self,dropout_rate = 0.15, gating = True, num_head = 8, 
                 orientation = 'per_row', shared_dropout = True, deterministic = False,
                 multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False,
                 name = 'msa_row_attention_with_pair_bias'):
        
        self.dropout_rate = dropout_rate
        self.gating = gating
        self.num_head = num_head
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self, msa_act,
               msa_mask,
               pair_act,
               key,
               is_training=True):
        
        """Builds MSARowAttentionWithPairBias module.
        Arguments:
          msa_act: [N_seq, N_res, c_m] MSA representation.
          msa_mask: [N_seq, N_res] mask of non-padded regions.
          pair_act: [N_res, N_res, c_z] pair representation.
          is_training: Whether the module is in training mode.
        Returns:
          Update to msa_act, shape [N_seq, N_res, c_m].
        """
        np.random.seed(key)
        assert len(msa_act.shape) == 3
        assert len(msa_mask.shape) == 2
        assert self.orientation == 'per_row'
        
        bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4
        #print('bias',bias)
        msa_act = LayerNorm(
            axis = -1, create_scale=True, create_offset=True, name='query_norm')(
                msa_act)
        #print('msa_act: ', msa_act)
        pair_act = LayerNorm(
            axis = -1,
            create_scale=True,
            create_offset=True,
            name='feat_2d_norm')(
                pair_act)
        #print('pair_act: ', pair_act)
        init_factor = 1. / np.sqrt(int(pair_act.shape[-1]))
        
        
        
        weights = np.random.normal(scale = init_factor, size = (pair_act.shape[-1], self.num_head))
        #print('weights: ',weights)
        nonbatched_bias = np.einsum('qkc,ch->hqk', pair_act, weights,optimize='optimal')
        #print('nonbatched_bias',nonbatched_bias)
        attn_mod = Attention(output_dim=msa_act.shape[-1],dropout_rate=self.dropout_rate,gating=self.gating,
                             num_head=self.num_head,orientation=self.orientation,
                             shared_dropout=self.shared_dropout,deterministic=self.deterministic,
                             multimer_mode=self.multimer_mode,subbatch_size=self.subbatch_size,
                             use_remat=self.use_remat,zero_init=self.zero_init)
        
        msa_act = mapping.inference_subbatch(
            attn_mod,
            subbatch_size = self.subbatch_size,
            batched_args=[msa_act, msa_act, bias],
            nonbatched_args=[nonbatched_bias],
            low_memory=False)
        
        return msa_act


class MSAColumnAttention():
    
    def __init__(self, dropout_rate = 0.0, gating = True, num_head = 8,
                 orientation = 'per_column', shared_dropout = True, deterministic = False,
                 multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False,
                 name = 'msa_column_attention'):
        
        self.dropout_rate = dropout_rate
        self.gating = gating
        self.num_head = num_head
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self, msa_act,
                 msa_mask,
                 key,
                 is_training=True):
        
        """Builds MSAColumnAttention module.
        Arguments:
          msa_act: [N_seq, N_res, c_m] MSA representation.
          msa_mask: [N_seq, N_res] mask of non-padded regions.
          is_training: Whether the module is in training mode.
        Returns:
          Update to msa_act, shape [N_seq, N_res, c_m]
        """
        np.random.seed(key)
        assert len(msa_act.shape) == 3
        assert len(msa_mask.shape) == 2
        assert self.orientation == 'per_column'
        
        msa_act = np.swapaxes(msa_act, -2, -3)
        msa_mask = np.swapaxes(msa_mask, -1, -2)

        bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4
        
        msa_act = LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name='query_norm')(
                msa_act)
                
        attn_mod = Attention(output_dim=msa_act.shape[-1],dropout_rate=self.dropout_rate,gating=self.gating,
                             num_head=self.num_head,orientation=self.orientation,
                             shared_dropout=self.shared_dropout,deterministic=self.deterministic,
                             multimer_mode=self.multimer_mode,subbatch_size=self.subbatch_size,
                             use_remat=self.use_remat,zero_init=self.zero_init)
        
        msa_act = mapping.inference_subbatch(
            attn_mod,
            subbatch_size = self.subbatch_size,
            batched_args=[msa_act, msa_act, bias],
            nonbatched_args=[],
            low_memory=False)
        
        msa_act = np.swapaxes(msa_act, -2, -3)
        
        return msa_act
    


class MSAColumnGlobalAttention():
    
    def __init__(self, dropout_rate = 0.0, gating = True, num_head = 8,
                 orientation = 'per_column', shared_dropout = True, deterministic = False,
                 multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False,
                 name = 'msa_column_global_attention'):
        
        self.dropout_rate = dropout_rate
        self.gating = gating
        self.num_head = num_head
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self,msa_act,
               msa_mask,
               key,
               is_training=True):
        
        """Builds MSAColumnGlobalAttention module.
        Arguments:
          msa_act: [N_seq, N_res, c_m] MSA representation.
          msa_mask: [N_seq, N_res] mask of non-padded regions.
          is_training: Whether the module is in training mode.
        Returns:
          Update to msa_act, shape [N_seq, N_res, c_m].
        """
        np.random.seed(key)
        assert len(msa_act.shape) == 3
        assert len(msa_mask.shape) == 2
        assert self.orientation == 'per_column'

        msa_act = np.swapaxes(msa_act, -2, -3)
        msa_mask = np.swapaxes(msa_mask, -1, -2)
        
        bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4
        
        msa_act = LayerNorm(axis=-1, create_scale=True, create_offset=True, name='query_norm')(
            msa_act)
        
        attn_mod = GlobalAttention(output_dim=msa_act.shape[-1],dropout_rate=self.dropout_rate,gating=self.gating,
                             num_head=self.num_head,orientation=self.orientation,
                             shared_dropout=self.shared_dropout,deterministic=self.deterministic,
                             multimer_mode=self.multimer_mode,subbatch_size=self.subbatch_size,
                             use_remat=self.use_remat,zero_init=self.zero_init,name = 'attention')
        
        msa_mask = np.expand_dims(msa_mask, axis=-1)
        
        msa_act = mapping.inference_subbatch(
            attn_mod,
            subbatch_size=self.subbatch_size,
            batched_args=[msa_act, msa_act, msa_mask],
            nonbatched_args=[],
            low_memory=False)
        
        msa_act = np.swapaxes(msa_act, -2, -3)
        
        return msa_act

        


class Transition():
    
    def __init__(self, dropout_rate = 0.0, num_intermediate_factor = 4, 
                  orientation = 'per_row', shared_dropout = True, deterministic = False,
                  multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False,
                  name = 'transition_block'):
        
        self.dropout_rate = dropout_rate
        self.num_intermediate_factor = num_intermediate_factor
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self, act, mask, key, is_training = True):
        
        """Builds Transition module.
        Arguments:
          act: A tensor of queries of size [batch_size, N_res, N_channel].
          mask: A tensor denoting the mask of size [batch_size, N_res].
          is_training: Whether the module is in training mode.
        Returns:
          A float32 tensor of size [batch_size, N_res, N_channel].
        """
        np.random.seed(key)
        _, _, nc = act.shape
        
        num_intermediate = int(nc * self.num_intermediate_factor)
        mask = np.expand_dims(mask, axis=-1)
        
        act = LayerNorm(axis = -1, 
                        create_scale=True,
                        create_offset=True,
                        name='input_layer_norm')(act)
        
        transition_module_1 = common_modules.Linear(num_intermediate,
                                                    initializer ='relu',
                                                    name = 'transition1')
        
        transition_module_2 = common_modules.Linear(nc,
                                                    initializer=utils.final_init(self.zero_init),
                                                    name = 'transition2')
        
        act1 = mapping.inference_subbatch(transition_module_1,
                                          subbatch_size = self.subbatch_size,
                                          batched_args = [act],
                                          nonbatched_args = [],
                                          low_memory = False)
        
        act2 = relu(act1)
        
        act3 = mapping.inference_subbatch(transition_module_2,
                                          subbatch_size = self.subbatch_size,
                                          batched_args = [act2],
                                          nonbatched_args = [],
                                          low_memory = False)
        
        return act3
        

class OuterProductMean():

    def __init__(self, num_output_channel, first = False, chunk_size = 128, dropout_rate = 0.0,
                 num_outer_channel = 32, orientation = 'per_row', shared_dropout = True,
                 deterministic = False, multimer_mode = False, subbatch_size = 4,
                 use_remat = False, zero_init = False, name = 'outer_product_mean'):
        
        self.num_output_channel = num_output_channel
        self.first = first
        self.chunk_size = chunk_size
        self.dropout_rate = dropout_rate
        self.num_outer_channel = num_outer_channel
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self, act, mask, key, is_training=True):
        
        """Builds OuterProductMean module.
        Arguments:
          act: MSA representation, shape [N_seq, N_res, c_m].
          mask: MSA mask, shape [N_seq, N_res].
          is_training: Whether the module is in training mode.
        Returns:
          Update to pair representation, shape [N_res, N_res, c_z].
        """
        
        np.random.seed(key)
        mask = mask[..., None]
        act = LayerNorm(axis = -1, create_scale = True, create_offset = True, name='layer_norm_input')(act)
        
        left_act = mask * common_modules.Linear(
            self.num_outer_channel,
            initializer='linear',
            name='left_projection')(
                act)
        
        right_act = mask * common_modules.Linear(
            self.num_outer_channel,
            initializer='linear',
            name='right_projection')(
                act)
        
        if self.zero_init:
            init_w = Constant(0.0)
        
        else:
            init_w = VarianceScaling(scale=2., mode='fan_in')
        
        output_w = init_w(shape = (self.num_outer_channel, self.num_outer_channel,self.num_output_channel),
                          dtype = np.float64)
        
        output_b = Constant(0.0)(shape = (self.num_output_channel,),
                                 dtype = np.float64)
        
        def compute_chunk(left_act):
            
            # This is equivalent to
            #
            # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
            # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
            #
            # but faster.
            
            left_act = np.transpose(left_act, [0, 2, 1])
            act = np.einsum('acb,ade->dceb', left_act, right_act,optimize='optimal')
            act = np.einsum('dceb,cef->dbf', act, output_w,optimize='optimal') + output_b
            
            return np.transpose(act, [1, 0, 2])
        
        act = mapping.inference_subbatch(
            compute_chunk,
            self.chunk_size,
            batched_args = [left_act],
            nonbatched_args = [],
            low_memory = False,
            input_subbatch_dim = 1,
            output_subbatch_dim = 0)
        
        epsilon = 1e-3
        norm = np.einsum('abc,adc->bdc', mask, mask,optimize='optimal')
        act /= epsilon + norm
        
        return act
    

class TriangleMultiplicationOutgoing():
    
    def __init__(self, dropout_rate = 0.25, equation = 'ikc,jkc->ijc', num_intermediate_channel = 128,
                 orientation = 'per_row', shared_dropout = True, deterministic = False,
                 multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False, 
                 name='triangle_multiplication_outgoing'):
        
        self.dropout_rate = dropout_rate
        self.equation = equation
        self.num_intermediate_channel = num_intermediate_channel
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self, act, mask, key, is_training=True):
        
        """Builds TriangleMultiplication module.
        Arguments:
          act: Pair activations, shape [N_res, N_res, c_z]
          mask: Pair mask, shape [N_res, N_res].
          is_training: Whether the module is in training mode.
        Returns:
          Outputs, same shape/type as act.
        """
        
        
        del is_training
        
        np.random.seed(key)
        mask = mask[..., None]
        
        act = LayerNorm(axis = -1, create_scale=True, create_offset=True,
                       name='layer_norm_input')(act)
        
        input_act = act
        
        left_projection = common_modules.Linear(self.num_intermediate_channel,name = 'left_projection')
        #print(left_projection)
        left_proj_act = mask * left_projection(act)
        #print('left_proj_act: ', left_proj_act)
        right_projection = common_modules.Linear(self.num_intermediate_channel,name = 'right_projection')
        
        right_proj_act = mask * right_projection(act)
        
        left_gate_values = sigmoid(common_modules.Linear(self.num_intermediate_channel,bias_init = 1., 
                                                 initializer = utils.final_init(self.zero_init),
                                                 name = 'left_gate')(act))
        
        right_gate_values = sigmoid(common_modules.Linear(self.num_intermediate_channel, bias_init = 1., 
                                                  initializer = utils.final_init(self.zero_init),
                                                  name = 'right_gate')(act))
        
        left_proj_act *= left_gate_values
        right_proj_act *= right_gate_values
        
        act = np.einsum(self.equation, left_proj_act, right_proj_act,optimize='optimal')
        
        act = LayerNorm(axis = -1, create_scale=True, create_offset=True, name='center_layer_norm')(act)
        
        output_channel = int(input_act.shape[-1])
        
        act = common_modules.Linear(output_channel, initializer = utils.final_init(self.zero_init),
                                    name = 'output_projection')(act)
        
        gate_values = sigmoid(common_modules.Linear(output_channel, bias_init=1., 
                                            initializer=utils.final_init(self.zero_init),
                                            name='gating_linear')(input_act))
        
        act *= gate_values
        
        return act

class TriangleMultiplicationIncoming():
    
    def __init__(self, dropout_rate = 0.25, equation = 'kjc,kic->ijc', num_intermediate_channel = 128,
                 orientation = 'per_row', shared_dropout = True, deterministic = False,
                 multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False, 
                 name='triangle_multiplication_incoming'):
        
        self.dropout_rate = dropout_rate
        self.equation = equation
        self.num_intermediate_channel = num_intermediate_channel
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
    
    def __call__(self, act, mask, key, is_training=True):
        
        """Builds TriangleMultiplication module.
        Arguments:
          act: Pair activations, shape [N_res, N_res, c_z]
          mask: Pair mask, shape [N_res, N_res].
          is_training: Whether the module is in training mode.
        Returns:
          Outputs, same shape/type as act.
        """
        
        
        del is_training
        
        np.random.seed(key)
        mask = mask[..., None]
        
        act = LayerNorm(axis = -1, create_scale=True, create_offset=True,
                       name='layer_norm_input')(act)
        # print(act)
        input_act = act
        
        left_projection = common_modules.Linear(self.num_intermediate_channel,name = 'left_projection')
        
        left_proj_act = mask * left_projection(act)
        
        #print('left_proj_act: ', left_proj_act)
        right_projection = common_modules.Linear(self.num_intermediate_channel,name = 'right_projection')
        
        right_proj_act = mask * right_projection(act)
        #print('right_proj_act: ',right_proj_act)
        left_gate_values = sigmoid(common_modules.Linear(self.num_intermediate_channel,bias_init = 1., 
                                                 initializer = utils.final_init(self.zero_init),
                                                 name = 'left_gate')(act))
        #print('left_gate_values: ',left_gate_values)

        
        right_gate_values = sigmoid(common_modules.Linear(self.num_intermediate_channel, bias_init = 1., 
                                                  initializer = utils.final_init(self.zero_init),
                                                  name = 'right_gate')(act))
        
        #print('right_gate_values: ',right_gate_values)
        
        left_proj_act *= left_gate_values
        #print('left_proj_act: ',left_proj_act)
        right_proj_act *= right_gate_values
        #print('right_proj_act: ',right_proj_act)
        act = np.einsum(self.equation, left_proj_act, right_proj_act,optimize='optimal')
        #print('act: ',act)
        act = LayerNorm(axis = -1, create_scale=True, create_offset=True, name='center_layer_norm')(act)
        #print('act: ',act)
        output_channel = int(input_act.shape[-1])
        
        act = common_modules.Linear(output_channel, initializer = utils.final_init(self.zero_init),
                                    name = 'output_projection')(act)
        
        gate_values = sigmoid(common_modules.Linear(output_channel, bias_init=1., 
                                            initializer=utils.final_init(self.zero_init),
                                            name='gating_linear')(input_act))
        
        act *= gate_values
        
        return act


class TriangleAttentionStartingNode():
    
    def __init__(self, dropout_rate = 0.25, gating = True, num_head = 4, orientation = 'per_row', 
                 shared_dropout = True, deterministic = False,multimer_mode = False, 
                 subbatch_size = 4, use_remat = False, zero_init = False, name ='triangle_attention_starting_node'):
        
        self.dropout_rate = dropout_rate
        self.gating = gating
        self.num_head = num_head
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self, pair_act, pair_mask, key, is_training=False):
        
        """Builds TriangleAttention module.
        Arguments:
          pair_act: [N_res, N_res, c_z] pair activations tensor
          pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
          is_training: Whether the module is in training mode.
        Returns:
          Update to pair_act, shape [N_res, N_res, c_z].
        """
        
        np.random.seed(key)
        assert len(pair_act.shape) == 3
        assert len(pair_mask.shape) == 2
        
        bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4
        
        pair_act = LayerNorm(axis = -1, create_scale=True, create_offset=True, name='query_norm')(pair_act)
        
        init_factor = 1./ np.sqrt(int(pair_act.shape[-1]))
        
        weights = RandomNormal(stddev = init_factor)(shape = (pair_act.shape[-1], self.num_head),dtype = np.float64)
        
        nonbatched_bias = np.einsum('qkc,ch->hqk', pair_act, weights,optimize='optimal')

        attn_mod = Attention(output_dim = pair_act.shape[-1], dropout_rate=self.dropout_rate,
                             gating=self.gating, num_head = self.num_head, orientation=self.orientation,
                             shared_dropout=self.shared_dropout,deterministic=self.deterministic,
                             multimer_mode=self.multimer_mode,subbatch_size=self.subbatch_size,
                             use_remat=self.use_remat,zero_init=self.zero_init)
        
        pair_act = mapping.inference_subbatch(
            attn_mod,
            self.subbatch_size,
            batched_args=[pair_act, pair_act, bias],
            nonbatched_args=[nonbatched_bias],
            low_memory = False)
        
        return pair_act


class TriangleAttentionEndingNode():
    
    def __init__(self, dropout_rate = 0.25, gating = True, num_head = 4, orientation = 'per_column', 
                 shared_dropout = True, deterministic = False,multimer_mode = False, 
                 subbatch_size = 4, use_remat = False, zero_init = False, name = 'triangle_attention_ending_node'):
        
        self.dropout_rate = dropout_rate
        self.gating = gating
        self.num_head = num_head
        self.orientation = orientation
        self.shared_dropout = shared_dropout
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
    
    def __call__(self, pair_act, pair_mask, key, is_training=False):
        
        """Builds TriangleAttention module.
        Arguments:
          pair_act: [N_res, N_res, c_z] pair activations tensor
          pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
          is_training: Whether the module is in training mode.
        Returns:
          Update to pair_act, shape [N_res, N_res, c_z].
        """
        
        np.random.seed(key)
        assert len(pair_act.shape) == 3
        assert len(pair_mask.shape) == 2
        
        pair_act = np.swapaxes(pair_act, -2, -3)
        pair_mask = np.swapaxes(pair_mask, -1, -2)
        
        bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4
        
        pair_act = LayerNorm(axis = -1, create_scale=True, create_offset=True, name='query_norm')(pair_act)
        
        init_factor = 1. / np.sqrt(int(pair_act.shape[-1]))
        
        weights = RandomNormal(stddev = init_factor)(shape = (pair_act.shape[-1], self.num_head),dtype = np.float64)
        
        nonbatched_bias = np.einsum('qkc,ch->hqk', pair_act, weights,optimize='optimal')
        
        attn_mod = Attention(output_dim = pair_act.shape[-1], dropout_rate=self.dropout_rate,
                             gating=self.gating, num_head = self.num_head, orientation=self.orientation,
                             shared_dropout=self.shared_dropout,deterministic=self.deterministic,
                             multimer_mode=self.multimer_mode,subbatch_size=self.subbatch_size,
                             use_remat=self.use_remat,zero_init=self.zero_init)        
        
        pair_act = mapping.inference_subbatch(
            attn_mod,
            self.subbatch_size,
            batched_args=[pair_act, pair_act, bias],
            nonbatched_args=[nonbatched_bias],
            low_memory = False)
        
        pair_act = np.swapaxes(pair_act, -2, -3)
        
        return pair_act

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    """Compute distogram from amino acid positions.
      Arguments:
        positions: [N_res, 3] Position coordinates.
        num_bins: The number of bins in the distogram.
        min_bin: The left edge of the first bin.
        max_bin: The left edge of the final bin. The final bin catches
            everything larger than `max_bin`.
      Returns:
        Distogram with the specified number of bins.
      """
      
    def squared_difference(x, y):
        return np.square(x - y)
    
    lower_breaks = np.linspace(min_bin, max_bin, num_bins)
    lower_breaks = np.square(lower_breaks)
    upper_breaks = np.concatenate([lower_breaks[1:],
                                  np.array([1e8], dtype=np.float32)], axis=-1)
    
    dist2 = np.sum(
        squared_difference(
            np.expand_dims(positions, axis=-2),
            np.expand_dims(positions, axis=-3)),
        axis=-1, keepdims=True)
    
    dgram = ((dist2 > lower_breaks).astype(np.float32) *
             (dist2 < upper_breaks).astype(np.float32))
    
    return dgram
    

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    # print('is_gly: ',is_gly)
    # print('aatype: ', aatype.shape)
    # print('tile: ', np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]))
    # print('ca: ',all_atom_positions[..., ca_idx, :])
    pseudo_beta = np.where(
        np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(
            is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        
        return pseudo_beta, pseudo_beta_mask
    
    else:
        return pseudo_beta



class EvoformerIteration():
    
    def __init__(self, is_extra_msa, deterministic = False, multimer_mode = False, subbatch_size = 4, use_remat = False,
                 zero_init = False, name='evoformer_iteration'):
        
        self.is_extra_msa = is_extra_msa
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
        
    def __call__(self, activations, masks, is_training=True, safe_key=None):
        
        """Builds EvoformerIteration module.
        Arguments:
          activations: Dictionary containing activations:
            * 'msa': MSA activations, shape [N_seq, N_res, c_m].
            * 'pair': pair activations, shape [N_res, N_res, c_z].
          masks: Dictionary of masks:
            * 'msa': MSA mask, shape [N_seq, N_res].
            * 'pair': pair mask, shape [N_res, N_res].
          is_training: Whether the module is in training mode.
          safe_key: prng.SafeKey encapsulating rng key.
        Returns:
          Outputs, same shape/type as act.
        """
        
        msa_act, pair_act = activations['msa'], activations['pair']
        
        if safe_key is None:
            safe_key = np.random.default_rng()
        else:
            safe_key = np.random.default_rng(safe_key)
        safe_key = safe_key.integers(low = 0, high = 10000, size=10).tolist()
        
        msa_mask, pair_mask = masks['msa'], masks['pair']
        
        
        subkeys = iter(safe_key)
        
        
        
        
        outer_module = OuterProductMean(num_output_channel=int(pair_act.shape[-1]),first = False, chunk_size = 128, 
                                        dropout_rate = 0.0,num_outer_channel = 32, orientation = 'per_row', 
                                        shared_dropout = True,deterministic=self.deterministic,
                                        multimer_mode=self.multimer_mode,subbatch_size=self.subbatch_size,
                                        use_remat=self.use_remat,zero_init=self.zero_init,name = 'outer_product_mean')
        
        if outer_module.__dict__['first']:
            
            pair_act = dropout_wrapper(module=outer_module, input_act = msa_act, mask = msa_mask,
                                       safe_key=next(subkeys),is_training=True, output_act=pair_act)
        #print(pair_act)
        msarow_module = MSARowAttentionWithPairBias(dropout_rate = 0.15, gating = True, num_head = 8, 
                                                    orientation = 'per_row', shared_dropout = True, 
                                                    deterministic=self.deterministic,multimer_mode=self.multimer_mode,
                                                    subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                                    zero_init=self.zero_init,name = 'msa_row_attention_with_pair_bias')
        
        msa_act = dropout_wrapper(module = msarow_module, input_act = msa_act, mask = msa_mask, 
                                  safe_key = next(subkeys), is_training=True, pair_act = pair_act)
        #print(msa_act)
        if not self.is_extra_msa:
            attn_mod = MSAColumnAttention(dropout_rate = 0.0, gating = True, num_head = 8,
                                          orientation = 'per_column', shared_dropout = True, 
                                          deterministic=self.deterministic,multimer_mode=self.multimer_mode,
                                          subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                          zero_init=self.zero_init,name = 'msa_column_attention')
        else:
            attn_mod = MSAColumnGlobalAttention(dropout_rate = 0.0, gating = True, num_head = 8,
                                                orientation = 'per_column', shared_dropout = True, 
                                                deterministic=self.deterministic,multimer_mode=self.multimer_mode,
                                                subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                                zero_init=self.zero_init,name = 'msa_column_global_attention')
        
        msa_act = dropout_wrapper(module = attn_mod, input_act = msa_act, mask = msa_mask,
                                  safe_key = next(subkeys), is_training = True)
        #print(msa_act)
        msa_transition_module = Transition(dropout_rate = 0.0, num_intermediate_factor = 4, 
                                           orientation = 'per_row', shared_dropout = True, 
                                           deterministic=self.deterministic,multimer_mode=self.multimer_mode,
                                           subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                           zero_init=self.zero_init,name = 'msa_transition')
        
        
        msa_act = dropout_wrapper(module = msa_transition_module, input_act = msa_act, mask = msa_mask,
                                  safe_key = next(subkeys), is_training=True)
        
        #print(msa_act)
        
        if not outer_module.__dict__['first']:
            pair_act = dropout_wrapper(module = outer_module, input_act = msa_act, mask = msa_mask,
                                       safe_key = next(subkeys), output_act = pair_act, is_training = True)
            
        #print(pair_act)
        
        
        outgoing_module = TriangleMultiplicationOutgoing(dropout_rate = 0.25, equation = 'ikc,jkc->ijc', 
                                                         num_intermediate_channel = 128,orientation = 'per_row', 
                                                         shared_dropout = True, deterministic=self.deterministic,
                                                         multimer_mode=self.multimer_mode,
                                                         subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                                         zero_init=self.zero_init,
                                                         name = 'triangle_multiplication_outgoing')
        
        
        pair_act = dropout_wrapper(module = outgoing_module, input_act = pair_act, mask = pair_mask,
                                   safe_key = next(subkeys), is_training = True)        
        
        #print(pair_act)
        
        incoming_module = TriangleMultiplicationIncoming(dropout_rate = 0.25, equation = 'kjc,kic->ijc', 
                                                         num_intermediate_channel = 128,orientation = 'per_row', 
                                                         shared_dropout = True, deterministic=self.deterministic,
                                                         multimer_mode=self.multimer_mode,
                                                         subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                                         zero_init=self.zero_init,
                                                         name = 'triangle_multiplication_incoming')
        
        pair_act = dropout_wrapper(module = incoming_module, input_act = pair_act, mask = pair_mask,
                                   safe_key = next(subkeys), is_training = True)
        
        #print(pair_act)
        
        starting_module = TriangleAttentionStartingNode(dropout_rate = 0.25, gating = True, num_head = 4, 
                                                        orientation = 'per_row', shared_dropout = True,
                                                        deterministic=self.deterministic,multimer_mode=self.multimer_mode,
                                                        subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                                        zero_init=self.zero_init,name = 'triangle_attention_starting_node')
        
        pair_act = dropout_wrapper(module = starting_module, input_act = pair_act, mask = pair_mask,
                                   safe_key = next(subkeys), is_training = True)
        
        #print(pair_act)
        
        ending_module = TriangleAttentionEndingNode(dropout_rate = 0.25, gating = True, num_head = 4, 
                                                    orientation = 'per_column', shared_dropout = True,
                                                    deterministic=self.deterministic,multimer_mode=self.multimer_mode,
                                                    subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                                    zero_init=self.zero_init,name = 'triangle_attention_starting_node')
        
        pair_act = dropout_wrapper(module = ending_module, input_act = pair_act, mask = pair_mask,
                                   safe_key = next(subkeys), is_training = True)
        
        #print(pair_act)
        
        pair_transition_module = Transition(dropout_rate = 0.0, num_intermediate_factor = 4, 
                                            orientation = 'per_row', shared_dropout = True, 
                                            deterministic=self.deterministic,multimer_mode=self.multimer_mode,
                                            subbatch_size=self.subbatch_size,use_remat=self.use_remat,
                                            zero_init=self.zero_init,name = 'pair_transition')
        
        pair_act = dropout_wrapper(module = pair_transition_module, input_act = pair_act, mask = pair_mask,
                                   safe_key = next(subkeys), is_training = True)
        
        #print(pair_act)
        
        return {'msa': msa_act, 'pair': pair_act}
    
    
class EmbeddingsAndEvoformer():
    
    def __init__(self, evoformer_num_block = 48, extra_msa_channel = 64, extra_msa_stack_num_block = 4, 
                 max_relative_feature = 32, msa_channel = 256, pair_channel = 128, prev_pos_min_bin = 3.25, 
                 prev_pos_max_bin = 20.75, prev_pos_num_bins = 15, deterministic = False, recycle_features = True, 
                 recycle_pos = True, seq_channel = 384, multimer_mode = False, subbatch_size = 4, use_remat = False, 
                 zero_init = False, name='evoformer'):
        
        self.evoformer_num_block = evoformer_num_block
        self.extra_msa_channel = extra_msa_channel
        self.extra_msa_stack_num_block = extra_msa_stack_num_block
        self.max_relative_feature = max_relative_feature
        self.msa_channel = msa_channel
        self.pair_channel = pair_channel
        self.prev_pos_min_bin = prev_pos_min_bin
        self.prev_pos_max_bin = prev_pos_max_bin
        self.prev_pos_num_bins = prev_pos_num_bins
        self.recycle_features = recycle_features
        self.recycle_pos = recycle_pos
        self.deterministic = deterministic
        self.multimer_mode = False
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        self.name = name
    
    def __call__(self, batch, is_training, safe_key=None):
        
        if safe_key is None:
            safe_key = np.random.default_rng()
        else:
            safe_key = np.random.default_rng(safe_key)
        
        # Embed clustered MSA.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        
        preprocess_1d = common_modules.Linear(self.msa_channel, name = 'preprocess_1d')(batch['target_feat'])
        
        preprocess_msa = common_modules.Linear(self.msa_channel, name = 'preprocess_msa')(batch['msa_feat'])
        
        msa_activations = np.expand_dims(preprocess_1d, axis=0) + preprocess_msa
        
        left_single = common_modules.Linear(self.pair_channel, name = 'left_single')(batch['target_feat'])
        
        right_single = common_modules.Linear(self.pair_channel, name = 'right_single')(batch['target_feat'])
        
        pair_activations = left_single[:, None] + right_single[None]
        #print(pair_activations.dtype)
        mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
        #print(mask_2d.dtype)
        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        
        if self.recycle_pos:
            prev_pseudo_beta = pseudo_beta_fn(
                batch['aatype'], batch['prev_pos'], None)
            
            dgram = dgram_from_positions(prev_pseudo_beta, num_bins = self.prev_pos_num_bins, 
                                         min_bin = self.prev_pos_min_bin, max_bin = self.prev_pos_max_bin)
            
            pair_activations += common_modules.Linear(self.pair_channel, name='prev_pos_linear')(dgram)
            
            pair_activations = pair_activations.astype(mask_2d.dtype)
        
        
        return pair_activations




class SingleTemplateEmbedding():
    """Embeds a single template.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9+11
    """
    
    def __init__(self, embed_torsion_angles = True, enabled = True, max_templates = 4, template_subbatch_size = 128, 
                 use_template_unit_vector = False, pair_stack_triangle_ending_node_value_dim = 64, 
                 dgram_features_min_bin = 3.25, dgram_features_max_bin = 50.75, dgram_features_num_bins = 39,
                 deterministic = False, multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False, 
                 name='single_template_embedding'):
        
        self.embed_torsion_angles = embed_torsion_angles
        self.enabled = enabled
        self.max_templates = max_templates
        self.template_subbatch_size = template_subbatch_size
        self.use_template_unit_vector = use_template_unit_vector
        self.pair_stack_triangle_ending_node_value_dim = pair_stack_triangle_ending_node_value_dim
        self.dgram_features_min_bin = dgram_features_min_bin
        self.dgram_features_max_bin = dgram_features_max_bin
        self.dgram_features_num_bins = dgram_features_num_bins
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        
    def __call__(self, query_embedding, batch, mask_2d, is_training):
        
        """Build the single template embedding.
        Arguments:
          query_embedding: Query pair representation, shape [N_res, N_res, c_z].
          batch: A batch of template features (note the template dimension has been
            stripped out as this module only runs over a single template).
          mask_2d: Padding mask (Note: this doesn't care if a template exists,
            unlike the template_pseudo_beta_mask).
          is_training: Whether the module is in training mode.
        Returns:
          A template embedding [N_res, N_res, c_z].
        """
        
        length = batch['template_aatype'].shape[0]
        
        act_list = []
        for i in range(length):
            assert mask_2d.dtype == query_embedding.dtype
            dtype = query_embedding.dtype
            num_res = batch['template_aatype'][i].shape[0]
            
            num_channels = (self.pair_stack_triangle_ending_node_value_dim)
            
            template_mask = batch['template_pseudo_beta_mask'][i]
            
            template_mask_2d = template_mask[:, None] * template_mask[None, :]
            template_mask_2d = template_mask_2d.astype(dtype)
            
            template_dgram = dgram_from_positions(batch['template_pseudo_beta'][i], num_bins = self.dgram_features_num_bins,
                                                  min_bin = self.dgram_features_min_bin, max_bin = self.dgram_features_max_bin)
            
            template_dgram = template_dgram.astype(dtype)
            
            to_concat = [template_dgram, template_mask_2d[:, :, None]]
            
            aatype = one_hot(x = batch['template_aatype'][i], num_classes = 22, dtype = dtype)
            
            to_concat.append(np.tile(aatype[None, :, :], [num_res, 1, 1]))
            to_concat.append(np.tile(aatype[:, None, :], [1, num_res, 1]))
            
            n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
    
            rot, trans = quat_affine.make_transform_from_reference(
                n_xyz=batch['template_all_atom_positions'][i][:, n],
                ca_xyz=batch['template_all_atom_positions'][i][:, ca],
                c_xyz=batch['template_all_atom_positions'][i][:, c])
            
            affines = quat_affine.QuatAffine(
                quaternion=quat_affine.rot_to_quat(rot, unstack_inputs=True),
                translation=trans,
                rotation=rot,
                unstack_inputs=True)
            
            points = [np.expand_dims(x, axis=-2) for x in affines.translation]
            affine_vec = affines.invert_point(points, extra_dims=1)
            inv_distance_scalar = 1. / np.sqrt(1e-6 + sum([np.square(x) for x in affine_vec]))
            
            # Backbone affine mask: whether the residue has C, CA, N
            # (the template mask defined above only considers pseudo CB).
            
            template_mask = (
                batch['template_all_atom_masks'][i][..., n] *
                batch['template_all_atom_masks'][i][..., ca] *
                batch['template_all_atom_masks'][i][..., c])
            
            template_mask_2d = template_mask[:, None] * template_mask[None, :]
            
            inv_distance_scalar *= template_mask_2d.astype(inv_distance_scalar.dtype)
            
            unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]
            
            unit_vector = [x.astype(dtype) for x in unit_vector]
            
            template_mask_2d = template_mask_2d.astype(dtype)
            
            if not self.use_template_unit_vector:
                unit_vector = [np.zeros_like(x) for x in unit_vector]
            
            to_concat.extend(unit_vector)
            
            to_concat.append(template_mask_2d[..., None])
            
            act = np.concatenate(to_concat, axis=-1)
            
            # Mask out non-template regions so we don't get arbitrary values in the
            # distogram for these regions.
            
            act *= template_mask_2d[..., None]
            
            # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 9
            act = common_modules.Linear(
                num_channels,
                initializer='relu',
                name='embedding2d')(
                    act).astype(dtype)
            
            # pairstack is here        
                    
                    
            act_list.append(act)
        act = np.stack(act_list)
        return act

class TemplateEmbedding():
    """Embeds a set of templates.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12
    Jumper et al. (2021) Suppl. Alg. 17 "TemplatePointwiseAttention"
    """
    
    def __init__(self, embed_torsion_angles = True, enabled = True, max_templates = 4, template_subbatch_size = 128, 
                 use_template_unit_vector = False, pair_stack_triangle_ending_node_value_dim = 64, 
                 dgram_features_min_bin = 3.25, dgram_features_max_bin = 50.75, dgram_features_num_bins = 39,
                 deterministic = False, multimer_mode = False, subbatch_size = 4, use_remat = False, zero_init = False, 
                 name='template_embedding'):
        
        self.embed_torsion_angles = embed_torsion_angles
        self.enabled = enabled
        self.max_templates = max_templates
        self.template_subbatch_size = template_subbatch_size
        self.use_template_unit_vector = use_template_unit_vector
        self.pair_stack_triangle_ending_node_value_dim = pair_stack_triangle_ending_node_value_dim
        self.dgram_features_min_bin = dgram_features_min_bin
        self.dgram_features_max_bin = dgram_features_max_bin
        self.dgram_features_num_bins = dgram_features_num_bins
        self.deterministic = deterministic
        self.multimer_mode = multimer_mode
        self.subbatch_size = subbatch_size
        self.use_remat = use_remat
        self.zero_init = zero_init
        
    def __call__(self, query_embedding, template_batch, mask_2d, is_training):
        """Build TemplateEmbedding module.
        Arguments:
          query_embedding: Query pair representation, shape [N_res, N_res, c_z].
          template_batch: A batch of template features.
          mask_2d: Padding mask (Note: this doesn't care if a template exists,
            unlike the template_pseudo_beta_mask).
          is_training: Whether the module is in training mode.
        Returns:
          A template embedding [N_res, N_res, c_z].
        """
        
        num_templates = template_batch['template_mask'].shape[0]
        
        num_channels = (self.pair_stack_triangle_ending_node_value_dim)
        
        num_res = query_embedding.shape[0]
        
        dtype = query_embedding.dtype
        template_mask = template_batch['template_mask']
        template_mask = template_mask.astype(dtype)
        
        query_num_channels = query_embedding.shape[-1]
        
        # Make sure the weights are shared across templates by constructing the
        # embedder here.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12

    
# ln = LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)

# x = np.asarray([1,2,3,4,5,6,8,10,12,14,16,18]).astype(np.float64)
# x = np.reshape(x,(2,2,3))
# print(x)
# result = ln(x)

# print(result)



# x = np.array([-8,-6,-4,-1,0,1,4,6,8,10,12,14])
# x = np.reshape(x,(2,2,3))

# print(erf(x))

# a = truncated_normal(-2.,2., dtype = np.float64, shape = (2,2))
# print(a)

# b = TruncatedNormal()(shape = (8,8), dtype = np.float64)
# print(type(b))

# c = VarianceScaling(scale=2., mode='fan_in')(shape = (8,3), dtype = np.float64)
# print(type(c))

# q_data = np.random.normal(size = (8,8,64))
# m_data = np.random.normal(size = (8,8,64))
# bias = np.random.normal(size = (8,8,8))
# q_mask = np.random.normal(size = (8,8,64))

# attn_mod = Attention(output_dim = 4)

# out = attn_mod(q_data, m_data, bias)
# print(out)

# newout = mapping.inference_subbatch(
#         attn_mod,
#         subbatch_size = 4,
#         batched_args=[q_data, q_data, bias],
#         nonbatched_args=[bias],
#         low_memory=False)

# print(newout.shape)

# globalattn_mod = GlobalAttention(output_dim=4,zero_init=False)
# out = globalattn_mod(q_data,m_data,q_mask)
# print(out)

# np.random.seed(49)
# msa_act = np.random.normal(size = (8,8,64)).astype(np.float64)
# #print(msa_act)
# msa_mask = np.random.normal(size = (8,8)).astype(np.float64)
# pair_act = np.random.normal(size = (8,8,64)).astype(np.float64)

# key = 66

# msa_act = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
#                     26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]).astype(np.float64)
# msa_act = np.reshape(msa_act,(2,3,8))

# msa_mask = np.array([1,2,3,4,5,6]).astype(np.float64)
# msa_mask = np.reshape(msa_mask,(2,3))

# pair_act = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
#                     26,27,28,29,30,31,32,33,34,35,36]).astype(np.float64)
# pair_act = np.reshape(pair_act,(3,3,4))

# pair_mask = np.array([1,2,3,4,5,6,7,8,9]).astype(np.float64)
# pair_mask = np.reshape(pair_mask,(3,3))

# activations = {'msa':msa_act, 'pair': pair_act}
# masks = {'msa':msa_mask, 'pair':pair_mask}



# output = MSARowAttentionWithPairBias(zero_init=False)(msa_act, msa_mask, pair_act,key)
# print(output)

# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.15, is_training=True,broadcast_dim=0)
# print(residual)

# output = MSAColumnAttention(zero_init=False)(msa_act, msa_mask,key)
# print(output)


# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.0, is_training=True,broadcast_dim=0)
# print(residual)


# output = MSAColumnGlobalAttention(zero_init=False)(msa_act, msa_mask,key)
# print(output)

# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.0, is_training=True,broadcast_dim=0)
# print(residual)


# output = Transition(zero_init=False)(msa_act,msa_mask,key)
# print(output)

# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.0, is_training=True,broadcast_dim=0)
# print(residual)


# output = OuterProductMean(num_output_channel=int(pair_act.shape[-1]),zero_init=False)(msa_act,msa_mask,key)
# print(output)


# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.0, is_training=True,broadcast_dim=0)
# print(residual)


# output = TriangleMultiplicationOutgoing(zero_init=False)(pair_act, pair_mask,key)
# print(output)


# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.25, is_training=True,broadcast_dim=0)
# print(residual)



# output = TriangleMultiplicationIncoming(zero_init=False)(pair_act, pair_mask,key)
# print(output)

# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.25, is_training=True,broadcast_dim=0)
# print(residual)

# output = TriangleAttentionStartingNode(zero_init=False)(pair_act,pair_mask,key)
# print(output)

# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.25, is_training=True,broadcast_dim=0)
# print(residual)

# output = TriangleAttentionEndingNode(zero_init=False)(pair_act,pair_mask,key)
# print(output)

# residual = apply_dropout(tensor=output,safe_key=key, rate = 0.25, is_training=True,broadcast_dim=0)
# print(residual)

# module = MSARowAttentionWithPairBias(zero_init=False)
# print(module.__dict__['deterministic'])
# dropout = dropout_wrapper(module = module, input_act = msa_act, mask = msa_mask, safe_key = key, is_training=True,pair_act = pair_act)
# print(dropout)

# module = MSAColumnAttention(zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = msa_act, mask = msa_mask, safe_key = key, is_training=True)
# print(dropout)

# module = MSAColumnGlobalAttention(zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = msa_act, mask = msa_mask, safe_key = key, is_training=True)
# print(dropout)

# module = Transition(zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = msa_act, mask = msa_mask, safe_key = key, is_training=True)
# print(dropout)

# module = OuterProductMean(num_output_channel=int(pair_act.shape[-1]), zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = msa_act, mask = msa_mask, safe_key = key, is_training=True,
#                           output_act=pair_act)
# print(dropout)

# module = TriangleMultiplicationOutgoing(zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = pair_act, mask = pair_mask, safe_key = key, is_training=True)
# print(dropout)

# module = TriangleMultiplicationIncoming(zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = pair_act, mask = pair_mask, safe_key = key, is_training=True)
# print(dropout)

# module = TriangleAttentionStartingNode(zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = pair_act, mask = pair_mask, safe_key = key, is_training=True)
# print(dropout)

# module = TriangleAttentionEndingNode(zero_init=False)
# dropout = dropout_wrapper(module = module, input_act = pair_act, mask = pair_mask, safe_key = key, is_training=True)
# print(dropout)

# module = EvoformerIteration(is_extra_msa=True,zero_init=False)
# output = module(activations = activations, masks = masks,safe_key=key)
# print(output)

file = open('./params/batch0_model1.pkl','rb')
batch = pickle.load(file)
file.close()
# print(batch)
# print('\n')

# module = EmbeddingsAndEvoformer(zero_init=False)
# output = module(batch, is_training=True)
# print(output)

# nb_classes = 6
# target = np.array([[2,3,4,0]])

# result = np.eye(nb_classes)[target]
# result2 = one_hot(target, 6)
# print(result)
# print(result2)

template_batch = {k: batch[k] for k in batch if k.startswith('template_')}

aatype = one_hot(x = template_batch['template_aatype'], num_classes = 22)
print(aatype.shape)

query_embedding = EmbeddingsAndEvoformer(zero_init=False)(batch, is_training=True)
print(query_embedding.shape)
template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
print(template_batch['template_all_atom_positions'][0][:,2].shape)
mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

module = SingleTemplateEmbedding(zero_init=False)
output = module(query_embedding, template_batch, mask_2d, is_training=True)
print(output.shape)

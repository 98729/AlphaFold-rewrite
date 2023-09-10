import numpy as np
from LayerNorm import LayerNorm
from attention import Attention


class MSAColumnAttention:
    """MSA per-column attention.

    Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
    """

    def __init__(self, q_data, m_data, bias, create_scale: bool, create_offset: bool,
                 scale_init=None, offset_init=None, nonbatched_bias=None, zero_init=True, gating=True):
        self.subbatch_size = 4
        self.dropout_rate = 0.0
        self.gating = True
        self.num_head = 8
        self.orientation = 'per_column'
        self.shared_dropout = True

        self.create_scale = create_scale
        self.create_offset = create_offset
        self.scale_init = scale_init or np.ones
        self.offset_init = offset_init or np.zeros

        self.q_data = q_data
        self.m_data = m_data
        self.bias = bias
        self.nonbatched_bias = nonbatched_bias
        self.num_head = 8
        self.zero_init = zero_init
        self.gating = gating

    def __call__(self, msa_act, msa_mask, is_training=False):
        """Builds MSAColumnAttention module.
        Arguments:
          msa_act: [N_seq, N_res, c_m] MSA representation.
          msa_mask: [N_seq, N_res] mask of non-padded regions.
          is_training: Whether the module is in training mode.
        Returns:
          Update to msa_act, shape [N_seq, N_res, c_m]
        """

        assert len(msa_act.shape) == 3
        assert len(msa_mask.shape) == 2
        assert self.orientation == 'per_column'
        self.output_dim = msa_act.shape[-1]

        msa_act = np.swapaxes(msa_act, -2, -3)
        msa_mask = np.swapaxes(msa_mask, -1, -2)

        bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4

        # msa_act layer norm
        # for i in range(len(msa_act)):
        #     for j in range(len(msa_act[i])):
        #         avg = np.sum(msa_act[i][j]) / len(msa_act[i][j])
        #         var = 0
        #         for k in range(len(msa_act[i][j])):
        #             var += np.power(msa_act[i][j][k] - avg, 2)
        #         var /= len(msa_act[i][j])
        #         for k in range(len(msa_act[i][j])):
        #             msa_act[i][j][k] = (msa_act[i][j][k] - avg) / np.sqrt(var)

        msa_act = LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='query_norm')(msa_act)

        # attention
        # attn_mod = Attention(
        #     c, self.global_config, msa_act.shape[-1])

        attn_mod = Attention(self.output_dim, self.q_data, self.m_data, self.gating)

        # key_dim = int(self.q_data.shape[-1])
        # value_dim = int(self.m_data.shape[-1])
        # num_head = self.num_head
        # assert key_dim % num_head == 0
        # assert value_dim % num_head == 0
        # key_dim = key_dim // num_head
        # value_dim = value_dim // num_head
        #
        # # weights for queues, keys, and values
        # # q_fan_in = q_data.shape[0] * q_data.shape[1] * q_data.shape[2]
        # # q_fan_out = q_data.shape[0] * q_data.shape[1] * num_head * key_dim
        # # q_limit = np.sqrt(6 / (q_fan_in + q_fan_out))
        # # q_weights = np.random.uniform(-q_limit, q_limit, size=(q_data.shape[-1], num_head, key_dim))
        # q_weights = glorot_uniform(shape=(q_data.shape[-1], num_head, key_dim))
        #
        # # k_fan_in = m_data.shape[0] * m_data.shape[1] * m_data.shape[2]
        # # k_fan_out = m_data.shape[0] * m_data.shape[1] * num_head * key_dim
        # # k_limit = np.sqrt(6 / (k_fan_in + k_fan_out))
        # # k_weights = np.random.uniform(-k_limit, k_limit, size=(m_data.shape[-1], num_head, key_dim))
        # k_weights = glorot_uniform(shape=(m_data.shape[-1], num_head, key_dim))
        #
        # # v_fan_in = m_data.shape[0] * m_data.shape[1] * m_data.shape[2]
        # # v_fan_out = m_data.shape[0] * m_data.shape[1] * num_head * key_dim
        # # v_limit = np.sqrt(6 / (v_fan_in + v_fan_out))
        # # v_weights = np.random.uniform(-v_limit, v_limit, size=(m_data.shape[-1], num_head, key_dim))
        # v_weights = glorot_uniform(shape=(m_data.shape[-1], num_head, key_dim))
        #
        # q = np.einsum('bqa,ahc->bqhc', q_data, np.array([q_weights])) * key_dim ** (-0.5)
        # k = np.einsum('bka,ahc->bkhc', m_data, np.array([k_weights]))
        # v = np.einsum('bka,ahc->bkhc', m_data, np.array([v_weights]))
        # logits = np.einsum('bqhc,bkhc->bhqk', q, k) + bias
        #
        # if self.nonbatched_bias is not None:
        #     logits += np.expand_dims(self.nonbatched_bias, axis=0)
        # weights = np.exp(logits) / np.sum(np.exp(logits))
        # weighted_avg = np.einsum('bhqk,bkhc->bqhc', weights, v)
        #
        # if self.gating:
        #     gating_weights = np.zeros(shape=(q_data.shape[-1], num_head, value_dim))
        #     gating_bias = np.ones(shape=(num_head, value_dim))
        #     gate_values = np.einsum('bqc, chv->bqhv', q_data, gating_weights) + gating_bias
        #     gate_values = 1 / (1 + np.exp(-gate_values))
        #     weighted_avg *= gate_values
        #
        # if self.zero_init:
        #     o_weights = np.zeros(shape=(num_head, value_dim, self.output_dim))
        # else:
        #     # o_fan_in = weighted_avg.shape[0] * weighted_avg.shape[1] * num_head * value_dim
        #     # o_fan_out = weighted_avg.shape[0] * weighted_avg.shape[1] * self.output_dim
        #     # o_limit = np.sqrt(6 / (o_fan_in + o_fan_out))
        #     # o_weights = np.random.uniform(-o_limit, o_limit, size=(num_head, value_dim, self.output_dim))
        #     o_weights = glorot_uniform(shape=(num_head, value_dim, self.output_dim))
        # o_bias = np.zeros(shape=(self.output_dim,))
        #
        # output = np.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

        # msa_act = inference_subbatch(
        #     attn_mod,
        #     self.subbatch_size,  # 4
        #     batched_args=[msa_act, msa_act, bias],
        #     nonbatched_args=[],
        #     low_memory=not is_training)

        msa_act = np.swapaxes(msa_act, -2, -3)

        return msa_act

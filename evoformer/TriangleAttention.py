import numpy as np
from LayerNorm import LayerNorm
from RandomNormal import RandomNormal
from attention import Attention


class TriangleAttention:
    """Triangle Attention.
    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """

    def __init__(self, q_data, m_data, orientation, nonbatched_bias=None, zero_init=True,
                 gating=True, name='triangle_attention'):
        super().__init__(name=name)
        self.orientation = orientation
        self.num_head = 4
        self.q_data = q_data
        self.m_data = m_data
        self.gating = gating
        self.nonbatched_bias = nonbatched_bias
        self.zero_init = zero_init

    def __call__(self, pair_act, pair_mask, is_training=False):
        """Builds TriangleAttention module.
        Arguments:
            pair_act: [N_res, N_res, c_z] pair activations tensor
            pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
            is_training: Whether the module is in training mode.
        Returns:
        Update to pair_act, shape [N_res, N_res, c_z].
        """

        assert len(pair_act.shape) == 3
        assert len(pair_mask.shape) == 2
        assert self.orientation in ['per_row', 'per_column']
        self.output_dim = pair_act.shape[-1]

        if self.orientation == 'per_column':
            pair_act = np.swapaxes(pair_act, -2, -3)
            pair_mask = np.swapaxes(pair_mask, -1, -2)

        bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4

        pair_act = LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='query_norm')(pair_act)

        init_factor = 1. / np.sqrt(int(pair_act.shape[-1]))
        # weights = hk.get_parameter(
        #     'feat_2d_weights',
        #     shape=(pair_act.shape[-1], c.num_head),
        #     init=hk.initializers.RandomNormal(stddev=init_factor))
        weights = RandomNormal(stddev=init_factor)((pair_act.shape[-1], self.num_head))
        nonbatched_bias = np.einsum('qkc,ch->hqk', pair_act, weights)

        # attention
        # attn_mod = Attention(
        #     c, self.global_config, pair_act.shape[-1])

        attn_mod = Attention(self.output_dim, self.q_data, self.m_data, self.gating, num_head=self.num_head)

        # pair_act = mapping.inference_subbatch(
        #     attn_mod,
        #     self.global_config.subbatch_size,
        #     batched_args=[pair_act, pair_act, bias],
        #     nonbatched_args=[nonbatched_bias],
        #     low_memory=not is_training)

        if self.orientation == 'per_column':
            pair_act = np.swapaxes(pair_act, -2, -3)

        return pair_act

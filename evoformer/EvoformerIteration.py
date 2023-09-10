from numpy import np
from OuterProductMean import OuterProductMean
from MSARowAttentionWithPairBias import MSARowAttentionWithPairBias
from MSAColumnAttention import MSAColumnAttention
from Transition import Transition
from TriangleMultiplicationOutgoing import TriangleMultiplicationOutgoing
from TriangleMultiplicationIncoming import TriangleMultiplicationIncoming
from TriangleAttention import TriangleAttention
import functools


def apply_dropout(*, tensor, safe_key, rate, is_training, broadcast_dim=None):
    """Applies dropout to a tensor."""
    if is_training and rate != 0.0:
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        keep_rate = 1.0 - rate
        # keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
        keep = np.random.binomial(1, p=keep_rate, size=shape)
        return keep * tensor / keep_rate
    else:
        return tensor


def dropout_wrapper(module,
                    input_act,
                    mask,
                    safe_key,
                    global_config,
                    output_act=None,
                    is_training=True,
                    **kwargs):
    """Applies module + dropout + residual update."""
    if output_act is None:
        output_act = input_act

    gc = global_config
    residual = module(input_act, mask, is_training=is_training, **kwargs)
    dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

    if module.config.shared_dropout:
        if module.config.orientation == 'per_row':
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


class EvoformerIteration:
    """Single iteration (block) of Evoformer stack.
    Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack" lines 2-10
    """

    def __init__(self, config, global_config, is_extra_msa, name='evoformer_iteration'):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa

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
        c = self.config
        gc = self.global_config

        msa_act, pair_act = activations['msa'], activations['pair']

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        msa_mask, pair_mask = masks['msa'], masks['pair']

        dropout_wrapper_fn = functools.partial(
            dropout_wrapper,
            is_training=is_training,
            global_config=gc)

        safe_key, *sub_keys = safe_key.split(10)
        sub_keys = iter(sub_keys)

        outer_module = OuterProductMean(
            config=c.outer_product_mean,
            global_config=self.global_config,
            num_output_channel=int(pair_act.shape[-1]),
            name='outer_product_mean')
        if c.outer_product_mean.first:
            pair_act = dropout_wrapper_fn(
                outer_module,
                msa_act,
                msa_mask,
                safe_key=next(sub_keys),
                output_act=pair_act)

        msa_act = dropout_wrapper_fn(
            MSARowAttentionWithPairBias(
                c.msa_row_attention_with_pair_bias, gc,
                name='msa_row_attention_with_pair_bias'),
            msa_act,
            msa_mask,
            safe_key=next(sub_keys),
            pair_act=pair_act)

        if not self.is_extra_msa:
            attn_mod = MSAColumnAttention(
                c.msa_column_attention, gc, name='msa_column_attention')
        else:
            attn_mod = MSAColumnGlobalAttention(
                c.msa_column_attention, gc, name='msa_column_global_attention')
        msa_act = dropout_wrapper_fn(
            attn_mod,
            msa_act,
            msa_mask,
            safe_key=next(sub_keys))

        msa_act = dropout_wrapper_fn(
            Transition(c.msa_transition, gc, name='msa_transition'),
            msa_act,
            msa_mask,
            safe_key=next(sub_keys))

        if not c.outer_product_mean.first:
            pair_act = dropout_wrapper_fn(
                outer_module,
                msa_act,
                msa_mask,
                safe_key=next(sub_keys),
                output_act=pair_act)

        pair_act = dropout_wrapper_fn(
            TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                                   name='triangle_multiplication_outgoing'),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys))
        pair_act = dropout_wrapper_fn(
            TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                                   name='triangle_multiplication_incoming'),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys))

        pair_act = dropout_wrapper_fn(
            TriangleAttention(c.triangle_attention_starting_node, gc,
                              name='triangle_attention_starting_node'),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys))
        pair_act = dropout_wrapper_fn(
            TriangleAttention(c.triangle_attention_ending_node, gc,
                              name='triangle_attention_ending_node'),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys))

        pair_act = dropout_wrapper_fn(
            Transition(c.pair_transition, gc, name='pair_transition'),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys))

        return {'msa': msa_act, 'pair': pair_act}

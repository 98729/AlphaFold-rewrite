import numpy as np
from MSAColumnAttention import LayerNorm
from OuterProductMean import Linear


class TriangleMultiplicationOutgoing:
    """Triangle multiplication layer ("outgoing" or "incoming").
    Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
    Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
    """

    def __init__(self, config, global_config, name='triangle_multiplication'):
        super().__init__(name=name)
        self.num_intermediate_channel = 128

    def __call__(self, act, mask, is_training=True):
        """Builds TriangleMultiplication module.
        Arguments:
            act: Pair activations, shape [N_res, N_res, c_z]
            mask: Pair mask, shape [N_res, N_res].
            is_training: Whether the module is in training mode.
        Returns:
        Outputs, same shape/type as act.
        """
        del is_training

        mask = mask[..., None]

        act = LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='layer_norm_input')(act)
        input_act = act

        # left_projection = Linear(
        #     self.num_intermediate_channel,
        #     name='left_projection')
        # left_proj_act = mask * left_projection(act)
        left_proj_act = mask * Linear(self.num_intermediate_channel)(act)

        # right_projection = Linear(
        #     self.num_intermediate_channel,
        #     name='right_projection')
        # right_proj_act = mask * right_projection(act)
        right_proj_act = mask * Linear(self.num_intermediate_channel)(act)

        # left_gate_values = jax.nn.sigmoid(Linear(
        #     self.num_intermediate_channel,
        #     bias_init=1.,
        #     initializer=utils.final_init(gc),
        #     name='left_gate')(act))
        act = Linear(self.num_intermediate_channel, bias_init=1., initializer='zeros')(act)
        left_gate_values = 1 / (1 + np.exp(-act))

        # right_gate_values = jax.nn.sigmoid(Linear(
        #     self.num_intermediate_channel,
        #     bias_init=1.,
        #     initializer=utils.final_init(gc),
        #     name='right_gate')(act))
        right_gate_values = 1 / (1 + np.exp(-act))

        left_proj_act *= left_gate_values
        right_proj_act *= right_gate_values

        # "Outgoing" edges equation: 'ikc,jkc->ijc'
        # "Incoming" edges equation: 'kjc,kic->ijc'
        # Note on the Suppl. Alg. 11 & 12 notation:
        # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
        # For the "incoming" edges, it's swapped:
        #   b = left_proj_act and a = right_proj_act
        act = np.einsum('ikc,jkc->ijc', left_proj_act, right_proj_act)  # using outgoing edges

        act = LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='center_layer_norm')(act)

        output_channel = int(input_act.shape[-1])

        # act = Linear(
        #     output_channel,
        #     initializer=utils.final_init(gc),
        #     name='output_projection')(act)
        act = Linear(output_channel, initializer='zeros')(act)

        # gate_values = jax.nn.sigmoid(Linear(
        #     output_channel,
        #     bias_init=1.,
        #     initializer=utils.final_init(gc),
        #     name='gating_linear')(input_act))
        input_act = Linear(output_channel, bias_init=1., initializer='zeros')(input_act)
        gate_values = 1 / (1 + np.exp(-input_act))
        act *= gate_values

        return act

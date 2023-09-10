import numpy as np
from MSAColumnAttention import LayerNorm
import numbers
from MSAColumnAttention import VarianceScaling, TruncatedNormal

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978, dtype=np.float32)


# def get_initializer_scale(initializer_name, input_shape):
#     """Get Initializer for weights and scale to multiply activations by."""
#
#     if initializer_name == 'zeros':
#         w_init = np.zeros(shape=input_shape)
#     else:
#         # fan-in scaling
#         scale = 1.
#         for channel_dim in input_shape:
#             scale /= channel_dim
#         if initializer_name == 'relu':
#             scale *= 2
#
#         noise_scale = scale
#
#         stddev = np.sqrt(noise_scale)
#         # Adjust stddev for truncation.
#         stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
#         w_init = TruncatedNormal(mean=0.0, stddev=stddev)
#
#     return w_init


class Linear:
    """Protein folding specific Linear module.
    This differs from the standard Haiku Linear in a few ways:
        * It supports inputs and outputs of arbitrary rank
        * Initializers are specified by strings
    """

    def __init__(self,
                 num_output,
                 initializer: str = 'linear',
                 num_input_dims: int = 1,
                 use_bias: bool = True,
                 bias_init: float = 0.,
                 precision=None,
                 name: str = 'linear'):
        """Constructs Linear Module.
        Args:
            num_output: Number of output channels. Can be tuple when outputting
            multiple dimensions.
            initializer: What initializer to use, should be one of {'linear', 'relu',
                'zeros'}
            num_input_dims: Number of dimensions from the end to project.
            use_bias: Whether to include trainable bias
            bias_init: Value used to initialize bias.
            precision: What precision to use for matrix multiplication, defaults
                to None.
            name: Name of module, used for name scopes.
        """
        super().__init__(name=name)
        if isinstance(num_output, numbers.Integral):
            self.output_shape = (num_output,)
        else:
            self.output_shape = tuple(num_output)
        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.num_input_dims = num_input_dims
        self.num_output_dims = len(self.output_shape)
        self.precision = precision

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

        # weight_init = get_initializer_scale(self.initializer, in_shape)
        weight_shape = in_shape + self.output_shape
        # weights = hk.get_parameter('weights', weight_shape, inputs.dtype, weight_init)

        if self.initializer == 'zeros':
            weights = np.zeros(shape=weight_shape)
        else:
            # fan-in scaling
            scale = 1.
            for channel_dim in in_shape:
                scale /= channel_dim
            if self.initializer == 'relu':
                scale *= 2
            noise_scale = scale
            stddev = np.sqrt(noise_scale)
            # Adjust stddev for truncation.
            stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
            weights = TruncatedNormal(stddev=stddev, mean=0.0)(weight_shape)

        in_letters = 'abcde'[:self.num_input_dims]
        out_letters = 'hijkl'[:self.num_output_dims]
        equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

        output = np.einsum(equation, inputs, weights, precision=self.precision)

        if self.use_bias:
            # bias = hk.get_parameter('bias', self.output_shape, inputs.dtype,
            #                         hk.initializers.Constant(self.bias_init))
            num = 1
            for i in range(len(self.output_shape)):
                num *= self.output_shape[i]
            bias = np.array([self.bias_init] * num, dtype=inputs.dtype).reshape(shape=self.output_shape)
            output += bias

        return output


class OuterProductMean:
    """Computes mean outer product.
    Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
    """

    def __init__(self,
                 num_output_channel,
                 name='outer_product_mean'):
        super().__init__(name=name)
        self.num_output_channel = num_output_channel
        self.num_outer_channel = 32
        self.zero_init = True

    def __call__(self, act, mask, is_training=True, zero_init=True):
        """Builds OuterProductMean module.
        Arguments:
            act: MSA representation, shape [N_seq, N_res, c_m].
            mask: MSA mask, shape [N_seq, N_res].
            is_training: Whether the module is in training mode.
        Returns:
        Update to pair representation, shape [N_res, N_res, c_z].
        """

        mask = mask[..., None]
        act = LayerNorm([-1], create_scale=True, create_offset=True, name='layer_norm_input')(act)

        left_act = mask * Linear(
            self.num_outer_channel,
            initializer='linear',
            name='left_projection')(
            act)

        right_act = mask * Linear(
            self.num_outer_channel,
            initializer='linear',
            name='right_projection')(
            act)

        if self.zero_init:
            output_w = np.zeros(shape=(self.num_outer_channel, self.num_outer_channel,
                                       self.num_output_channel))
        else:
            output_w = VarianceScaling(scale=2., mode='fan_in',
                                       shape=(self.num_outer_channel, self.num_outer_channel, self.num_output_channel))

        # output_w = hk.get_parameter(
        #     'output_w',
        #     shape=(c.num_outer_channel, c.num_outer_channel,
        #            self.num_output_channel),
        #     init=init_w)
        # output_b = hk.get_parameter(
        #     'output_b', shape=(self.num_output_channel,),
        #     init=hk.initializers.Constant(0.0))
        output_b = np.zeros(shape=(self.num_output_channel,))

        def compute_chunk(left_act):
            # This is equivalent to
            #
            # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
            # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
            #
            # but faster.
            left_act = np.transpose(left_act, [0, 2, 1])
            act = np.einsum('acb,ade->dceb', left_act, right_act)
            act = np.einsum('dceb,cef->dbf', act, output_w) + output_b
            return np.transpose(act, [1, 0, 2])

        # act = mapping.inference_subbatch(
        #     compute_chunk,
        #     c.chunk_size,
        #     batched_args=[left_act],
        #     nonbatched_args=[],
        #     low_memory=True,
        #     input_subbatch_dim=1,
        #     output_subbatch_dim=0)

        epsilon = 1e-3
        norm = np.einsum('abc,adc->bdc', mask, mask)
        act /= epsilon + norm

        return act

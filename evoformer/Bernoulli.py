import numpy as np

_dtype_to_32bit_dtype = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}


@functools.lru_cache(maxsize=None)
def _canonicalize_dtype(x64_enabled, dtype):
    """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
    try:
        dtype = np.dtype(dtype)
    except TypeError as e:
        raise TypeError(f'dtype {dtype!r} not understood') from e

    if x64_enabled:
        return dtype
    else:
        return _dtype_to_32bit_dtype.get(dtype, dtype)


def canonicalize_dtype(dtype):
    return _canonicalize_dtype(config.x64_enabled, dtype)


def _check_shape(name, shape: Union[Sequence[int], NamedShape], *param_shapes):
    shape = core.as_named_shape(shape)

    if param_shapes:
        shape_ = lax.broadcast_shapes(shape.positional, *param_shapes)
        if shape.positional != shape_:
            msg = ("{} parameter shapes must be broadcast-compatible with shape "
                   "argument, and the result of broadcasting the shapes must equal "
                   "the shape argument, but got result {} for shape argument {}.")
            raise ValueError(msg.format(name, shape_, shape))


def uniform(key: jnp.ndarray,
            shape: Union[Sequence[int], NamedShape] = (),
            dtype: DTypeLikeFloat = dtypes.float_,
            minval: RealArray = 0.,
            maxval: RealArray = 1.) -> jnp.ndarray:
    """Sample uniform random values in [minval, maxval) with given shape/dtype.
    Args:
        key: a PRNGKey used as the random key.
        shape: optional, a tuple of nonnegative integers representing the result
            shape. Default ().
        dtype: optional, a float dtype for the returned values (default float64 if
            jax_enable_x64 is true, otherwise float32).
        minval: optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
        maxval: optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).
    Returns:
        A random array with the specified shape and dtype.
    """
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype argument to `uniform` must be a float dtype, "
                         f"got {dtype}")
    dtype = dtypes.canonicalize_dtype(dtype)
    shape = core.as_named_shape(shape)
    return _uniform(key, shape, dtype, minval, maxval)  # type: ignore


@partial(jit, static_argnums=(1, 2))
def _uniform(key, shape, dtype, minval, maxval) -> jnp.ndarray:
    _check_shape("uniform", shape)
    if not jnp.issubdtype(dtype, np.floating):
        raise TypeError("uniform only accepts floating point dtypes.")

    minval = lax.convert_element_type(minval, dtype)
    maxval = lax.convert_element_type(maxval, dtype)
    minval = lax.broadcast_to_rank(minval, shape.positional_rank)
    maxval = lax.broadcast_to_rank(maxval, shape.positional_rank)

    finfo = jnp.finfo(dtype)
    nbits, nmant = finfo.bits, finfo.nmant

    if nbits not in (16, 32, 64):
        raise TypeError("uniform only accepts 32- or 64-bit dtypes.")

    bits = _random_bits(key, nbits, shape)

    # The strategy here is to randomize only the mantissa bits with an exponent of
    # 1 (after applying the bias), then shift and scale to the desired range. The
    # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
    # equivalent float representations, which might not be true on all platforms.
    float_bits = lax.bitwise_or(
        lax.shift_right_logical(bits, np.array(nbits - nmant, lax.dtype(bits))),
        np.array(1., dtype).view(_UINT_DTYPES[nbits]))
    floats = lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
    return lax.max(minval, lax.reshape(floats * (maxval - minval) + minval, shape.positional))


@jit
def _fold_in(key, data):
    return threefry_2x32(key, PRNGKey(data))


@jit
def threefry_2x32(keypair, count):
    """Apply the Threefry 2x32 hash.
    Args:
        keypair: a pair of 32bit unsigned integers used for the key.
        count: an array of dtype uint32 used for the counts.
    Returns:
        An array of dtype uint32 with the same shape as `count`.
    """
    key1, key2 = keypair
    if not lax.dtype(key1) == lax.dtype(key2) == lax.dtype(count) == np.uint32:
        msg = "threefry_2x32 requires uint32 arguments, got {}"
        raise TypeError(msg.format([lax.dtype(x) for x in [key1, key2, count]]))

    odd_size = count.size % 2
    if odd_size:
        x = list(jnp.split(jnp.concatenate([count.ravel(), np.uint32([0])]), 2))
    else:
        x = list(jnp.split(count.ravel(), 2))

    x = threefry2x32_p.bind(key1, key2, x[0], x[1])
    out = jnp.concatenate(x)
    assert out.dtype == np.uint32
    return lax.reshape(out[:-1] if odd_size else out, count.shape)


@partial(jit, static_argnums=(1, 2))
def _random_bits(key, bit_width, shape):
    """Sample uniform random bits of given width and shape using PRNG key."""
    if not _is_prng_key(key):
        raise TypeError("_random_bits got invalid prng key.")
    if bit_width not in (8, 16, 32, 64):
        raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
    shape = core.as_named_shape(shape)
    for name, size in shape.named_items:
        real_size = lax.psum(1, name)
        if real_size != size:
            raise ValueError(f"The shape of axis {name} was specified as {size}, "
                             f"but it really is {real_size}")
        axis_index = lax.axis_index(name)
        key = fold_in(key, axis_index)
    size = prod(shape.positional)
    max_count = int(np.ceil(bit_width * size / 32))

    nblocks, rem = divmod(max_count, jnp.iinfo(np.uint32).max)
    if not nblocks:
        bits = threefry_2x32(key, lax.iota(np.uint32, rem))
    else:
        *subkeys, last_key = split(key, nblocks + 1)
        blocks = [threefry_2x32(k, lax.iota(np.uint32, jnp.iinfo(np.uint32).max))
                  for k in subkeys]
        last = threefry_2x32(last_key, lax.iota(np.uint32, rem))
        bits = lax.concatenate(blocks + [last], 0)

    dtype = _UINT_DTYPES[bit_width]
    if bit_width == 64:
        bits = [lax.convert_element_type(x, dtype) for x in jnp.split(bits, 2)]
        bits = lax.shift_left(bits[0], dtype(32)) | bits[1]
    elif bit_width in [8, 16]:
        # this is essentially bits.view(dtype)[:size]
        bits = lax.bitwise_and(
            np.uint32(np.iinfo(dtype).max),
            lax.shift_right_logical(
                lax.broadcast(bits, (1,)),
                lax.mul(
                    np.uint32(bit_width),
                    lax.broadcasted_iota(np.uint32, (32 // bit_width, 1), 0)
                )
            )
        )
        bits = lax.reshape(bits, (np.uint32(max_count * 32 // bit_width),), (1, 0))
        bits = lax.convert_element_type(bits, dtype)[:size]
    return lax.reshape(bits, shape)


def convert_element_type(operand: Array, new_dtype: DType) -> Array:
    """Elementwise cast.
    Wraps XLA's `ConvertElementType
    <https://www.tensorflow.org/xla/operation_semantics#convertelementtype>`_
    operator, which performs an elementwise conversion from one type to another.
    Similar to a C++ `static_cast`.
    Args:
        operand: an array or scalar value to be cast
        new_dtype: a NumPy dtype representing the target type.
    Returns:
        An array with the same shape as `operand`, cast elementwise to `new_dtype`.
    """
    if hasattr(operand, '__jax_array__'):
        operand = operand.__jax_array__()
    return _convert_element_type(operand, new_dtype, weak_type=False)


def _convert_element_type(operand: Array, new_dtype: Optional[DType] = None,
                          weak_type: bool = False):
    # Don't canonicalize old_dtype because x64 context might cause
    # un-canonicalized operands to be passed in.
    old_dtype = dtypes.dtype(operand, canonicalize=False)
    old_weak_type = dtypes.is_weakly_typed(operand)

    if new_dtype is None:
        new_dtype = old_dtype
    else:
        new_dtype = np.dtype(new_dtype)
    new_dtype = dtypes.dtype(new_dtype, canonicalize=True)
    new_weak_type = bool(weak_type)

    if (dtypes.issubdtype(old_dtype, np.complexfloating) and
            not dtypes.issubdtype(new_dtype, np.complexfloating)):
        msg = "Casting complex values to real discards the imaginary part"
        warnings.warn(msg, np.ComplexWarning, stacklevel=2)

    # Python has big integers, but convert_element_type(2 ** 100, np.float32) need
    # not be an error since the target dtype fits the value. Handle this case by
    # converting to a NumPy array before calling bind. Without this step, we'd
    # first canonicalize the input to a value of dtype int32 or int64, leading to
    # an overflow error.
    if type(operand) is int:
        operand = np.asarray(operand, new_dtype)
        old_weak_type = False

    if ((old_dtype, old_weak_type) == (new_dtype, new_weak_type)
            and isinstance(operand, (core.Tracer, device_array.DeviceArray))):
        return operand
    else:
        return convert_element_type_p.bind(operand, new_dtype=new_dtype, weak_type=new_weak_type)


def as_named_shape(shape) -> NamedShape:
    if isinstance(shape, NamedShape):
        return shape
    return NamedShape(*shape)


class NamedShape:
    def __init__(self, *args, **kwargs):
        self.__positional = canonicalize_shape(args)
        # TODO: Assert that kwargs match axis env?
        self.__named = dict(kwargs)


def canonicalize_shape(shape: Shape) -> Shape:
    """Canonicalizes and checks for errors in a user-provided shape value.
    Args:
        shape: a Python value that represents a shape.
    Returns:
        A tuple of integers.
    """
    try:
        return tuple(map(_canonicalize_dimension, shape))
    except TypeError:
        pass
    raise _invalid_shape_error(shape)


def _invalid_shape_error(shape: Shape):
    msg = ("Shapes must be 1D sequences of concrete values of integer type, "
           "got {}.")
    if any(isinstance(x, Tracer) and isinstance(get_aval(x), ShapedArray)
           and not isinstance(get_aval(x), ConcreteArray) for x in shape):
        msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
                "smaller subfunctions.")
    return TypeError(msg.format(shape))


bool_ = np.bool_
int_: np.dtype = np.int64  # type: ignore
float_: np.dtype = np.float64  # type: ignore
complex_ = np.complex128
# Default dtypes corresponding to Python scalars.
python_scalar_dtypes: dict = {
    bool: np.dtype(bool_),
    int: np.dtype(int_),
    float: np.dtype(float_),
    complex: np.dtype(complex_),
}
# bfloat16 support
bfloat16: type = xla_client.bfloat16
_bfloat16_dtype: np.dtype = np.dtype(bfloat16)
_jax_types = [
    np.dtype('bool'),
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
    np.dtype(bfloat16),
    np.dtype('float16'),
    np.dtype('float32'),
    np.dtype('float64'),
    np.dtype('complex64'),
    np.dtype('complex128'),
]
# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0 = np.dtype([('float0', np.void, 0)])
_jax_dtype_set = set(_jax_types) | {float0}


def dtype(x):
    if type(x) in python_scalar_dtypes:
        return python_scalar_dtypes[type(x)]
    dt = np.result_type(x)
    if dt not in _jax_dtype_set:
        raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                        "type. Only arrays of numeric types are supported by JAX.")
    return dt


def bernoulli(key: jnp.ndarray,
              p: RealArray = np.float32(0.5),
              shape: Optional[Union[Sequence[int], NamedShape]] = None) -> jnp.ndarray:
    """Sample Bernoulli random values with given shape and mean.
    Args:
        key: a PRNGKey used as the random key.
        p: optional, a float or array of floats for the mean of the random
            variables. Must be broadcast-compatible with ``shape``. Default 0.5.
        shape: optional, a tuple of nonnegative integers representing the result
            shape. Must be broadcast-compatible with ``p.shape``. The default (None)
            produces a result shape equal to ``p.shape``.
    Returns:
        A random array with boolean dtype and shape given by ``shape`` if ``shape``
        is not None, or else ``p.shape``.
    """
    dtype = canonicalize_dtype(lax.dtype(p))
    if shape is not None:
        shape = core.as_named_shape(shape)
    if not jnp.issubdtype(dtype, np.floating):
        msg = "bernoulli probability `p` must have a floating dtype, got {}."
        raise TypeError(msg.format(dtype))
    p = convert_element_type(p, dtype)
    return _bernoulli(key, p, shape)  # type: ignore


@partial(jit, static_argnums=(2,))
def _bernoulli(key, p, shape) -> jnp.ndarray:
    if shape is None:
        # TODO: Use the named part of `p` as well
        shape = np.shape(p)
    else:
        _check_shape("bernoulli", shape, np.shape(p))

    return uniform(key, shape, lax.dtype(p)) < p

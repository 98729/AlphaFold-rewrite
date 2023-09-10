from functools import partial
import numpy as np


def full_like(x: Array, fill_value: Array, dtype: Optional[DType] = None,
              shape: Optional[Shape] = None) -> Array:
    """Create a full array like np.full based on the example array `x`.
    Args:
        x: example array-like, used for shape and dtype information.
        fill_value: a scalar value to fill the entries of the output array.
        dtype: optional, a dtype parameter for the output ndarray.
        shape: optional, a shape parameter for the output ndarray.
    Returns:
        An ndarray with the same shape as `x` with its entries set equal to
        `fill_value`, similar to the output of np.full.
    """
    fill_shape = np.shape(x) if shape is None else canonicalize_shape(shape)
    weak_type = dtype is None and dtypes.is_weakly_typed(x)
    dtype = dtype or _dtype(x)
    return full(fill_shape, _convert_element_type(fill_value, dtype, weak_type))


_zero: Callable = partial(full_like, shape=(), fill_value=0)


def canonicalize_shape(shape: Shape, context: str = "") -> Shape:
    """Canonicalizes and checks for errors in a user-provided shape value.
    Args:
        shape: a Python value that represents a shape.
    Returns:
        A tuple of canonical dimension values.
    """
    try:
        return tuple(unsafe_map(_canonicalize_dimension, shape))
    except TypeError:
        pass
    raise _invalid_shape_error(shape, context)


def full(shape: Shape, fill_value: Array, dtype: Optional[DType] = None) -> Array:
    """Returns an array of `shape` filled with `fill_value`.
    Args:
        shape: sequence of integers, describing the shape of the output array.
        fill_value: the value to fill the new array with.
        dtype: the type of the output array, or `None`. If not `None`, `fill_value`
            will be cast to `dtype`.
    """
    shape = canonicalize_shape(shape)
    if np.shape(fill_value):
        msg = "full must be called with scalar fill_value, got fill_value.shape {}."
        raise TypeError(msg.format(np.shape(fill_value)))
    weak_type = dtype is None and dtypes.is_weakly_typed(fill_value)
    dtype = dtypes.canonicalize_dtype(dtype or _dtype(fill_value))
    fill_value = _convert_element_type(fill_value, dtype, weak_type)
    return broadcast(fill_value, shape)


def broadcast(operand: Array, sizes: Sequence[int]) -> Array:
    """Broadcasts an array, adding new leading dimensions
    Args:
        operand: an array
        sizes: a sequence of integers, giving the sizes of new leading dimensions
            to add to the front of the array.
    Returns:
        An array containing the result.
    See Also:
        jax.lax.broadcast_in_dim : add new dimensions at any location in the array shape.
    """
    dims = tuple(range(len(sizes), len(sizes) + np.ndim(operand)))
    return broadcast_in_dim(operand, tuple(sizes) + np.shape(operand), dims)


def broadcast_in_dim(operand: Array, shape: Shape,
                     broadcast_dimensions: Sequence[int]) -> Array:
    """Wraps XLA's `BroadcastInDim
    <https://www.tensorflow.org/xla/operation_semantics#broadcastindim>`_
    operator.
    Args:
        operand: an array
        shape: the shape of the target array
        broadcast_dimensions: to which dimension in the target shape each dimension
            of the operand shape corresponds to
    Returns:
        An array containing the result.
    See Also:
        jax.lax.broadcast : simpler interface to add new leading dimensions.
    """
    shape = _broadcast_in_dim_shape_rule(
        operand, shape=shape, broadcast_dimensions=broadcast_dimensions)
    if (np.ndim(operand) == len(shape) and not len(broadcast_dimensions)
            and isinstance(operand, (device_array.DeviceArray, core.Tracer))):
        return operand
    if config.jax_dynamic_shapes:
        # We must gate this behavior under a flag because otherwise the errors
        # raised are different (and have worse source provenance information).
        dyn_shape = [d for d in shape if isinstance(d, core.Tracer)]
        shape_ = [d if not isinstance(d, core.Tracer) else None for d in shape]
    else:
        dyn_shape = []
        shape_ = shape  # type: ignore
    return broadcast_in_dim_p.bind(
        operand, *dyn_shape, shape=tuple(shape_),
        broadcast_dimensions=tuple(broadcast_dimensions))


def canonicalize_dim(d: DimSize, context: str = "") -> DimSize:
    """Canonicalizes and checks for errors in a user-provided shape dimension value.
    Args:
        f: a Python value that represents a dimension.
    Returns:
        A canonical dimension value.
    """
    return canonicalize_shape((d,), context)[0]


def _dynamic_slice_indices(operand, start_indices: Any):
    # Normalize the start_indices w.r.t. operand.shape
    if len(start_indices) != operand.ndim:
        msg = ("Length of slice indices must match number of operand dimensions ({} "
               "vs {})")
        raise ValueError(msg.format(len(start_indices), operand.shape))
    if not isinstance(start_indices, (tuple, list)):
        if start_indices.ndim != 1:
            raise ValueError("Slice indices must be a 1D sequence, got {}"
                             .format(start_indices.shape))
        start_indices = [i for i in start_indices]
    return [np.asarray(i + d if i < 0 else i, lax._dtype(i))
            if isinstance(i, (int, np.integer)) and core.is_constant_dim(d)
            else lax.select(
        lax.lt(i, lax._const(i, 0)),
        lax.add(i, lax.convert_element_type(core.dimension_as_value(d), lax._dtype(i))),
        i)
            for i, d in zip(start_indices, operand.shape)]


def dynamic_slice(operand: Array, start_indices: Sequence[Array],
                  slice_sizes: Shape) -> Array:
    """Wraps XLA's `DynamicSlice
    <https://www.tensorflow.org/xla/operation_semantics#dynamicslice>`_
    operator.
    Args:
        operand: an array to slice.
        start_indices: a list of scalar indices, one per dimension. These values
            may be dynamic.
        slice_sizes: the size of the slice. Must be a sequence of non-negative
            integers with length equal to `ndim(operand)`. Inside a JIT compiled
            function, only static values are supported (all JAX arrays inside JIT
            must have statically known size).
    Returns:
        An array containing the slice.
    Examples:
        Here is a simple two-dimensional dynamic slice:
        >>> x = jnp.arange(12).reshape(3, 4)
        >>> x
    DeviceArray([[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)
    >>> dynamic_slice(x, (1, 1), (2, 3))
    DeviceArray([[ 5,  6,  7],
                 [ 9, 10, 11]], dtype=int32)
    Note the potentially surprising behavior for the case where the requested slice
    overruns the bounds of the array; in this case the start index is adjusted to
    return a slice of the requested size:
    >>> dynamic_slice(x, (1, 1), (2, 4))
    DeviceArray([[ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)
    """
    start_indices = _dynamic_slice_indices(operand, start_indices)
    return dynamic_slice_p.bind(operand, *start_indices,
                                slice_sizes=core.canonicalize_shape(slice_sizes))


def dynamic_slice_in_dim(operand: Array, start_index: Array,
                         slice_size: int, axis: int = 0) -> Array:
    """Convenience wrapper around dynamic_slice applying to one dimension."""
    start_indices = [lax._zero(start_index)] * operand.ndim
    slice_sizes = list(operand.shape)
    axis = int(axis)
    start_indices[axis] = start_index
    slice_sizes[axis] = core._canonicalize_dimension(slice_size)
    return dynamic_slice(operand, start_indices, slice_sizes)

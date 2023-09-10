from typing import Any, Optional, Tuple, Union, Hashable, Callable, Dict, List, Type
import numpy as np
import ShapedArray
import operator


Array = Any
AxisName = Hashable


def canonicalize_axis(axis, num_dims) -> int:
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
    axis = operator.index(axis)
    if not -num_dims <= axis < num_dims:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {num_dims}")
    if axis < 0:
        axis = axis + num_dims
    return axis


def _one_hot(x: Array, num_classes: int, *,
             dtype: Any, axis: Union[int, AxisName]) -> Array:
    num_classes = concrete_or_error(
        int, num_classes,
        "The error arose in jax.nn.one_hot argument `num_classes`.")
    dtype = canonicalize_dtype(dtype)
    x = np.asarray(x)
    try:
        output_pos_axis = canonicalize_axis(axis, x.ndim + 1)
    except TypeError:
        # axis_size = lax.psum(1, axis)
        axis_size = np.sum(1, axis)
        if num_classes != axis_size:
            raise ValueError(f"Expected num_classes to match the size of axis {axis}, "
                             f"but {num_classes} != {axis_size}") from None
        axis_idx = lax.axis_index(axis)
        return np.asarray(x == axis_idx, dtype=dtype)
    axis = operator.index(axis)  # type: ignore[arg-type]
    lhs = np.expand_dims(x, (axis,))
    rhs_shape = [1] * x.ndim
    rhs_shape.insert(output_pos_axis, num_classes)
    rhs = lax.broadcasted_iota(x.dtype, rhs_shape, output_pos_axis)
    return np.asarray(lhs == rhs, dtype=dtype)


def one_hot(x: Array, num_classes: int, *,
            dtype: Any = np.float_, axis: Union[int, AxisName] = -1) -> Array:
    """One-hot encodes the given indicies.
    Each index in the input ``x`` is encoded as a vector of zeros of length
    ``num_classes`` with the element at ``index`` set to one::
        >>> jax.nn.one_hot(jnp.array([0, 1, 2]), 3)
        DeviceArray([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]], dtype=float32)
    Indicies outside the range [0, num_classes) will be encoded as zeros::
        >>> jax.nn.one_hot(jnp.array([-1, 3]), 3)
        DeviceArray([[0., 0., 0.],
                     [0., 0., 0.]], dtype=float32)
    Args:
        x: A tensor of indices.
        num_classes: Number of classes in the one-hot dimension.
        dtype: optional, a float dtype for the returned values (default :obj:`jnp.float_`).
        axis: the axis or axes along which the function should be
            computed.
    """
    num_classes = concrete_or_error(
        int, num_classes,
        "The error arose in jax.nn.one_hot argument `num_classes`.")
    return _one_hot(x, num_classes, dtype=dtype, axis=axis)


def concrete_or_error(force: Any, val: Any, context=""):
    if force is None:
        force = lambda x: x
    if isinstance(val, Tracer):
        if isinstance(val.aval, ConcreteArray):
            return force(val.aval.val)
        else:
            return print(context)
    else:
        return force(val)


class Tracer:
    @property
    def aval(self):
        raise NotImplementedError("must override")

    def _assert_live(self) -> None:
        pass  # Override for liveness checking


_weak_types = [int, float, complex]


def is_weakly_typed(x):
    try:
        return x.aval.weak_type
    except AttributeError:
        return type(x) in _weak_types


_dtype_to_32bit_dtype = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}


def _canonicalize_dtype(x64_enabled, dtype):
    """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
    try:
        dtype = np.dtype(dtype)
    except TypeError as e:
        raise TypeError(f'd type {dtype!r} not understood') from e

    if x64_enabled:
        return dtype
    else:
        return _dtype_to_32bit_dtype.get(dtype, dtype)


# default Config.jax_enable_x64 = False
jax_enable_x64 = False
x64_enabled = jax_enable_x64


def canonicalize_dtype(dtype):
    return _canonicalize_dtype(x64_enabled, dtype)


class ConcreteArray(ShapedArray):
    __slots__ = ['val']
    array_abstraction_level = 0

    def __init__(self, dtype, val, weak_type=None):
        super().__init__(
            np.shape(val), dtype,
            weak_type=is_weakly_typed(val) if weak_type is None else weak_type)
        # Note: canonicalized self.dtype doesn't necessarily match self.val
        assert self.dtype == canonicalize_dtype(np.result_type(val)), (val, dtype)
        self.val = val
        assert self.dtype != np.dtype('O'), val

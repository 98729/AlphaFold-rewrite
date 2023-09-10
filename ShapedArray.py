import UnshapedArray
from typing import (Any, Sequence, Union, Dict)
import numpy as np
from one_hot import Tracer
import operator

# Shapes are tuples of dimension sizes, which are normally integers. We allow
# modules to extend the set of dimension sizes to contain other types, e.g.,
# symbolic dimensions in jax2tf.shape_poly.DimVar and masking.Poly.
DimSize = Union[int, Any]  # extensible
Shape = Sequence[DimSize]


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))


map, unsafe_map = safe_map, map


class DimensionHandler:
    """Operations on dimension sizes.
    Dimension sizes are normally integer constants, but can also be symbolic,
    e.g., masking.Poly or jax2tf.shape_poly.DimVar.
    The base class works for integers only. Subclasses are invoked when at least
    one of the operands has a type registered in _SPECIAL_DIMENSION_HANDLERS. In
    that case, all operands are guaranteed to be either the special dimension
    type, or Python integer scalars.
    Subclasses should raise InconclusiveDimensionOperation if the result cannot
    be computed in some contexts.
    """

    def is_constant(self, d: DimSize) -> bool:
        """The dimension is a constant."""
        return True

    def symbolic_equal(self, d1: DimSize, d2: DimSize) -> bool:
        """True iff the dimension sizes are equal in all contexts; False otherwise.
        Unlike `d1 == d2` this never raises InconclusiveDimensionOperation.
        """
        return d1 == d2

    def greater_equal(self, d1: DimSize, d2: DimSize) -> bool:
        """Computes `d1 >= d2`.
        Raise InconclusiveDimensionOperation if the result is different in
        different contexts.
        """
        return d1 >= d2

    def sum(self, *ds: DimSize) -> DimSize:
        """Sum of dimensions.
        Raises InconclusiveDimensionOperation if the result cannot be represented
        by the same DimSize in all contexts.
        """
        return sum(ds)

    def diff(self, d1: DimSize, d2: DimSize) -> DimSize:
        """Difference of dimensions.
        Raises InconclusiveDimensionOperation if the result cannot be represented
        by the same DimSize in all contexts.
        """
        return d1 - d2

    def divide_shape_sizes(self, s1: Shape, s2: Shape) -> DimSize:
        """Computes integer "i" such that i  * size(s2) == size(s1).
        Raise InconclusiveDimensionOperation if there is no such integer for all
        contexts,
        """
        sz1 = int(np.prod(s1))
        sz2 = int(np.prod(s2))
        if sz1 == 0 and sz2 == 0:
            return 1
        if sz1 % sz2:
            print(f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}")
        return sz1 // sz2

    def stride(self, d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
        """(d - window_size) // window_stride + 1.
        If d == 0 or window_size > d, returns 0.
        """
        if d == 0 or window_size > d: return 0
        return (d - window_size) // window_stride + 1

    def dilate(self, d: DimSize, dilation: int) -> DimSize:
        """Implements `0 if d == 0 else 1 + dilation * (d - 1))`"""
        return 0 if d == 0 else 1 + dilation * (d - 1)

    def as_value(self, d: DimSize):
        """Turns a dimension size into a JAX value that we can compute with."""
        return d


_dimension_handler_int = DimensionHandler()
_SPECIAL_DIMENSION_HANDLERS: Dict[type, DimensionHandler] = {}


class ShapedArray(UnshapedArray):
    __slots__ = ['shape', 'named_shape']
    array_abstraction_level = 1

    def __init__(self, shape, dtype, weak_type=False, named_shape=None):
        super().__init__(dtype, weak_type=weak_type)
        self.shape = canonicalize_shape(shape)
        self.named_shape = {} if named_shape is None else dict(named_shape)


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
    msg = ("Shapes must be 1D sequences of concrete values of integer type, "
           f"got {shape}.")
    if context:
        msg += f" {context}."
    print(msg)


# default config.jax_dynamic_shapes
jax_dynamic_shapes = False


def _canonicalize_dimension(dim: DimSize) -> DimSize:
    if (type(dim) in _SPECIAL_DIMENSION_HANDLERS or
            isinstance(dim, Tracer) and jax_dynamic_shapes):
        return dim
    else:
        return operator.index(dim)

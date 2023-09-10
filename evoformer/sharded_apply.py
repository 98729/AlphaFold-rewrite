import numpy as np


def eval_shape(fun: Callable, *args, **kwargs):
    """Compute the shape/dtype of ``fun`` without any FLOPs.
    This utility function is useful for performing shape inference. Its
    input/output behavior is defined by::
        def eval_shape(fun, *args, **kwargs):
            out = fun(*args, **kwargs)
            return jax.tree_util.tree_map(shape_dtype_struct, out)
        def shape_dtype_struct(x):
            return ShapeDtypeStruct(x.shape, x.dtype)
        class ShapeDtypeStruct:
            __slots__ = ["shape", "dtype"]
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
    In particular, the output is a pytree of objects that have ``shape`` and
    ``dtype`` attributes, but nothing else about them is guaranteed by the API.
    But instead of applying ``fun`` directly, which might be expensive, it uses
    JAX's abstract interpretation machinery to evaluate the shapes without doing
    any FLOPs.
    Using :py:func:`eval_shape` can also catch shape errors, and will raise same
    shape errors as evaluating ``fun(*args, **kwargs)``.
    Args:
        fun: The function whose output shape should be evaluated.
        *args: a positional argument tuple of arrays, scalars, or (nested) standard
            Python containers (tuples, lists, dicts, namedtuples, i.e. pytrees) of
            those types. Since only the ``shape`` and ``dtype`` attributes are
            accessed, only values that duck-type arrays are required, rather than real
            ndarrays. The duck-typed objects cannot be namedtuples because those are
            treated as standard Python containers. See the example below.
        **kwargs: a keyword argument dict of arrays, scalars, or (nested) standard
            Python containers (pytrees) of those types. As in ``args``, array values
            need only be duck-typed to have ``shape`` and ``dtype`` attributes.
    For example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> f = lambda A, x: jnp.tanh(jnp.dot(A, x))
    >>> class MyArgArray(object):
    ...   def __init__(self, shape, dtype):
    ...     self.shape = shape
    ...     self.dtype = jnp.dtype(dtype)
    ...
    >>> A = MyArgArray((2000, 3000), jnp.float32)
    >>> x = MyArgArray((3000, 1000), jnp.float32)
     >>> out = jax.eval_shape(f, A, x)  # no FLOPs performed
    >>> print(out.shape)
    (2000, 1000)
    >>> print(out.dtype)
    float32
     """
    args_flat, in_tree = tree_flatten((args, kwargs))
    wrapped_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    debug_info = pe.debug_info(fun, in_tree, True, "eval_shape")
    out = pe.abstract_eval_fun(wrapped_fun.call_wrapped,
                               *map(shaped_abstractify, args_flat),
                               debug_info=debug_info)
    out = [ShapeDtypeStruct(x.shape, x.dtype, x.named_shape) for x in out]
    return tree_unflatten(out_tree(), out)


def flatten_axes(name, treedef, axis_tree, *, kws=False, tupled_args=False):
    # given an axis spec tree axis_tree (a pytree with integers and Nones at the
    # leaves, i.e. the Nones are to be considered leaves) that is a tree prefix of
    # the given treedef, build a complete axis spec tree with the same structure
    # and return the flattened result
    # TODO(mattjj,phawkins): improve this implementation

    proxy = object()
    dummy = tree_unflatten(treedef, [object()] * treedef.num_leaves)
    axes = []
    add_leaves = lambda i, x: axes.extend([i] * len(tree_flatten(x)[0]))
    try:
        tree_multimap(add_leaves, _replace_nones(proxy, axis_tree), dummy)
    except ValueError:
        if kws:
            # if keyword arguments are included in the tree, we make adapt the error
            # message only to be about the positional arguments
            treedef, leaf = treedef_children(treedef)
            assert treedef_is_leaf(leaf)
            axis_tree, _ = axis_tree
        hint = ""
        if tupled_args:
            hint += (f" Note that {name} that are non-trivial pytrees should always be "
                     f"wrapped in a tuple representing the argument list.")
            if len(treedef.children()) == 1:
                try:
                    flatten_axes(name, treedef, (axis_tree,))
                except ValueError:
                    pass  # That's not the issue.
                else:
                    hint += (f" In particular, you're passing in a single argument which "
                             f"means that {name} might need to be wrapped in "
                             f"a singleton tuple.")
        raise ValueError(f"{name} specification must be a tree prefix of the "
                         f"corresponding value, got specification {axis_tree} "
                         f"for value tree {treedef}.{hint}") from None
    axes = [None if a is proxy else a for a in axes]
    assert len(axes) == treedef.num_leaves
    return axes


def tree_unflatten(treedef, leaves):
    """Reconstructs a pytree from the treedef and the leaves.
    The inverse of :func:`tree_flatten`.
    Args:
        treedef: the treedef to reconstruct
        leaves: the list of leaves to use for reconstruction. The list must match
            the leaves of the treedef.
    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure
        described by ``treedef``.
    """
    return treedef.unflatten(leaves)


def tree_flatten(tree, is_leaf: Optional[Callable[[Any], bool]] = None):
    """Flattens a pytree.
    Args:
        tree: a pytree to flatten.
            is_leaf: an optionally specified function that will be called at each
            flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
    Returns:
        A pair where the first element is a list of leaf values and the second
        element is a treedef representing the structure of the flattened tree.
    """
    return pytree.flatten(tree, is_leaf)


def tree_map(f: Callable[..., Any], tree: Any, *rest: Any,
             is_leaf: Optional[Callable[[Any], bool]] = None) -> Any:
    """Maps a multi-input function over pytree args to produce a new pytree.
    Args:
        f: function that takes ``1 + len(rest)`` arguments, to be applied at the
          corresponding leaves of the pytrees.
        tree: a pytree to be mapped over, with each leaf providing the first
          positional argument to ``f``.
        *rest: a tuple of pytrees, each of which has the same structure as tree or
          or has tree as a prefix.
        is_leaf: an optionally specified function that will be called at each
          flattening step. It should return a boolean, which indicates whether
          the flattening should traverse the current object, or if it should be
          stopped immediately, with the whole subtree being treated as a leaf.
    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each
        leaf given by ``f(x, *xs)`` where ``x`` is the value at the corresponding
        leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in
        ``rest``.
    """
    leaves, treedef = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


tree_multimap = tree_map


def _expand_axes(axes, values, name='sharded_apply'):
    values_tree_def = jax.tree_flatten(values)[1]
    flat_axes = jax.api_util.flatten_axes(name, values_tree_def, axes)
    # Replace None's with PROXY
    flat_axes = [PROXY if x is None else x for x in flat_axes]
    return jax.tree_unflatten(values_tree_def, flat_axes)


def inference_subbatch(
        module,
        subbatch_size: int,
        batched_args,
        nonbatched_args,
        low_memory: bool = True,
        input_subbatch_dim: int = 0,
        output_subbatch_dim=None):
    """Run through subbatches (like batch apply but with split and concat)."""
    assert len(batched_args) > 0  # pylint: disable=g-explicit-length-test

    if not low_memory:
        args = list(batched_args) + list(nonbatched_args)
        return module(*args)

    if output_subbatch_dim is None:
        output_subbatch_dim = input_subbatch_dim

    def run_module(*batched_args):
        args = list(batched_args) + list(nonbatched_args)
        return module(*args)

    sharded_module = sharded_apply(run_module,  # fun
                                   shard_size=subbatch_size,
                                   in_axes=input_subbatch_dim,
                                   out_axes=output_subbatch_dim)
    return sharded_module(*batched_args)


def sharded_apply(
        fun,  # pylint: disable=g-bare-generic
        shard_size=1,
        in_axes=0,
        out_axes=0,
        new_out_axes: bool = False):
    """Sharded apply.
    Applies `fun` over shards to axes, in a way similar to vmap,
    but does so in shards of `shard_size`. Shards are stacked after.
    This allows a smooth trade-off between
    memory usage (as in a plain map) vs higher throughput (as in a vmap).
    Args:
        fun: Function to apply smap transform to.
        shard_size: Integer denoting shard size.
        in_axes: Either integer or pytree describing which axis to map over for each
            input to `fun`, None denotes broadcasting.
        out_axes: integer or pytree denoting to what axis in the output the mapped
            over axis maps.
        new_out_axes: whether to stack outputs on new axes. This assumes that the
            output sizes for each shard (including the possible remainder shard) are
            the same.
    Returns:
        function with smap applied.
    """
    docstr = ('Mapped version of {fun}. Takes similar arguments to {fun} '
              'but with additional array axes over which {fun} is mapped.')
    if new_out_axes:
        raise NotImplementedError('New output axes not yet implemented.')

    # shard size None denotes no sharding
    if shard_size is None:
        return fun

    def mapped_fn(*args):
        # Expand in axes and Determine Loop range
        in_axes_ = _expand_axes(in_axes, args)

        in_sizes = jax.tree_multimap(_maybe_get_size, args, in_axes_)
        flat_sizes = jax.tree_flatten(in_sizes)[0]
        in_size = max(flat_sizes)
        assert all(i in {in_size, -1} for i in flat_sizes)

        num_extra_shards = (in_size - 1) // shard_size

        # Fix Up if necessary
        last_shard_size = in_size % shard_size
        last_shard_size = shard_size if last_shard_size == 0 else last_shard_size

        def apply_fun_to_slice(slice_start, slice_size):
            input_slice = jax.tree_multimap(
                lambda array, axis: _maybe_slice(array, slice_start, slice_size, axis
                                                 ), args, in_axes_)
            return fun(*input_slice)

        remainder_shape_dtype = hk.eval_shape(partial(apply_fun_to_slice, 0, last_shard_size))
        out_dtypes = jax.tree_map(lambda x: x.dtype, remainder_shape_dtype)
        out_shapes = jax.tree_map(lambda x: x.shape, remainder_shape_dtype)
        out_axes_ = _expand_axes(out_axes, remainder_shape_dtype)

        if num_extra_shards > 0:
            regular_shard_shape_dtype = hk.eval_shape(
                partial(apply_fun_to_slice, 0, shard_size))
            shard_shapes = jax.tree_map(lambda x: x.shape, regular_shard_shape_dtype)

            def make_output_shape(axis, shard_shape, remainder_shape):
                return shard_shape[:axis] + (
                    shard_shape[axis] * num_extra_shards +
                    remainder_shape[axis],) + shard_shape[axis + 1:]

            out_shapes = jax.tree_multimap(make_output_shape, out_axes_, shard_shapes,
                                           out_shapes)

        # Calls dynamic Update slice with different argument order
        # This is here since tree_multimap only works with positional arguments
        def dynamic_update_slice_in_dim(full_array, update, axis, i):
            return jax.lax.dynamic_update_slice_in_dim(full_array, update, i, axis)

        def compute_shard(outputs, slice_start, slice_size):
            slice_out = apply_fun_to_slice(slice_start, slice_size)
            update_slice = partial(dynamic_update_slice_in_dim, i=slice_start)
            return jax.tree_multimap(update_slice, outputs, slice_out, out_axes_)

        def scan_iteration(outputs, i):
            new_outputs = compute_shard(outputs, i, shard_size)
            return new_outputs, ()

        slice_starts = np.arange(0, in_size - shard_size + 1, shard_size)

        def allocate_buffer(dtype, shape):
            return np.zeros(shape, dtype=dtype)

        outputs = jax.tree_multimap(allocate_buffer, out_dtypes, out_shapes)

        if slice_starts.shape[0] > 0:
            outputs, _ = hk.scan(scan_iteration, outputs, slice_starts)

        if last_shard_size != shard_size:
            remainder_start = in_size - last_shard_size
            outputs = compute_shard(outputs, remainder_start, last_shard_size)

        return outputs

    return mapped_fn

def dropout(rng: PRNGKey, rate: float, x: jnp.ndarray) -> jnp.ndarray:
    """Randomly drop units in the input at a given rate.
   See: http://www.cs.toronto.edu/~hinton/absps/dropout.pdf
   Args:
        rng: A JAX random key.
        rate: Probability that each element of ``x`` is discarded. Must be a scalar
            in the range ``[0, 1)``.
        x: The value to be dropped out.
    Returns:
        x, but dropped out and scaled by ``1 / (1 - rate)``.
    Note:
        This involves generating `x.size` pseudo-random samples from U([0, 1))
        computed with the full precision required to compare them with `rate`. When
        `rate` is a Python float, this is typically 32 bits, which is often more
        than what applications require. A work-around is to pass `rate` with a lower
        precision, e.g. using `np.float16(rate)`.
    """
    if rate < 0 or rate >= 1:
        raise ValueError("rate must be in [0, 1).")
    if rate == 0.0:
        return x
    keep_rate = 1.0 - rate
    shape = list(x.shape)
    keep = np.random.binomial(n=1, p=keep_rate, size=tuple(shape))
    return keep * x / keep_rate


def safe_dropout(*, tensor, safe_key, rate, is_deterministic, is_training):
    if is_training and rate != 0.0 and not is_deterministic:
        return dropout(safe_key.get(), rate, tensor)
    else:
        return tensor


# class SafeKey:
#     """Safety wrapper for PRNG keys."""
#
#     def __init__(self, key):
#         self._key = key
#         self._used = False
#
#     def _assert_not_used(self):
#         if self._used:
#             raise RuntimeError('Random key has been used previously.')
#
#     def get(self):
#         self._assert_not_used()
#         self._used = True
#         return self._key
#
#     def split(self, num_keys=2):
#         self._assert_not_used()
#         self._used = True
#         new_keys = jax.random.split(self._key, num_keys)
#         return jax.tree_map(SafeKey, tuple(new_keys))
#
#     def duplicate(self, num_keys=2):
#         self._assert_not_used()
#         self._used = True
#         return tuple(SafeKey(self._key) for _ in range(num_keys))
#
#
# def _safe_key_flatten(safe_key):
#     # Flatten transfers "ownership" to the tree
#     return (safe_key._key,), safe_key._used  # pylint: disable=protected-access
#
#
# def _safe_key_unflatten(aux_data, children):
#     ret = SafeKey(children[0])
#     ret._used = aux_data  # pylint: disable=protected-access
#     return ret
#
#
# jax.tree_util.register_pytree_node(SafeKey, _safe_key_flatten, _safe_key_unflatten)

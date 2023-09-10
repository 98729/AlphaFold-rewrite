import numpy as np

QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]  # kr

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]]

QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, -1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0],
                          [0, 0, 0, -1],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, -1, 0, 0],
                          [1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]


def quat_to_rot(normalized_quat):
    """Convert a normalized quaternion to a rotation matrix."""
    rot_tensor = np.sum(
        np.reshape(QUAT_TO_ROT, (4, 4, 9)) *
        normalized_quat[..., :, None, None] *
        normalized_quat[..., None, :, None],
        axis=(-3, -2))
    rot = np.moveaxis(rot_tensor, -1, 0)  # Unstack.
    return [[rot[0], rot[1], rot[2]],
            [rot[3], rot[4], rot[5]],
            [rot[6], rot[7], rot[8]]]


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    return np.sum(
        QUAT_MULTIPLY_BY_VEC *
        quat[..., :, None, None] *
        vec[..., None, :, None],
        axis=(-3, -2))


def apply_rot_to_vec(rot, vec, unstack=False):
    """Multiply rotation matrix by a vector."""
    if unstack:
        x, y, z = [vec[:, i] for i in range(3)]
    else:
        x, y, z = vec
    return [rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
            rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
            rot[2][0] * x + rot[2][1] * y + rot[2][2] * z]


def apply_inverse_rot_to_vec(rot, vec):
    """Multiply the inverse of a rotation matrix by a vector."""
    # Inverse rotation is just transpose
    return [rot[0][0] * vec[0] + rot[1][0] * vec[1] + rot[2][0] * vec[2],
            rot[0][1] * vec[0] + rot[1][1] * vec[1] + rot[2][1] * vec[2],
            rot[0][2] * vec[0] + rot[1][2] * vec[1] + rot[2][2] * vec[2]]


class QuatAffine(object):
    """Affine transformation represented by quaternion and vector."""

    def __init__(self, quaternion, translation, rotation=None, normalize=True, unstack_inputs=False):
        """Initialize from quaternion and translation.
        Args:
            quaternion: Rotation represented by a quaternion, to be applied
                before translation.  Must be a unit quaternion unless normalize==True.
            translation: Translation represented as a vector.
            rotation: Same rotation as the quaternion, represented as a (..., 3, 3)
                tensor.  If None, rotation will be calculated from the quaternion.
            normalize: If True, l2 normalize the quaternion on input.
            unstack_inputs: If True, translation is a vector with last component 3
        """

        if quaternion is not None:
            assert quaternion.shape[-1] == 4

        if unstack_inputs:
            if rotation is not None:
                rotation = [np.moveaxis(x, -1, 0)  # Unstack.
                            for x in np.moveaxis(rotation, -2, 0)]  # Unstack.
            translation = np.moveaxis(translation, -1, 0)  # Unstack.

        if normalize and quaternion is not None:
            quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).astype(np.float32)

        if rotation is None:
            rotation = quat_to_rot(quaternion)

        self.quaternion = quaternion
        self.rotation = [list(row) for row in rotation]
        self.translation = list(translation)

        assert all(len(row) == 3 for row in self.rotation)
        assert len(self.translation) == 3

    def to_tensor(self):
        return np.concatenate(
            [self.quaternion] +
            [np.expand_dims(x, axis=-1) for x in self.translation],
            axis=-1)

    def apply_tensor_fn(self, tensor_fn):
        """Return a new QuatAffine with tensor_fn applied (e.g. stop_gradient)."""
        return QuatAffine(
            tensor_fn(self.quaternion),
            [tensor_fn(x) for x in self.translation],
            rotation=[[tensor_fn(x) for x in row] for row in self.rotation],
            normalize=False)

    def apply_rotation_tensor_fn(self, tensor_fn):
        """Return a new QuatAffine with tensor_fn applied to the rotation part."""
        return QuatAffine(
            tensor_fn(self.quaternion),
            [x for x in self.translation],
            rotation=[[tensor_fn(x) for x in row] for row in self.rotation],
            normalize=False)

    def scale_translation(self, position_scale):
        """Return a new quat affine with a different scale for translation."""

        return QuatAffine(
            self.quaternion,
            [x * position_scale for x in self.translation],
            rotation=[[x for x in row] for row in self.rotation],
            normalize=False)

    @classmethod
    def from_tensor(cls, tensor, normalize=False):
        quaternion, tx, ty, tz = np.split(tensor, [4, 5, 6], axis=-1)
        return cls(quaternion,
                   [tx[..., 0], ty[..., 0], tz[..., 0]],
                   normalize=normalize)

    def pre_compose(self, update):
        """Return a new QuatAffine which applies the transformation update first.
        Args:
            update: Length-6 vector. 3-vector of x, y, and z such that the quaternion
                update is (1, x, y, z) and zero for the 3-vector is the identity
                quaternion. 3-vector for translation concatenated.
        Returns:
            New QuatAffine object.
        """
        vector_quaternion_update, x, y, z = np.split(update, [3, 4, 5], axis=-1)
        trans_update = [np.squeeze(x, axis=-1),
                        np.squeeze(y, axis=-1),
                        np.squeeze(z, axis=-1)]

        new_quaternion = (self.quaternion +
                          quat_multiply_by_vec(self.quaternion,
                                               vector_quaternion_update))

        trans_update = apply_rot_to_vec(self.rotation, trans_update)
        new_translation = [
            self.translation[0] + trans_update[0],
            self.translation[1] + trans_update[1],
            self.translation[2] + trans_update[2]]

        return QuatAffine(new_quaternion, new_translation)

    def apply_to_point(self, point, extra_dims=0):
        """Apply affine to a point.
        Args:
            point: List of 3 tensors to apply affine.
            extra_dims:  Number of dimensions at the end of the transformed_point
                shape that are not present in the rotation and translation.  The most
                common use is rotation N points at once with extra_dims=1 for use in a
                network.
        Returns:
            Transformed point after applying affine.
        """
        rotation = self.rotation
        translation = self.translation
        # for _ in range(extra_dims):
        #     expand_fn = functools.partial(jnp.expand_dims, axis=-1)
        #     rotation = jax.tree_map(expand_fn, rotation)
        #     translation = jax.tree_map(expand_fn, translation)

        rot_point = apply_rot_to_vec(rotation, point)
        return [
            rot_point[0] + translation[0],
            rot_point[1] + translation[1],
            rot_point[2] + translation[2]]

    def invert_point(self, transformed_point, extra_dims=0):
        """Apply inverse of transformation to a point.
        Args:
            transformed_point: List of 3 tensors to apply affine
            extra_dims:  Number of dimensions at the end of the transformed_point
                shape that are not present in the rotation and translation.  The most
                common use is rotation N points at once with extra_dims=1 for use in a
                network.
        Returns:
            Transformed point after applying affine.
        """
        rotation = self.rotation
        translation = self.translation
        # for _ in range(extra_dims):
        #     expand_fn = functools.partial(jnp.expand_dims, axis=-1)
        #     rotation = jax.tree_map(expand_fn, rotation)
        #     translation = jax.tree_map(expand_fn, translation)

        rot_point = [
            transformed_point[0] - translation[0],
            transformed_point[1] - translation[1],
            transformed_point[2] - translation[2]]

        return apply_inverse_rot_to_vec(rotation, rot_point)

    def __repr__(self):
        return 'QuatAffine(%r, %r)' % (self.quaternion, self.translation)

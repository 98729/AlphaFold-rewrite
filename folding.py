import numpy as np
import common_modules
from distribution import LayerNorm, Constant
import quat_affine
import r3
import all_atom
from typing import Dict
import residue_constants
import utils


def one_hot(x, num_classes, axis=-1, dtype=np.float_):
    indices = x.flatten()
    arr = np.zeros(shape=(len(indices) * num_classes,))
    for i in range(len(indices)):
        if 0 <= indices[i] < num_classes:
            arr[num_classes * i + indices[i]] = 1.
    size = x.shape + (num_classes,)
    return arr.reshape(size)


def squared_difference(x, y):
    return np.square(x - y)


def softmax(x: int or float or np.ndarray):
    x_max = np.amax(x, axis=-1, keepdims=True)
    unnormalized = np.exp(x - x_max)

    return unnormalized / np.sum(unnormalized, axis=-1, keepdims=True)


def softplus(x: int or float or np.ndarray):
    return np.logaddexp(x, 0).astype(np.float32)


def relu(x: int or float or np.ndarray):
    return np.maximum(0, x).astype(np.float32)


def dropout(safe_key, rate: float, x: np.ndarray) -> np.ndarray:
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
    np.random.seed(safe_key)
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
        return dropout(safe_key, rate, tensor)
    else:
        return tensor


class InvariantPointAttention:
    """Invariant Point attention module.
    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).
    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.
    Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
    """

    def __init__(self, zero_init=True, dist_epsilon=1e-8, num_head=12, num_scalar_qk=16,
                 num_point_qk=4, num_scalar_v=16, num_point_v=8, num_output=384,
                 name='invariant_point_attention'):
        """Initialize.
        Args:
            config: Structure Module Config
            global_config: Global Config of Model.
            dist_epsilon: Small value to avoid NaN in distance calculation.
            name: Haiku Module name.
        """
        super().__init__(name=name)

        self._dist_epsilon = dist_epsilon
        self.zero_init = zero_init
        self._zero_initialize_last = zero_init
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_point_qk = num_point_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_v = num_point_v
        self.num_output = num_output

    def __call__(self, inputs_1d, inputs_2d, mask, affine):
        """Compute geometry-aware attention.
        Given a set of query residues (defined by affines and associated scalar
        features), this function computes geometry-aware attention between the
        query residues and target residues.
        The residues produce points in their local reference frame, which
        are converted into the global frame in order to compute attention via
        euclidean distance.
        Equivalently, the target residues produce points in their local frame to be
        used as attention values, which are converted into the query residues'
        local frames.
        Args:
            inputs_1d: (N, C) 1D input embedding that is the basis for the
                scalar queries.
            inputs_2d: (N, M, C') 2D input embedding, used for biases and values.
            mask: (N, 1) mask to indicate which elements of inputs_1d participate
                in the attention.
            affine: QuatAffine object describing the position and orientation of
                every element in inputs_1d.
        Returns:
            Transformation of the input embedding.
        """
        num_residues, _ = inputs_1d.shape

        assert self.num_scalar_qk > 0
        assert self.num_point_qk > 0
        assert self.num_point_v > 0

        # Construct scalar queries of shape:
        # [num_query_residues, num_head, num_points]
        q_scalar = common_modules.Linear(
            self.num_head * self.num_scalar_qk, name='q_scalar')(
            inputs_1d)
        q_scalar = np.reshape(
            q_scalar, [num_residues, self.num_head, self.num_scalar_qk])

        # Construct scalar keys/values of shape:
        # [num_target_residues, num_head, num_points]
        kv_scalar = common_modules.Linear(
            self.num_head * (self.num_scalar_v + self.num_scalar_qk), name='kv_scalar')(
            inputs_1d)
        kv_scalar = np.reshape(kv_scalar,
                               [num_residues, self.num_head,
                                self.num_scalar_v + self.num_scalar_qk])
        k_scalar, v_scalar = np.split(kv_scalar, [self.num_scalar_qk], axis=-1)

        # Construct query points of shape:
        # [num_residues, num_head, num_point_qk]

        # First construct query points in local frame.
        q_point_local = common_modules.Linear(
            self.num_head * 3 * self.num_point_qk, name='q_point_local')(
            inputs_1d)
        q_point_local = np.split(q_point_local, 3, axis=-1)
        # Project query points into global frame.
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        # Reshape query point for later use.
        q_point = [
            np.reshape(x, [num_residues, self.num_head, self.num_point_qk])
            for x in q_point_global]

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        kv_point_local = common_modules.Linear(
            self.num_head * 3 * (self.num_point_qk + self.num_point_v), name='kv_point_local')(
            inputs_1d)
        kv_point_local = np.split(kv_point_local, 3, axis=-1)
        # Project key and value points into global frame.
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [
            np.reshape(x, [num_residues,
                           self.num_head, (self.num_point_qk + self.num_point_v)])
            for x in kv_point_global]
        # Split key and value points.
        k_point, v_point = list(
            zip(*[
                np.split(x, [self.num_point_qk, ], axis=-1)
                for x in kv_point_global
            ]))

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(self.num_scalar_qk, 1) * 1.
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(self.num_point_qk, 1) * 9. / 2

        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.

        num_logit_terms = 3

        scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance)).astype(np.float32)
        point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance)).astype(np.float32)
        attention_2d_weights = np.sqrt(1.0 / num_logit_terms).astype(np.float32)

        # Trainable per-head weights for points.
        # trainable_point_weights = jax.nn.softplus(hk.get_parameter(
        #     'trainable_point_weights', shape=[num_head],
        #     # softplus^{-1} (1)
        #     init=hk.initializers.Constant(np.log(np.exp(1.) - 1.))))
        trainable_point_weights = softplus(Constant(np.log(np.exp(1.) - 1.))(shape=(self.num_head,), dtype=np.float32))

        point_weights *= np.expand_dims(trainable_point_weights, axis=1)

        v_point = [np.swapaxes(x, -2, -3) for x in v_point]

        q_point = [np.swapaxes(x, -2, -3) for x in q_point]
        k_point = [np.swapaxes(x, -2, -3) for x in k_point]
        dist2 = [
            squared_difference(qx[:, :, None, :], kx[:, None, :, :])
            for qx, kx in zip(q_point, k_point)
        ]
        dist2 = sum(dist2)
        attn_qk_point = -0.5 * np.sum(
            point_weights[:, None, None, :] * dist2, axis=-1)

        v = np.swapaxes(v_scalar, -2, -3)
        q = np.swapaxes(scalar_weights * q_scalar, -2, -3)
        k = np.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = np.matmul(q, np.swapaxes(k, -2, -1))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = common_modules.Linear(
            self.num_head, name='attention_2d')(
            inputs_2d)

        attention_2d = np.transpose(attention_2d, [2, 0, 1])
        attention_2d = attention_2d_weights * attention_2d
        attn_logits += attention_2d

        mask_2d = mask * np.swapaxes(mask, -1, -2)
        attn_logits -= 1e5 * (1. - mask_2d)

        # [num_head, num_query_residues, num_target_residues]
        attn = softmax(attn_logits)

        # [num_head, num_query_residues, num_head * num_scalar_v]
        result_scalar = np.matmul(attn, v)

        # For point result, implement matmul manually so that it will be a float32
        # on TPU.  This is equivalent to
        # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
        #                        for vx in v_point]
        # but on the TPU, doing the multiply and reduce_sum ensures the
        # computation happens in float32 instead of bfloat16.
        result_point_global = [np.sum(
            attn[:, :, :, None] * vx[:, None, :, :],
            axis=-2) for vx in v_point]

        # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
        result_scalar = np.swapaxes(result_scalar, -2, -3)
        result_point_global = [
            np.swapaxes(x, -2, -3)
            for x in result_point_global]

        # Features used in the linear output projection. Should have the size
        # [num_query_residues, ?]
        output_features = []

        result_scalar = np.reshape(
            result_scalar, [num_residues, self.num_head * self.num_scalar_v])
        output_features.append(result_scalar)

        result_point_global = [
            np.reshape(r, [num_residues, self.num_head * self.num_point_v])
            for r in result_point_global]
        result_point_local = affine.invert_point(result_point_global, extra_dims=1)
        output_features.extend(result_point_local)

        output_features.append(np.sqrt(self._dist_epsilon +
                                       np.square(result_point_local[0]) +
                                       np.square(result_point_local[1]) +
                                       np.square(result_point_local[2])).astype(np.float32))

        # Dimensions: h = heads, i and j = residues,
        # c = inputs_2d channels
        # Contraction happens over the second residue dimension, similarly to how
        # the usual attention is performed.
        result_attention_over_2d = np.einsum('hij, ijc->ihc', attn, inputs_2d)
        num_out = self.num_head * result_attention_over_2d.shape[-1]
        output_features.append(
            np.reshape(result_attention_over_2d,
                       [num_residues, num_out]))

        final_init = 'zeros' if self._zero_initialize_last else 'linear'

        final_act = np.concatenate(output_features, axis=-1)

        return common_modules.Linear(
            self.num_output,
            initializer=final_init,
            name='output_projection')(final_act)


def l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / np.sqrt(np.maximum(np.sum(x ** 2, axis=axis, keepdims=True), epsilon)).astype(np.float32)


class MultiRigidSidechain:
    """Class to make side chain atoms."""

    def __init__(self, num_channel=128, zero_init=True, num_residual_block=2,
                 name='rigid_sidechain'):
        super().__init__(name=name)
        self.num_channel = num_channel
        self.zero_init = zero_init
        self.num_residual_block = num_residual_block

    def __call__(self, affine, representations_list, aatype):
        """Predict side chains using multi-rigid representations.
        Args:
            affine: The affines for each residue (translations in angstroms).
            representations_list: A list of activations to predict side chains from.
            aatype: Amino acid types.
        Returns:
            Dict containing atom positions and frames (in angstroms).
        """
        act = [
            common_modules.Linear(  # pylint: disable=g-complex-comprehension
                self.num_channel,
                name='input_projection')(relu(x))
            for x in representations_list
        ]
        # Sum the activation list (equivalent to concat then Linear).
        act = sum(act)

        final_init = 'zeros' if self.zero_init else 'linear'

        # Mapping with some residual blocks.
        for _ in range(self.num_residual_block):
            old_act = act
            act = common_modules.Linear(
                self.num_channel,
                initializer='relu',
                name='resblock1')(
                relu(act))
            act = common_modules.Linear(
                self.num_channel,
                initializer=final_init,
                name='resblock2')(
                relu(act))
            act += old_act

        # Map activations to torsion angles. Shape: (num_res, 14).
        num_res = act.shape[0]
        unnormalized_angles = common_modules.Linear(
            14, name='unnormalized_angles')(
            relu(act))
        unnormalized_angles = np.reshape(
            unnormalized_angles, [num_res, 7, 2])
        angles = l2_normalize(unnormalized_angles, axis=-1)

        outputs = {
            'angles_sin_cos': angles,  # jnp.ndarray (N, 7, 2)
            'unnormalized_angles_sin_cos':
                unnormalized_angles,  # jnp.ndarray (N, 7, 2)
        }

        # Map torsion angles to frames.
        backb_to_global = r3.rigids_from_quataffine(affine)

        # Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates"

        # r3.Rigids with shape (N, 8).
        all_frames_to_global = all_atom.torsion_angles_to_frames(
            aatype,
            backb_to_global,
            angles)

        # Use frames and literature positions to create the final atom coordinates.
        # r3.Vecs with shape (N, 14).
        pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global)

        outputs.update({
            'atom_pos': pred_positions,  # r3.Vecs (N, 14)
            'frames': all_frames_to_global,  # r3.Rigids (N, 8)
        })
        return outputs


class FoldIteration:
    """A single iteration of the main structure module loop.
    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" lines 6-21
    First, each residue attends to all residues using InvariantPointAttention.
    Then, we apply transition layers to update the hidden representations.
    Finally, we use the hidden representations to produce an update to the
    affine of each residue.
    """

    def __init__(self, deterministic=False, dropout=0.1, zero_init=True,
                 num_layer_in_transition=3, num_channel=384, position_scale=10.0,
                 name='fold_iteration'):
        super().__init__(name=name)
        self.deterministic = deterministic
        self.dropout = dropout
        self.zero_init = zero_init
        self.num_layer_in_transition = num_layer_in_transition
        self.num_channel = num_channel
        self.position_scale = position_scale

    def __call__(self,
                 activations,
                 sequence_mask,
                 update_affine,
                 is_training,
                 initial_act,
                 safe_key=None,
                 static_feat_2d=None,
                 aatype=None):

        # if safe_key is None:
        #     safe_key = prng.SafeKey(hk.next_rng_key())
        if safe_key is None:
            safe_key = np.random.default_rng()

        def safe_dropout_fn(tensor, safe_key):
            return safe_dropout(
                tensor=tensor,
                safe_key=safe_key,
                rate=self.dropout,
                is_deterministic=self.deterministic,
                is_training=is_training)

        affine = quat_affine.QuatAffine.from_tensor(activations['affine'])

        act = activations['act']
        attention_module = InvariantPointAttention()
        # Attention
        attn = attention_module(
            inputs_1d=act,
            inputs_2d=static_feat_2d,
            mask=sequence_mask,
            affine=affine)
        act += attn
        # safe_key, *sub_keys = safe_key.split(3)
        safe_key = safe_key.integers(low=0, high=10000, size=3).tolist()
        sub_keys = iter(safe_key)
        act = safe_dropout_fn(act, next(sub_keys))
        act = LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name='attention_layer_norm')(
            act)

        final_init = 'zeros' if self.zero_init else 'linear'

        # Transition
        input_act = act
        for i in range(self.num_layer_in_transition):
            init = 'relu' if i < self.num_layer_in_transition - 1 else final_init
            act = common_modules.Linear(
                self.num_channel,
                initializer=init,
                name='transition')(
                act)
            if i < self.num_layer_in_transition - 1:
                act = relu(act)
        act += input_act
        act = safe_dropout_fn(act, next(sub_keys))
        act = LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name='transition_layer_norm')(act)

        if update_affine:
            # This block corresponds to
            # Jumper et al. (2021) Alg. 23 "Backbone update"
            affine_update_size = 6

            # Affine update
            affine_update = common_modules.Linear(
                affine_update_size,
                initializer=final_init,
                name='affine_update')(
                act)

            affine = affine.pre_compose(affine_update)

        sc = MultiRigidSidechain()(affine.scale_translation(self.position_scale), [act, initial_act], aatype)

        outputs = {'affine': affine.to_tensor(), 'sc': sc}

        # affine = affine.apply_rotation_tensor_fn(jax.lax.stop_gradient)

        new_activations = {
            'act': act,
            'affine': affine.to_tensor()
        }
        return new_activations, outputs


def generate_new_affine(sequence_mask):
    num_residues, _ = sequence_mask.shape
    quaternion = np.tile(np.reshape(np.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])

    translation = np.zeros([num_residues, 3])
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)


def generate_affines(representations, batch, is_training, safe_key):
    """Generate predicted affines for a single chain.
    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
    This is the main part of the structure module - it iteratively applies
    folding to produce a set of predicted residue positions.
    Args:
        representations: Representations dictionary.
        batch: Batch dictionary.
        config: Config for the structure module.
        global_config: Global config.
        is_training: Whether the model is being trained.
        safe_key: A prng.SafeKey object that wraps a PRNG key.
    Returns:
        A dictionary containing residue affines and sidechain positions.
    """
    sequence_mask = batch['seq_mask'][:, None]

    act = LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='single_layer_norm')(
        representations['single'])

    initial_act = act

    num_channel = 384
    act = common_modules.Linear(num_channel, name='initial_projection')(act)

    affine = generate_new_affine(sequence_mask)

    fold_iteration = FoldIteration(name='fold_iteration')

    assert len(batch['seq_mask'].shape) == 1

    activations = {'act': act,
                   'affine': affine.to_tensor(),
                   }

    act_2d = LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='pair_layer_norm')(
        representations['pair'])

    outputs = []
    num_layer = 8
    safe_keys = safe_key.integers(low=0, high=10000, size=num_layer).tolist()
    for sub_key in safe_keys:
        activations, output = fold_iteration(
            activations,
            initial_act=initial_act,
            static_feat_2d=act_2d,
            safe_key=sub_key,
            sequence_mask=sequence_mask,
            update_affine=True,
            is_training=is_training,
            aatype=batch['aatype'])
        outputs.append(output)

    # output = jax.tree_map(lambda *x: np.stack(x), *outputs)
    output = np.stack(*outputs)
    # Include the activations in the output dict for use by the LDDT-Head.
    output['act'] = activations['act']

    return output


class StructureModule:
    """StructureModule as a network head.
    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
    """

    def __init__(self, compute_loss=True, compute_in_graph_metrics=True, position_scale=10.0,
                 weight_frac=0.5, structural_violation_loss_weight=1.0, name='structure_module'):
        self.compute_loss = compute_loss
        self.compute_in_graph_metrics = compute_in_graph_metrics
        self.position_scale = position_scale
        self.weight_frac = weight_frac
        self.structural_violation_loss_weight = structural_violation_loss_weight

    def __call__(self, representations, batch, is_training, safe_key=None):
        ret = {}

        # if safe_key is None:
        #     safe_key = prng.SafeKey(hk.next_rng_key())
        if safe_key is None:
            safe_key = np.random.default_rng()
        else:
            safe_key = np.random.default_rng(safe_key)

        output = generate_affines(
            representations=representations,
            batch=batch,
            is_training=is_training,
            safe_key=safe_key)

        ret['representations'] = {'structure_module': output['act']}

        ret['traj'] = output['affine'] * np.array([1.] * 4 + [self.position_scale] * 3)

        ret['sidechains'] = output['sc']

        atom14_pred_positions = r3.vecs_to_tensor(output['sc']['atom_pos'])[-1]
        ret['final_atom14_positions'] = atom14_pred_positions  # (N, 14, 3)
        ret['final_atom14_mask'] = batch['atom14_atom_exists']  # (N, 14)

        atom37_pred_positions = all_atom.atom14_to_atom37(atom14_pred_positions, batch)
        atom37_pred_positions *= batch['atom37_atom_exists'][:, :, None]
        ret['final_atom_positions'] = atom37_pred_positions  # (N, 37, 3)

        ret['final_atom_mask'] = batch['atom37_atom_exists']  # (N, 37)
        ret['final_affines'] = ret['traj'][-1]

        if self.compute_loss:
            return ret
        else:
            no_loss_features = ['final_atom_positions', 'final_atom_mask', 'representations']
            no_loss_ret = {k: ret[k] for k in no_loss_features}
            return no_loss_ret

    def loss(self, value, batch):
        ret = {'loss': 0.}

        ret['metrics'] = {}
        # If requested, compute in-graph metrics.
        # compute_in_graph_metrics = True
        if self.compute_in_graph_metrics:
            atom14_pred_positions = value['final_atom14_positions']
            # Compute renaming and violations.
            value.update(compute_renamed_ground_truth(batch, atom14_pred_positions))
            value['violations'] = find_structural_violations(batch, atom14_pred_positions)

            # Several violation metrics:
            violation_metrics = compute_violation_metrics(
                batch=batch,
                atom14_pred_positions=atom14_pred_positions,
                violations=value['violations'])
            ret['metrics'].update(violation_metrics)

        backbone_loss(ret, batch, value)

        if 'renamed_atom14_gt_positions' not in value:
            value.update(compute_renamed_ground_truth(batch, value['final_atom14_positions']))
        sc_loss = sidechain_loss(batch, value)

        ret['loss'] = ((1 - self.weight_frac) * ret['loss'] + self.weight_frac * sc_loss['loss'])
        ret['sidechain_fape'] = sc_loss['fape']

        supervised_chi_loss(ret, batch, value)

        if self.structural_violation_loss_weight:
            if 'violations' not in value:
                value['violations'] = find_structural_violations(batch, value['final_atom14_positions'])
            structural_violation_loss(ret, batch, value)

        return ret


def compute_renamed_ground_truth(
        batch: Dict[str, np.ndarray],
        atom14_pred_positions: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Find optimal renaming of ground truth based on the predicted positions.
    Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms"
    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.
    Shape (N).
    Args:
        batch: Dictionary containing:
            * atom14_gt_positions: Ground truth positions.
            * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
            * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
                renaming swaps.
            * atom14_gt_exists: Mask for which atoms exist in ground truth.
            * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
                after renaming.
            * atom14_atom_exists: Mask for whether each atom is part of the given
                amino acid type.
        atom14_pred_positions: Array of atom positions in global frame with shape
            (N, 14, 3).
    Returns:
        Dictionary containing:
            alt_naming_is_better: Array with 1.0 where alternative swap is better.
            renamed_atom14_gt_positions: Array of optimal ground truth positions
                after renaming swaps are performed.
            renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    alt_naming_is_better = all_atom.find_optimal_renaming(
        atom14_gt_positions=batch['atom14_gt_positions'],
        atom14_alt_gt_positions=batch['atom14_alt_gt_positions'],
        atom14_atom_is_ambiguous=batch['atom14_atom_is_ambiguous'],
        atom14_gt_exists=batch['atom14_gt_exists'],
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['atom14_atom_exists'])

    renamed_atom14_gt_positions = (
            (1. - alt_naming_is_better[:, None, None])
            * batch['atom14_gt_positions']
            + alt_naming_is_better[:, None, None]
            * batch['atom14_alt_gt_positions'])

    renamed_atom14_gt_mask = (
            (1. - alt_naming_is_better[:, None]) * batch['atom14_gt_exists']
            + alt_naming_is_better[:, None] * batch['atom14_alt_gt_exists'])

    return {
        'alt_naming_is_better': alt_naming_is_better,  # (N)
        'renamed_atom14_gt_positions': renamed_atom14_gt_positions,  # (N, 14, 3)
        'renamed_atom14_gt_exists': renamed_atom14_gt_mask,  # (N, 14)
    }


def find_structural_violations(
        batch: Dict[str, np.ndarray],
        atom14_pred_positions: np.ndarray,  # (N, 14, 3)
):
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    violation_tolerance_factor = 12.0
    connection_violations = all_atom.between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch['atom14_atom_exists'].astype(np.float32),
        residue_index=batch['residue_index'].astype(np.float32),
        aatype=batch['aatype'],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = np.array([
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ])
    # utils.batched_gather(dim=0) by default
    atom14_atom_radius = batch['atom14_atom_exists'] * utils.batched_gather(
        atomtype_radius, batch['residx_atom14_to_atom37'])

    # Compute the between residue clash loss.
    clash_overlap_tolerance = 1.5
    between_residue_clashes = all_atom.between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['atom14_atom_exists'],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch['residue_index'],
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance)

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor)
    atom14_dists_lower_bound = utils.batched_gather(
        restype_atom14_bounds['lower_bound'], batch['aatype'])
    atom14_dists_upper_bound = utils.batched_gather(
        restype_atom14_bounds['upper_bound'], batch['aatype'])
    within_residue_violations = all_atom.within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['atom14_atom_exists'],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0)

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = np.max(np.stack([
        connection_violations['per_residue_violation_mask'],
        np.max(between_residue_clashes['per_atom_clash_mask'], axis=-1),
        np.max(within_residue_violations['per_atom_violations'],
               axis=-1)]), axis=0)

    return {
        'between_residues': {
            'bonds_c_n_loss_mean':
                connection_violations['c_n_loss_mean'],  # ()
            'angles_ca_c_n_loss_mean':
                connection_violations['ca_c_n_loss_mean'],  # ()
            'angles_c_n_ca_loss_mean':
                connection_violations['c_n_ca_loss_mean'],  # ()
            'connections_per_residue_loss_sum':
                connection_violations['per_residue_loss_sum'],  # (N)
            'connections_per_residue_violation_mask':
                connection_violations['per_residue_violation_mask'],  # (N)
            'clashes_mean_loss':
                between_residue_clashes['mean_loss'],  # ()
            'clashes_per_atom_loss_sum':
                between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
            'clashes_per_atom_clash_mask':
                between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
        },
        'within_residues': {
            'per_atom_loss_sum':
                within_residue_violations['per_atom_loss_sum'],  # (N, 14)
            'per_atom_violations':
                within_residue_violations['per_atom_violations'],  # (N, 14),
        },
        'total_per_residue_violations_mask':
            per_residue_violations_mask,  # (N)
    }


def compute_violation_metrics(
        batch: Dict[str, np.ndarray],
        atom14_pred_positions: np.ndarray,  # (N, 14, 3)
        violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute several metrics to assess the structural violations."""

    ret = {}
    extreme_ca_ca_violations = all_atom.extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch['atom14_atom_exists'].astype(np.float32),
        residue_index=batch['residue_index'].astype(np.float32))
    ret['violations_extreme_ca_ca_distance'] = extreme_ca_ca_violations
    ret['violations_between_residue_bond'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=violations['between_residues'][
            'connections_per_residue_violation_mask'])
    ret['violations_between_residue_clash'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=np.max(
            violations['between_residues']['clashes_per_atom_clash_mask'],
            axis=-1))
    ret['violations_within_residue'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=np.max(
            violations['within_residues']['per_atom_violations'], axis=-1))
    ret['violations_per_residue'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=violations['total_per_residue_violations_mask'])
    return ret


def backbone_loss(ret, batch, value):
    """Backbone FAPE Loss.
    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17
    Args:
        ret: Dictionary to write outputs into, needs to contain 'loss'.
        batch: Batch, needs to contain 'backbone_affine_tensor',
            'backbone_affine_mask'.
        value: Dictionary containing structure module output, needs to contain
            'traj', a trajectory of rigids.
        config: Configuration of loss, should contain 'fape.clamp_distance' and
            'fape.loss_unit_distance'.
    """
    affine_trajectory = quat_affine.QuatAffine.from_tensor(value['traj'])
    rigid_trajectory = r3.rigids_from_quataffine(affine_trajectory)

    gt_affine = quat_affine.QuatAffine.from_tensor(
        batch['backbone_affine_tensor'])
    gt_rigid = r3.rigids_from_quataffine(gt_affine)
    backbone_mask = batch['backbone_affine_mask']

    clamp_distance = 10.0
    loss_unit_distance = 10.0
    # fape_loss_fn = functools.partial(
    #     all_atom.frame_aligned_point_error,
    #     l1_clamp_distance=clamp_distance,
    #     length_scale=loss_unit_distance)
    # fape_loss_fn = jax.vmap(fape_loss_fn, (0, None, None, 0, None, None))  # function
    # fape_loss = fape_loss_fn(rigid_trajectory,    # shape (num_frames)  # result
    #                          gt_rigid,    # shape (num_frames)
    #                          backbone_mask,   # shape (num_frames)
    #                          rigid_trajectory.trans,  # shape (num_positions)
    #                          gt_rigid.trans,  # shape (num_positions)
    #                          backbone_mask)   # shape (num_positions)
    fape_loss = all_atom.frame_aligned_point_error(rigid_trajectory, gt_rigid, backbone_mask,
                                                   rigid_trajectory.trans, gt_rigid.trans, backbone_mask,
                                                   length_scale=loss_unit_distance,
                                                   l1_clamp_distance=clamp_distance)  # not sliced

    if 'use_clamped_fape' in batch:
        # Jumper et al. (2021) Suppl. Sec. 1.11.5 "Loss clamping details"
        use_clamped_fape = np.asarray(batch['use_clamped_fape'], np.float32)
        # unclamped_fape_loss_fn = functools.partial(
        #     all_atom.frame_aligned_point_error,
        #     l1_clamp_distance=None,
        #     length_scale=loss_unit_distance)
        # unclamped_fape_loss_fn = jax.vmap(unclamped_fape_loss_fn, (0, None, None, 0, None, None))
        # fape_loss_unclamped = unclamped_fape_loss_fn(rigid_trajectory, gt_rigid, backbone_mask,
        #                                              rigid_trajectory.trans, gt_rigid.trans, backbone_mask)
        fape_loss_unclamped = all_atom.frame_aligned_point_error(rigid_trajectory, gt_rigid, backbone_mask,
                                                                 rigid_trajectory.trans, gt_rigid.trans, backbone_mask,
                                                                 length_scale=loss_unit_distance,
                                                                 l1_clamp_distance=None)  # not sliced

        fape_loss = (fape_loss * use_clamped_fape + fape_loss_unclamped * (1 - use_clamped_fape))

    ret['fape'] = fape_loss[-1]
    ret['loss'] += np.mean(fape_loss)


def sidechain_loss(batch, value):
    """All Atom FAPE Loss using renamed rigids."""
    # Rename Frames
    # Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms" line 7
    alt_naming_is_better = value['alt_naming_is_better']
    renamed_gt_frames = (
            (1. - alt_naming_is_better[:, None, None])
            * batch['rigidgroups_gt_frames']
            + alt_naming_is_better[:, None, None]
            * batch['rigidgroups_alt_gt_frames'])

    flat_gt_frames = r3.rigids_from_tensor_flat12(np.reshape(renamed_gt_frames, [-1, 12]))
    flat_frames_mask = np.reshape(batch['rigidgroups_gt_exists'], [-1])

    flat_gt_positions = r3.vecs_from_tensor(np.reshape(value['renamed_atom14_gt_positions'], [-1, 3]))
    flat_positions_mask = np.reshape(value['renamed_atom14_gt_exists'], [-1])

    # Compute frame_aligned_point_error score for the final layer.
    pred_frames = value['sidechains']['frames']  # Rigids
    pred_positions = value['sidechains']['atom_pos']  # Vecs

    def _slice_last_layer_and_flatten(x):
        return np.reshape(x[-1], [-1])

    # flat_pred_frames = jax.tree_map(_slice_last_layer_and_flatten, pred_frames)
    # flat_pred_positions = jax.tree_map(_slice_last_layer_and_flatten, pred_positions)
    flat_pred_frames = r3.Rigids(_slice_last_layer_and_flatten(pred_frames[0]),
                                 _slice_last_layer_and_flatten(pred_frames[1]))
    flat_pred_positions = r3.Vecs(_slice_last_layer_and_flatten(pred_positions[0]),
                                  _slice_last_layer_and_flatten(pred_positions[1]),
                                  _slice_last_layer_and_flatten(pred_positions[2]))

    # FAPE Loss on sidechains
    atom_clamp_distance = 10.0
    length_scale = 10.
    fape = all_atom.frame_aligned_point_error(
        pred_frames=flat_pred_frames,
        target_frames=flat_gt_frames,
        frames_mask=flat_frames_mask,
        pred_positions=flat_pred_positions,
        target_positions=flat_gt_positions,
        positions_mask=flat_positions_mask,
        l1_clamp_distance=atom_clamp_distance,
        length_scale=length_scale)

    return {
        'fape': fape,
        'loss': fape}


def supervised_chi_loss(ret, batch, value):
    """Computes loss for direct chi angle supervision.
    Jumper et al. (2021) Suppl. Alg. 27 "torsionAngleLoss"
    Args:
        ret: Dictionary to write outputs into, needs to contain 'loss'.
        batch: Batch, needs to contain 'seq_mask', 'chi_mask', 'chi_angles'.
        value: Dictionary containing structure module output, needs to contain
            value['sidechains']['angles_sin_cos'] for angles and
            value['sidechains']['unnormalized_angles_sin_cos'] for unnormalized
                angles.
        config: Configuration of loss, should contain 'chi_weight' and
            'angle_norm_weight', 'angle_norm_weight' scales angle norm term,
            'chi_weight' scales torsion term.
    """
    eps = 1e-6

    sequence_mask = batch['seq_mask']
    num_res = sequence_mask.shape[0]
    chi_mask = batch['chi_mask'].astype(np.float32)
    pred_angles = np.reshape(
        value['sidechains']['angles_sin_cos'], [-1, num_res, 7, 2])
    pred_angles = pred_angles[:, :, 3:]

    residue_type_one_hot = one_hot(batch['aatype'], residue_constants.restype_num + 1, dtype=np.float32)[None]
    chi_pi_periodic = np.einsum('ijk, kl->ijl', residue_type_one_hot,
                                np.asarray(residue_constants.chi_pi_periodic))

    true_chi = batch['chi_angles'][None]
    sin_true_chi = np.sin(true_chi)
    cos_true_chi = np.cos(true_chi)
    sin_cos_true_chi = np.stack([sin_true_chi, cos_true_chi], axis=-1)

    # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
    shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi

    sq_chi_error = np.sum(
        squared_difference(sin_cos_true_chi, pred_angles), -1)
    sq_chi_error_shifted = np.sum(
        squared_difference(sin_cos_true_chi_shifted, pred_angles), -1)
    sq_chi_error = np.minimum(sq_chi_error, sq_chi_error_shifted)

    sq_chi_loss = utils.mask_mean(mask=chi_mask[None], value=sq_chi_error)
    ret['chi_loss'] = sq_chi_loss
    chi_weight = 0.5
    ret['loss'] += chi_weight * sq_chi_loss
    unnormed_angles = np.reshape(
        value['sidechains']['unnormalized_angles_sin_cos'], [-1, num_res, 7, 2])
    angle_norm = np.sqrt(np.sum(np.square(unnormed_angles), axis=-1) + eps).astype(np.float32)
    norm_error = np.abs(angle_norm - 1.).astype(np.float32)
    angle_norm_loss = utils.mask_mean(mask=sequence_mask[None, :, None], value=norm_error)

    ret['angle_norm_loss'] = angle_norm_loss
    angle_norm_weight = 0.01
    ret['loss'] += angle_norm_weight * angle_norm_loss


def structural_violation_loss(ret, batch, value):
    """Computes loss for structural violations."""
    weight_frac = 0.5
    assert weight_frac

    # Put all violation losses together to one large loss.
    violations = value['violations']
    num_atoms = np.sum(batch['atom14_atom_exists']).astype(np.float32)
    structural_violation_loss_weight = 1.0
    ret['loss'] += (structural_violation_loss_weight * (
            violations['between_residues']['bonds_c_n_loss_mean'] +
            violations['between_residues']['angles_ca_c_n_loss_mean'] +
            violations['between_residues']['angles_c_n_ca_loss_mean'] +
            np.sum(
                violations['between_residues']['clashes_per_atom_loss_sum'] +
                violations['within_residues']['per_atom_loss_sum']) /
            (1e-6 + num_atoms)))

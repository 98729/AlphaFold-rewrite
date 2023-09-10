import r3
import numpy as np
import utils
import residue_constants
from typing import Dict, Optional
from folding import one_hot


def relu(x: int or float or np.ndarray):
    return np.maximum(0, x).astype(np.float32)


def squared_difference(x, y):
    return np.square(x - y)


def torsion_angles_to_frames(
        aatype: np.ndarray,  # (N)
        backb_to_global: r3.Rigids,  # (N)
        torsion_angles_sin_cos: np.ndarray  # (N, 7, 2)
) -> r3.Rigids:  # (N, 8)
    """Compute rigid group frames from torsion angles.
    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" lines 2-10
    Jumper et al. (2021) Suppl. Alg. 25 "makeRotX"
    Args:
        aatype: aatype for each residue
        backb_to_global: Rigid transformations describing transformation from
          backbone frame to global frame.
        torsion_angles_sin_cos: sin and cosine of the 7 torsion angles
    Returns:
        Frames corresponding to all the Sidechain Rigid Transforms
    """
    assert len(aatype.shape) == 1
    assert len(backb_to_global.rot.xx.shape) == 1
    assert len(torsion_angles_sin_cos.shape) == 3
    assert torsion_angles_sin_cos.shape[1] == 7
    assert torsion_angles_sin_cos.shape[2] == 2

    # Gather the default frames for all rigid groups.
    # r3.Rigids with shape (N, 8)
    m = utils.batched_gather(residue_constants.restype_rigid_group_default_frame, aatype)
    default_frames = r3.rigids_from_tensor4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues, = aatype.shape
    sin_angles = np.concatenate([np.zeros([num_residues, 1]), sin_angles], axis=-1)
    cos_angles = np.concatenate([np.ones([num_residues, 1]), cos_angles], axis=-1)
    zeros = np.zeros_like(sin_angles)
    ones = np.ones_like(sin_angles)

    # all_rots are r3.Rots with shape (N, 8)
    all_rots = r3.Rots(ones, zeros, zeros,
                       zeros, cos_angles, -sin_angles,
                       zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = r3.rigids_mul_rots(default_frames, all_rots)  # Rigids

    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    # chi2_frame_to_frame = jax.tree_map(lambda x: x[:, 5], all_frames)
    # chi3_frame_to_frame = jax.tree_map(lambda x: x[:, 6], all_frames)
    # chi4_frame_to_frame = jax.tree_map(lambda x: x[:, 7], all_frames)
    # chi1_frame_to_backb = jax.tree_map(lambda x: x[:, 4], all_frames)
    chi2_frame_to_frame = r3.Rigids(all_frames.rot[:, 5], all_frames.trans[:, 5])
    chi3_frame_to_frame = r3.Rigids(all_frames.rot[:, 6], all_frames.trans[:, 6])
    chi4_frame_to_frame = r3.Rigids(all_frames.rot[:, 7], all_frames.trans[:, 7])
    chi1_frame_to_backb = r3.Rigids(all_frames.rot[:, 4], all_frames.trans[:, 4])

    chi2_frame_to_backb = r3.rigids_mul_rigids(chi1_frame_to_backb,  # Rigids
                                               chi2_frame_to_frame)  # Rigids
    chi3_frame_to_backb = r3.rigids_mul_rigids(chi2_frame_to_backb,  # Rigids
                                               chi3_frame_to_frame)  # Rigids
    chi4_frame_to_backb = r3.rigids_mul_rigids(chi3_frame_to_backb,  # Rigids
                                               chi4_frame_to_frame)  # Rigids

    # Recombine them to a r3.Rigids with shape (N, 8).
    def _concat_frames(xall, x5, x6, x7):
        return np.concatenate([xall[:, 0:5], x5[:, None], x6[:, None], x7[:, None]], axis=-1)

    # all_frames_to_backb = jax.tree_map(
    #     _concat_frames,
    #     all_frames,  # Rigids
    #     chi2_frame_to_backb,  # Rigids
    #     chi3_frame_to_backb,  # Rigids
    #     chi4_frame_to_backb)  # Rigids
    all_frames_to_backb = [_concat_frames(all_frames.rot, chi2_frame_to_backb.rot, chi3_frame_to_backb.rot,
                                          chi4_frame_to_backb.rot),
                           _concat_frames(all_frames.trans, chi2_frame_to_backb.trans, chi3_frame_to_backb.trans,
                                          chi4_frame_to_backb.trans)]

    # Create the global frames.
    # shape (N, 8)
    arr = np.expand_dims(backb_to_global.rot, backb_to_global.trans, axis=1)
    all_frames_to_global = r3.rigids_mul_rigids(
        r3.Rigids(arr[0], arr[1]),  # Rigids
        r3.Rigids(all_frames_to_backb[0], all_frames_to_backb[1]))  # Rigids

    return all_frames_to_global


def frame_aligned_point_error(
        pred_frames: r3.Rigids,  # shape (num_frames)           1
        target_frames: r3.Rigids,  # shape (num_frames)         2
        frames_mask: np.ndarray,  # shape (num_frames)          3
        pred_positions: r3.Vecs,  # shape (num_positions)       4
        target_positions: r3.Vecs,  # shape (num_positions)     5
        positions_mask: np.ndarray,  # shape (num_positions)    6
        length_scale: float,
        l1_clamp_distance: Optional[float] = None,
        epsilon=1e-4) -> np.ndarray:  # shape ()
    """Measure point error under different alignments.
    Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"
    Computes error between two structures with B points under A alignments derived
    from the given pairs of frames.
    Args:
        pred_frames: num_frames reference frames for 'pred_positions'.
        target_frames: num_frames reference frames for 'target_positions'.
        frames_mask: Mask for frame pairs to use.
        pred_positions: num_positions predicted positions of the structure.
        target_positions: num_positions target positions of the structure.
        positions_mask: Mask on which positions to score.
        length_scale: length scale to divide loss by.
        l1_clamp_distance: Distance cutoff on error beyond which gradients will
            be zero.
        epsilon: small value used to regularize denominator for masked average.
    Returns:
        Masked Frame Aligned Point Error.
    """
    assert pred_frames.rot.xx.ndim == 1
    assert target_frames.rot.xx.ndim == 1
    assert frames_mask.ndim == 1, frames_mask.ndim
    assert pred_positions.x.ndim == 1
    assert target_positions.x.ndim == 1
    assert positions_mask.ndim == 1

    # Compute array of predicted positions in the predicted frames.
    # r3.Vecs (num_frames, num_positions)
    # local_pred_pos = r3.rigids_mul_vecs(
    #     jax.tree_map(lambda r: r[:, None], r3.invert_rigids(pred_frames)),
    #     jax.tree_map(lambda x: x[None, :], pred_positions))
    arr = np.expand_dims(r3.invert_rigids(pred_frames).rot, r3.invert_rigids(pred_frames).trans, axis=1)
    arr1 = np.expand_dims(pred_positions.x, pred_positions.y, pred_positions.z, axis=0)
    local_pred_pos = r3.rigids_mul_vecs(
        r3.Rigids(arr[0], arr[1]),  # Rigids
        r3.Vecs(arr1[0], arr1[1], arr1[2]))  # Vecs

    # Compute array of target positions in the target frames.
    # r3.Vecs (num_frames, num_positions)
    # local_target_pos = r3.rigids_mul_vecs(
    #     jax.tree_map(lambda r: r[:, None], r3.invert_rigids(target_frames)),
    #     jax.tree_map(lambda x: x[None, :], target_positions))
    arr2 = np.expand_dims(r3.invert_rigids(target_frames).rot, r3.invert_rigids(target_frames).trans, axis=1)
    arr3 = np.expand_dims(target_positions.x, target_positions.y, target_positions.z, axis=0)
    local_target_pos = r3.rigids_mul_vecs(
        r3.Rigids(arr2[0], arr2[1]),  # Rigids
        r3.Vecs(arr3[0], arr3[1], arr3[2]))  # Vecs

    # Compute errors between the structures.
    # jnp.ndarray (num_frames, num_positions)
    error_dist = np.sqrt(
        r3.vecs_squared_distance(local_pred_pos, local_target_pos)
        + epsilon).astype(np.float32)

    if l1_clamp_distance:
        error_dist = np.clip(error_dist, 0, l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error *= np.expand_dims(frames_mask, axis=-1)
    normed_error *= np.expand_dims(positions_mask, axis=-2)

    normalization_factor = (np.sum(frames_mask, axis=-1) * np.sum(positions_mask, axis=-1))
    return np.sum(normed_error, axis=(-2, -1)) / (epsilon + normalization_factor)


def atom14_to_atom37(atom14_data: np.ndarray,  # (N, 14, ...)
                     batch: Dict[str, np.ndarray]
                     ) -> np.ndarray:  # (N, 37, ...)
    """Convert atom14 to atom37 representation."""
    assert len(atom14_data.shape) in [2, 3]
    assert 'residx_atom37_to_atom14' in batch
    assert 'atom37_atom_exists' in batch

    atom37_data = utils.batched_gather(atom14_data,
                                       batch['residx_atom37_to_atom14'],  # 172 * 37
                                       batch_dims=1)
    if len(atom14_data.shape) == 2:
        atom37_data *= batch['atom37_atom_exists']
    elif len(atom14_data.shape) == 3:
        atom37_data *= batch['atom37_atom_exists'][:, :, None].astype(atom37_data.dtype)
    return atom37_data


def find_optimal_renaming(
        atom14_gt_positions: np.ndarray,  # (N, 14, 3)
        atom14_alt_gt_positions: np.ndarray,  # (N, 14, 3)
        atom14_atom_is_ambiguous: np.ndarray,  # (N, 14)
        atom14_gt_exists: np.ndarray,  # (N, 14)
        atom14_pred_positions: np.ndarray,  # (N, 14, 3)
        atom14_atom_exists: np.ndarray,  # (N, 14)
) -> np.ndarray:  # (N):
    """Find optimal renaming for ground truth that maximizes LDDT.
    Jumper et al. (2021) Suppl. Alg. 26
    "renameSymmetricGroundTruthAtoms" lines 1-5
    Args:
        atom14_gt_positions: Ground truth positions in global frame of ground truth.
        atom14_alt_gt_positions: Alternate ground truth positions in global frame of
            ground truth with coordinates of ambiguous atoms swapped relative to
            'atom14_gt_positions'.
        atom14_atom_is_ambiguous: Mask denoting whether atom is among ambiguous
            atoms, see Jumper et al. (2021) Suppl. Table 3
        atom14_gt_exists: Mask denoting whether atom at positions exists in ground truth.
        atom14_pred_positions: Predicted positions of atoms in
            global prediction frame
        atom14_atom_exists: Mask denoting whether atom at positions exists for given
            amino acid type
    Returns:
        Float array of shape [N] with 1. where atom14_alt_gt_positions is closer to
        prediction and 0. otherwise
    """
    assert len(atom14_gt_positions.shape) == 3
    assert len(atom14_alt_gt_positions.shape) == 3
    assert len(atom14_atom_is_ambiguous.shape) == 2
    assert len(atom14_gt_exists.shape) == 2
    assert len(atom14_pred_positions.shape) == 3
    assert len(atom14_atom_exists.shape) == 2

    # Create the pred distance matrix.
    # shape (N, N, 14, 14)
    pred_dists = np.sqrt(1e-10 + np.sum(
        squared_difference(
            atom14_pred_positions[:, None, :, None, :],
            atom14_pred_positions[None, :, None, :, :]),
        axis=-1)).astype(np.float32)

    # Compute distances for ground truth with original and alternative names.
    # shape (N, N, 14, 14)
    gt_dists = np.sqrt(1e-10 + np.sum(
        squared_difference(
            atom14_gt_positions[:, None, :, None, :],
            atom14_gt_positions[None, :, None, :, :]),
        axis=-1)).astype(np.float32)
    alt_gt_dists = np.sqrt(1e-10 + np.sum(
        squared_difference(
            atom14_alt_gt_positions[:, None, :, None, :],
            atom14_alt_gt_positions[None, :, None, :, :]),
        axis=-1)).astype(np.float32)

    # Compute LDDT's.
    # shape (N, N, 14, 14)
    lddt = np.sqrt(1e-10 + squared_difference(pred_dists, gt_dists)).astype(np.float32)
    alt_lddt = np.sqrt(1e-10 + squared_difference(pred_dists, alt_gt_dists)).astype(np.float32)

    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    # shape (N ,N, 14, 14)
    mask = (atom14_gt_exists[:, None, :, None] *  # rows
            atom14_atom_is_ambiguous[:, None, :, None] *  # rows
            atom14_gt_exists[None, :, None, :] *  # cols
            (1. - atom14_atom_is_ambiguous[None, :, None, :]))  # cols

    # Aggregate distances for each residue to the non-amibuguous atoms.
    # shape (N)
    per_res_lddt = np.sum(mask * lddt, axis=[1, 2, 3])
    alt_per_res_lddt = np.sum(mask * alt_lddt, axis=[1, 2, 3])

    # Decide for each residue, whether alternative naming is better.
    # shape (N)
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).astype(np.float32)

    return alt_naming_is_better  # shape (N)


def between_residue_bond_loss(
        pred_atom_positions: np.ndarray,  # (N, 37(14), 3)
        pred_atom_mask: np.ndarray,  # (N, 37(14))
        residue_index: np.ndarray,  # (N)
        aatype: np.ndarray,  # (N)
        tolerance_factor_soft=12.0,
        tolerance_factor_hard=12.0
) -> Dict[str, np.ndarray]:
    """Flat-bottom loss to penalize structural violations between residues.
    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.
    Args:
        pred_atom_positions: Atom positions in atom37/14 representation
        pred_atom_mask: Atom mask in atom37/14 representation
        residue_index: Residue index for given amino acid, this is assumed to be
          monotonically increasing.
        aatype: Amino acid type of given residue
        tolerance_factor_soft: soft tolerance factor measured in standard deviations
          of pdb distributions
        tolerance_factor_hard: hard tolerance factor measured in standard deviations
          of pdb distributions
    Returns:
        Dict containing:
            * 'c_n_loss_mean': Loss for peptide bond length violations
            * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
                by CA, C, N
            * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
                by C, N, CA
            * 'per_residue_loss_sum': sum of all losses for each residue
            * 'per_residue_violation_mask': mask denoting all residues with violation
                present.
    """
    assert len(pred_atom_positions.shape) == 3
    assert len(pred_atom_mask.shape) == 2
    assert len(residue_index.shape) == 1
    assert len(aatype.shape) == 1

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:-1, 1]  # (N - 1)
    this_c_pos = pred_atom_positions[:-1, 2, :]  # (N - 1, 3)
    this_c_mask = pred_atom_mask[:-1, 2]  # (N - 1)
    next_n_pos = pred_atom_positions[1:, 0, :]  # (N - 1, 3)
    next_n_mask = pred_atom_mask[1:, 0]  # (N - 1)
    next_ca_pos = pred_atom_positions[1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(np.float32)

    # Compute loss for the C--N bond.
    c_n_bond_length = np.sqrt(
        1e-6 + np.sum(squared_difference(this_c_pos, next_n_pos), axis=-1)).astype(np.float32)

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (aatype[1:] == residue_constants.resname_to_idx['PRO']).astype(np.float32)
    gt_length = (
            (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
            + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = (
            (1. - next_is_proline) *
            residue_constants.between_res_bond_length_stddev_c_n[0] +
            next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = np.sqrt(1e-6 + np.square(c_n_bond_length - gt_length)).astype(np.float32)
    c_n_loss_per_residue = relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = np.sum(mask * c_n_loss_per_residue) / (np.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (
            c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = np.sqrt(1e-6 + np.sum(
        squared_difference(this_ca_pos, this_c_pos), axis=-1)).astype(np.float32)
    n_ca_bond_length = np.sqrt(1e-6 + np.sum(
        squared_difference(next_n_pos, next_ca_pos), axis=-1)).astype(np.float32)

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

    ca_c_n_cos_angle = np.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = np.sqrt(
        1e-6 + np.square(ca_c_n_cos_angle - gt_angle)).astype(np.float32)
    ca_c_n_loss_per_residue = relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = np.sum(mask * ca_c_n_loss_per_residue) / (np.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = np.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = np.sqrt(1e-6 + np.square(c_n_ca_cos_angle - gt_angle)).astype(np.float32)
    c_n_ca_loss_per_residue = relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = np.sum(mask * c_n_ca_loss_per_residue) / (np.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (c_n_loss_per_residue +
                            ca_c_n_loss_per_residue +
                            c_n_ca_loss_per_residue)
    per_residue_loss_sum = 0.5 * (np.pad(per_residue_loss_sum, [[0, 1]]) +
                                  np.pad(per_residue_loss_sum, [[1, 0]]))

    # Compute hard violations.
    violation_mask = np.max(
        np.stack([c_n_violation_mask,
                  ca_c_n_violation_mask,
                  c_n_ca_violation_mask]), axis=0)
    violation_mask = np.maximum(
        np.pad(violation_mask, [[0, 1]]),
        np.pad(violation_mask, [[1, 0]]))

    return {'c_n_loss_mean': c_n_loss,  # shape ()
            'ca_c_n_loss_mean': ca_c_n_loss,  # shape ()
            'c_n_ca_loss_mean': c_n_ca_loss,  # shape ()
            'per_residue_loss_sum': per_residue_loss_sum,  # shape (N)
            'per_residue_violation_mask': violation_mask  # shape (N)
            }


def between_residue_clash_loss(
        atom14_pred_positions: np.ndarray,  # (N, 14, 3)
        atom14_atom_exists: np.ndarray,  # (N, 14)
        atom14_atom_radius: np.ndarray,  # (N, 14)
        residue_index: np.ndarray,  # (N)
        overlap_tolerance_soft=1.5,
        overlap_tolerance_hard=1.5
) -> Dict[str, np.ndarray]:
    """Loss to penalize steric clashes between residues.
    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.
    Args:
        atom14_pred_positions: Predicted positions of atoms in global prediction frame
        atom14_atom_exists: Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_atom_radius: Van der Waals radius for each atom.
        residue_index: Residue index for given amino acid.
        overlap_tolerance_soft: Soft tolerance factor.
        overlap_tolerance_hard: Hard tolerance factor.
    Returns:
        Dict containing:
            * 'mean_loss': average clash loss
            * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
            * 'per_atom_clash_mask': mask whether atom clashes with any other atom
                shape (N, 14)
    """
    assert len(atom14_pred_positions.shape) == 3
    assert len(atom14_atom_exists.shape) == 2
    assert len(atom14_atom_radius.shape) == 2
    assert len(residue_index.shape) == 1

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = np.sqrt(1e-10 + np.sum(
        squared_difference(
            atom14_pred_positions[:, None, :, None, :],
            atom14_pred_positions[None, :, None, :, :]),
        axis=-1)).astype(np.float32)

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (atom14_atom_exists[:, None, :, None] *
                  atom14_atom_exists[None, :, None, :])

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask *= (
            residue_index[:, None, None, None] < residue_index[None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = one_hot(2, num_classes=14)
    n_one_hot = one_hot(0, num_classes=14)
    neighbour_mask = ((residue_index[:, None, None, None] +
                       1) == residue_index[None, :, None, None])
    c_n_bonds = neighbour_mask * c_one_hot[None, None, :, None] * n_one_hot[None, None, None, :]
    dists_mask *= (1. - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
    cys_sg_one_hot = one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (cys_sg_one_hot[None, None, :, None] *
                       cys_sg_one_hot[None, None, None, :])
    dists_mask *= (1. - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] +
                                      atom14_atom_radius[None, :, None, :])

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * relu(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape ()
    mean_loss = (np.sum(dists_to_low_error)
                 / (1e-6 + np.sum(dists_mask)))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = (np.sum(dists_to_low_error, axis=[0, 2]) +
                         np.sum(dists_to_low_error, axis=[1, 3]))

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
            dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = np.maximum(
        np.max(clash_mask, axis=[0, 2]),
        np.max(clash_mask, axis=[1, 3]))

    return {'mean_loss': mean_loss,  # shape ()
            'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
            'per_atom_clash_mask': per_atom_clash_mask  # shape (N, 14)
            }


def within_residue_violations(
        atom14_pred_positions: np.ndarray,  # (N, 14, 3)
        atom14_atom_exists: np.ndarray,  # (N, 14)
        atom14_dists_lower_bound: np.ndarray,  # (N, 14, 14)
        atom14_dists_upper_bound: np.ndarray,  # (N, 14, 14)
        tighten_bounds_for_loss=0.0,
) -> Dict[str, np.ndarray]:
    """Loss to penalize steric clashes within residues.
    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.
    Args:
        atom14_pred_positions: Predicted positions of atoms in global prediction frame
        atom14_atom_exists: Mask denoting whether atom at positions exists for given amino acid type
        atom14_dists_lower_bound: Lower bound on allowed distances.
        atom14_dists_upper_bound: Upper bound on allowed distances
        tighten_bounds_for_loss: Extra factor to tighten loss
    Returns:
        Dict containing:
            * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
            * 'per_atom_clash_mask': mask whether atom clashes with any other atom
                shape (N, 14)
    """
    assert len(atom14_pred_positions.shape) == 3
    assert len(atom14_atom_exists.shape) == 2
    assert len(atom14_dists_lower_bound.shape) == 3
    assert len(atom14_dists_upper_bound.shape) == 3

    # Compute the mask for each residue.
    # shape (N, 14, 14)
    dists_masks = (1. - np.eye(14, 14)[None])
    dists_masks *= (atom14_atom_exists[:, :, None] *
                    atom14_atom_exists[:, None, :])

    # Distance matrix
    # shape (N, 14, 14)
    dists = np.sqrt(1e-10 + np.sum(
        squared_difference(
            atom14_pred_positions[:, :, None, :],
            atom14_pred_positions[:, None, :, :]),
        axis=-1)).astype(np.float32)

    # Compute the loss.
    # shape (N, 14, 14)
    dists_to_low_error = relu(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = relu(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = (np.sum(loss, axis=1) +
                         np.sum(loss, axis=2))

    # Compute the violations mask.
    # shape (N, 14, 14)
    violations = dists_masks * ((dists < atom14_dists_lower_bound) |
                                (dists > atom14_dists_upper_bound))

    # Compute the per atom violations.
    # shape (N, 14)
    per_atom_violations = np.maximum(np.max(violations, axis=1), np.max(violations, axis=2))

    return {'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
            'per_atom_violations': per_atom_violations  # shape (N, 14)
            }


def extreme_ca_ca_distance_violations(
        pred_atom_positions: np.ndarray,  # (N, 37(14), 3)
        pred_atom_mask: np.ndarray,  # (N, 37(14))
        residue_index: np.ndarray,  # (N)
        max_angstrom_tolerance=1.5
) -> np.ndarray:
    """Counts residues whose Ca is a large distance from its neighbour.
  Measures the fraction of CA-CA pairs between consecutive amino acids that are
  more than 'max_angstrom_tolerance' apart.
  Args:
    pred_atom_positions: Atom positions in atom37/14 representation
    pred_atom_mask: Atom mask in atom37/14 representation
    residue_index: Residue index for given amino acid, this is assumed to be
      monotonically increasing.
    max_angstrom_tolerance: Maximum distance allowed to not count as violation.
  Returns:
    Fraction of consecutive CA-CA pairs with violation.
  """
    this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:-1, 1]  # (N - 1)
    next_ca_pos = pred_atom_positions[1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
        np.float32)
    ca_ca_distance = np.sqrt(
        1e-6 + np.sum(squared_difference(this_ca_pos, next_ca_pos), axis=-1)).astype(np.float32)
    violations = (ca_ca_distance - residue_constants.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return utils.mask_mean(mask=mask, value=violations)


def frames_and_literature_positions_to_atom14_pos(
        aatype: np.ndarray,  # (N)
        all_frames_to_global: r3.Rigids  # (N, 8)
) -> r3.Vecs:  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.
    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11
    Args:
        aatype: aatype for each residue.
        all_frames_to_global: All per residue coordinate frames.
    Returns:
        Positions of all atom coordinates in global frame.
    """

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = utils.batched_gather(
        residue_constants.restype_atom14_to_rigid_group, aatype)
    group_mask = one_hot(residx_to_group_idx, num_classes=8)  # shape (N, 14, 8)

    # r3.Rigids with shape (N, 14)
    # map_atoms_to_global = jax.tree_map(
    #     lambda x: np.sum(x[:, None, :] * group_mask, axis=-1),
    #     all_frames_to_global)  # Rigids
    map_atoms_to_global = r3.Rigids(np.sum(all_frames_to_global.rot[:, None, :] * group_mask, axis=-1),
                                    np.sum(all_frames_to_global.trans[:, None, :] * group_mask, axis=-1))

    # Gather the literature atom positions for each residue.
    # r3.Vecs with shape (N, 14)
    lit_positions = r3.vecs_from_tensor(
        utils.batched_gather(
            residue_constants.restype_atom14_rigid_group_positions, aatype))

    # Transform each atom from its local frame to the global frame.
    # r3.Vecs with shape (N, 14)
    pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    mask = utils.batched_gather(residue_constants.restype_atom14_mask, aatype)
    # pred_positions = jax.tree_map(lambda x: x * mask, pred_positions)
    pred_positions *= mask

    return pred_positions

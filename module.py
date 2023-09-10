import numpy as np
import folding
import residue_constants
import common_modules
import utils
from distribution import LayerNorm
import lddt
import quat_affine


def one_hot(x, num_classes, axis=-1, dtype=np.float_):
    indices = x.flatten()
    arr = np.zeros(shape=(len(indices) * num_classes,))
    for i in range(len(indices)):
        if 0 <= indices[i] < num_classes:
            arr[num_classes * i + indices[i]] = 1.
    size = x.shape + (num_classes,)
    return arr.reshape(size)


def log_softmax(x, axis=-1, where=None, initial=None):
    r"""Log-Softmax function.
    Computes the logarithm of the :code:`softmax` function, which rescales
     elements to the range :math:`[-\infty, 0)`.
    .. math ::
        \mathrm{log\_softmax}(x) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
        \right)
    Args:
        x : input array
        axis: the axis or axes along which the :code:`log_softmax` should be
            computed. Either an integer or a tuple of integers.
        where: Elements to include in the :code:`log_softmax`.
        initial: The minimum value used to shift the input array. Must be present
            when :code:`where` is not None.
    """
    x_max = np.max(x, axis, where=where, initial=initial, keepdims=True)
    shifted = x - x_max
    shifted_logsumexp = np.log(np.sum(np.exp(shifted), axis, where=where, keepdims=True))
    return shifted - shifted_logsumexp


def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -np.sum(labels * log_softmax(logits), axis=-1)
    return np.asarray(loss)


def relu(x: int or float or np.ndarray):
    return np.maximum(0, x)


def softplus(x: int or float or np.ndarray):
    return np.logaddexp(x, 0).astype(np.float32)


def log_sigmoid(x):
    r"""Log-sigmoid activation function.
    Computes the element-wise function:
        .. math::
    \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})
    Args:
        x : input array
    """
    return -softplus(-x)


def sigmoid_cross_entropy(logits, labels):
    """Computes sigmoid cross entropy given logits and multiple class labels."""
    log_p = log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
    log_not_p = log_sigmoid(-logits)
    loss = -labels * log_p - (1. - labels) * log_not_p
    return np.asarray(loss)


class AlphaFoldIteration:
    """A single recycling iteration of AlphaFold architecture.
    Computes ensembled (averaged) representations from the provided features.
    These representations are then passed to the various heads
    that have been requested by the configuration file. Each head also returns a
    loss which is combined as a weighted sum to produce the total loss.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
    """

    def __init__(self, name='alphafold_iteration'):
        super().__init__(name=name)

    def __call__(self,
                 ensembled_batch,
                 non_ensembled_batch,
                 is_training,
                 compute_loss=False,
                 ensemble_representations=False,
                 return_representations=False):

        num_ensemble = np.asarray(ensembled_batch['seq_length'].shape[0])

        if not ensemble_representations:
            assert ensembled_batch['seq_length'].shape[0] == 1

        def slice_batch(i):
            b = {k: v[i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b

        # Compute representations for each batch element and average.
        evoformer_module = EmbeddingsAndEvoformer(evoformer_num_block=48, extra_msa_channel=64,
                                                  extra_msa_stack_num_block=4,
                                                  max_relative_feature=32, msa_channel=256, pair_channel=128,
                                                  prev_pos_min_bin=3.25,
                                                  prev_pos_max_bin=20.75, prev_pos_num_bins=15, deterministic=False,
                                                  recycle_features=True,
                                                  recycle_pos=True, seq_channel=384, multimer_mode=False,
                                                  subbatch_size=4, use_remat=False,
                                                  zero_init=False, name='evoformer')
        batch0 = slice_batch(0)
        representations = evoformer_module(batch0, is_training)

        # MSA representations are not ensembled so
        # we don't pass tensor into the loop.
        msa_representation = representations['msa']
        del representations['msa']

        # Average the representations (except MSA) over the batch dimension.
        if ensemble_representations:
            def body(x):
                """Add one element to the representations ensemble."""
                i, current_representations = x
                feats = slice_batch(i)
                representations_update = evoformer_module(feats, is_training)

                new_representations = {}
                for k in current_representations:
                    new_representations[k] = (current_representations[k] + representations_update[k])
                return i + 1, new_representations

            if hk.running_init():
                # When initializing the Haiku module, run one iteration of the
                # while_loop to initialize the Haiku modules used in `body`.
                _, representations = body((1, representations))
            else:
                _, representations = hk.while_loop(
                    lambda x: x[0] < num_ensemble,  # condition
                    body,  # body
                    (1, representations))  # loop variant

            for k in representations:
                if k != 'msa':
                    representations[k] /= num_ensemble.astype(representations[k].dtype)

        representations['msa'] = msa_representation
        batch = batch0  # We are not ensembled from here on.

        heads = {}  # functions collection
        for head_name, head_config in sorted(self.config.heads.items()):
            if not head_config.weight:
                continue  # Do not instantiate zero-weight heads.

            head_factory = {
                'masked_msa': MaskedMsaHead,
                'distogram': DistogramHead,
                'structure_module': folding.StructureModule,
                'predicted_lddt': PredictedLDDTHead,
                'predicted_aligned_error': PredictedAlignedErrorHead,
                'experimentally_resolved': ExperimentallyResolvedHead,
            }[head_name]
            heads[head_name] = (head_config,  # parameters of functions in head_factory
                                head_factory(head_config, self.global_config))  # run functions in head_factory

        total_loss = 0.
        ret = {}
        ret['representations'] = representations

        def loss(module, head_config, ret, name, filter_ret=True):
            if filter_ret:
                value = ret[name]
            else:
                value = ret
            loss_output = module.loss(value, batch)
            ret[name].update(loss_output)
            loss = head_config.weight * ret[name]['loss']
            return loss

        for name, (head_config, module) in heads.items():
            # Skip PredictedLDDTHead and PredictedAlignedErrorHead until
            # StructureModule is executed.
            if name in ('predicted_lddt', 'predicted_aligned_error'):
                continue
            else:
                ret[name] = module(representations, batch, is_training)
                if 'representations' in ret[name]:
                    # Extra representations from the head. Used by the structure module
                    # to provide activations for the PredictedLDDTHead.
                    representations.update(ret[name].pop('representations'))
            if compute_loss:
                total_loss += loss(module, head_config, ret, name)

        if self.config.heads.get('predicted_lddt.weight', 0.0):
            # Add PredictedLDDTHead after StructureModule executes.
            name = 'predicted_lddt'
            # Feed all previous results to give access to structure_module result.
            head_config, module = heads[name]
            ret[name] = module(representations, batch, is_training)
            if compute_loss:
                total_loss += loss(module, head_config, ret, name, filter_ret=False)

        if ('predicted_aligned_error' in self.config.heads
                and self.config.heads.get('predicted_aligned_error.weight', 0.0)):
            # Add PredictedAlignedErrorHead after StructureModule executes.
            name = 'predicted_aligned_error'
            # Feed all previous results to give access to structure_module result.
            head_config, module = heads[name]
            ret[name] = module(representations, batch, is_training)
            if compute_loss:
                total_loss += loss(module, head_config, ret, name, filter_ret=False)

        if compute_loss:
            return ret, total_loss
        else:
            return ret


class AlphaFold:
    """AlphaFold model with recycling.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference"
    """

    def __init__(self, resample_msa_in_recycling=True, num_recycle=3,
                 recycle_pos=True, recycle_features=True, msa_channel=256,
                 pair_channel=128, name='alphafold'):
        super().__init__(name=name)
        self.resample_msa_in_recycling = resample_msa_in_recycling
        self.num_recycle = num_recycle
        self.recycle_pos = recycle_pos
        self.recycle_features = recycle_features
        self.msa_channel = msa_channel
        self.pair_channel = pair_channel

    def __call__(
            self,
            batch,
            is_training,
            compute_loss=False,
            ensemble_representations=False,
            return_representations=False):
        """Run the AlphaFold model.
        Arguments:
            batch: Dictionary with inputs to the AlphaFold model.
            is_training: Whether the system is in training or inference mode.
            compute_loss: Whether to compute losses (requires extra features
                to be present in the batch and knowing the true structure).
            ensemble_representations: Whether to use ensembling of representations.
            return_representations: Whether to also return the intermediate
                representations.
        Returns:
            When compute_loss is True:
                a tuple of loss and output of AlphaFoldIteration.
            When compute_loss is False:
                just output of AlphaFoldIteration.
            The output of AlphaFoldIteration is a nested dictionary containing
            predictions from the various heads.
        """

        impl = AlphaFoldIteration()
        batch_size, num_residues = batch['aatype'].shape

        def get_prev(ret):
            new_prev = {
                'prev_pos': ret['structure_module']['final_atom_positions'],
                'prev_msa_first_row': ret['representations']['msa_first_row'],
                'prev_pair': ret['representations']['pair'],
            }
            return new_prev

        def do_call(prev,
                    recycle_idx,
                    compute_loss=compute_loss):
            if self.resample_msa_in_recycling:  # True
                num_ensemble = batch_size // (self.num_recycle + 1)  # batch_size / 4

                def dynamic_slice_in_dim(operand, start_index, slice_size, axis=0):
                    operand_size = list(operand.shape)
                    if len(slice_size) != len(operand_size):
                        raise ValueError("Length of slice indices must match number of operand dimensions")
                    for i in range(len(operand_size)):
                        if start_index[i] < 0 or start_index[i] >= operand_size[i]:
                            raise ValueError("start index does not match array shape")
                        if slice_size[i] < 0 or slice_size[i] >= operand_size[i]:
                            raise ValueError("slice size does not match array shape")
                    slice_index = []
                    for i in range(len(operand_size)):
                        if start_index[i] + slice_size[i] > operand_size[i]:
                            slice_index.append([i for i in range(operand_size[i] - slice_size[i], operand_size[i])])
                        else:
                            slice_index.append([i for i in range(start_index[i], start_index[i] + slice_size[i])])
                    return operand[np.ix_(*slice_index)]

                def slice_recycle_idx(x):
                    start = recycle_idx * num_ensemble
                    size = num_ensemble
                    return dynamic_slice_in_dim(x, start, size, axis=0)

                ensembled_batch = slice_recycle_idx(batch)
            else:
                num_ensemble = batch_size
                ensembled_batch = batch

            non_ensembled_batch = prev

            return impl(
                ensembled_batch=ensembled_batch,
                non_ensembled_batch=non_ensembled_batch,
                is_training=is_training,
                compute_loss=compute_loss,
                ensemble_representations=ensemble_representations)

        prev = {}
        if self.recycle_pos:
            prev['prev_pos'] = np.zeros(
                [num_residues, residue_constants.atom_type_num, 3])
        if self.recycle_features:
            prev['prev_msa_first_row'] = np.zeros(
                [num_residues, self.msa_channel])
            prev['prev_pair'] = np.zeros(
                [num_residues, num_residues, self.pair_channel])

        if self.num_recycle:  # 3
            if 'num_iter_recycling' in batch:
                # Training time: num_iter_recycling is in batch.
                # The value for each ensemble batch is the same, so arbitrarily taking 0-th.
                num_iter = batch['num_iter_recycling'][0]

                # Add insurance that we will not run more
                # recyclings than the model is configured to run.
                num_iter = np.minimum(num_iter, self.num_recycle)
            else:
                # Eval mode or tests: use the maximum number of iterations.
                num_iter = self.num_recycle

            body = lambda x: (x[0] + 1,  # pylint: disable=g-long-lambda
                              get_prev(do_call(x[1], recycle_idx=x[0],
                                               compute_loss=False)))
            if hk.running_init():
                # When initializing the Haiku module, run one iteration of the
                # while_loop to initialize the Haiku modules used in `body`.
                _, prev = body((0, prev))
            else:
                _, prev = hk.while_loop(
                    lambda x: x[0] < num_iter,
                    body,
                    (0, prev))
        else:
            num_iter = 0

        ret = do_call(prev=prev, recycle_idx=num_iter)
        if compute_loss:
            ret = ret[0], [ret[1]]

        if not return_representations:
            del (ret[0] if compute_loss else ret)['representations']  # pytype: disable=unsupported-operands
        return ret


class MaskedMsaHead:
    """Head to predict MSA at the masked locations.
    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """

    def __init__(self, multimer_mode=False, zero_init=True, num_output=23, name='masked_msa_head'):
        super().__init__(name=name)
        self.zero_init = zero_init

        if multimer_mode:
            self.num_output = len(residue_constants.restypes_with_x_and_gap)
        else:
            self.num_output = num_output

    def __call__(self, representations, batch, is_training):
        """Builds MaskedMsaHead module.
        Arguments:
          representations: Dictionary of representations, must contain:
            * 'msa': MSA representation, shape [N_seq, N_res, c_m].
          batch: Batch, unused.
          is_training: Whether the module is in training mode.
        Returns:
          Dictionary containing:
            * 'logits': logits of shape [N_seq, N_res, N_aatype] with
                (unnormalized) log probabilies of predicted aatype at position.
        """
        del batch
        logits = common_modules.Linear(
            self.num_output,
            initializer=utils.final_init(self.zero_init),
            name='logits')(
            representations['msa'])
        return dict(logits=logits)

    def loss(self, value, batch):
        errors = softmax_cross_entropy(
            labels=one_hot(batch['true_msa'], num_classes=self.num_output),
            logits=value['logits'])
        loss = (np.sum(errors * batch['bert_mask'], axis=(-2, -1)) /
                (1e-8 + np.sum(batch['bert_mask'], axis=(-2, -1))))
        return {'loss': loss}


class DistogramHead:
    """Head to predict a distogram.
    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """

    def __init__(self, num_bins=64, zero_init=True, first_break=2.3125,
                 last_break=21.6875, name='distogram_head'):
        super().__init__(name=name)
        self.num_bins = num_bins
        self.zero_init = zero_init
        self.first_break = first_break
        self.last_break = last_break

    def __call__(self, representations, batch, is_training):
        """Builds DistogramHead module.
        Arguments:
            representations: Dictionary of representations, must contain:
                * 'pair': pair representation, shape [N_res, N_res, c_z].
            batch: Batch, unused.
            is_training: Whether the module is in training mode.
        Returns:
            Dictionary containing:
                * logits: logits for distogram, shape [N_res, N_res, N_bins].
                * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
        """
        half_logits = common_modules.Linear(
            self.num_bins,
            initializer=utils.final_init(self.zero_init),
            name='half_logits')(
            representations['pair'])

        logits = half_logits + np.swapaxes(half_logits, -2, -3)
        breaks = np.linspace(self.first_break, self.last_break, self.num_bins - 1)

        return dict(logits=logits, bin_edges=breaks)

    def loss(self, value, batch):
        return _distogram_log_loss(value['logits'], value['bin_edges'], batch, self.num_bins)


def _distogram_log_loss(logits, bin_edges, batch, num_bins):
    """Log loss of a distogram."""

    assert len(logits.shape) == 3
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']

    assert positions.shape[-1] == 3

    sq_breaks = np.square(bin_edges)

    dist2 = np.sum(
        np.square(
            np.expand_dims(positions, axis=-2) -
            np.expand_dims(positions, axis=-3)),
        axis=-1,
        keepdims=True)

    true_bins = np.sum(dist2 > sq_breaks, axis=-1)

    errors = softmax_cross_entropy(
        labels=one_hot(true_bins, num_bins), logits=logits)

    square_mask = np.expand_dims(mask, axis=-2) * np.expand_dims(mask, axis=-1)

    avg_error = (
            np.sum(errors * square_mask, axis=(-2, -1)) /
            (1e-6 + np.sum(square_mask, axis=(-2, -1))))
    dist2 = dist2[..., 0]
    return dict(loss=avg_error, true_dist=np.sqrt(1e-6 + dist2).astype(np.float32))


class PredictedLDDTHead:
    """Head to predict the per-residue LDDT to be used as a confidence measure.
    Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
    Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
    """

    def __init__(self, num_channels=128, num_bins=50,
                 zero_init=True, filter_by_resolution=True,
                 min_resolution=0.1, max_resolution=3.0, name='predicted_lddt_head'):
        super().__init__(name=name)
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.zero_init = zero_init
        self.filter_by_resolution = filter_by_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def __call__(self, representations, batch, is_training):
        """Builds PredictedLDDTHead module.
        Arguments:
            representations: Dictionary of representations, must contain:
                * 'structure_module': Single representation from the structure module,
                     shape [N_res, c_s].
            batch: Batch, unused.
            is_training: Whether the module is in training mode.
        Returns:
            Dictionary containing:
                * 'logits': logits of shape [N_res, N_bins] with
                    (unnormalized) log probabilies of binned predicted lDDT.
        """
        act = representations['structure_module']

        act = LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name='input_layer_norm')(
            act)

        act = common_modules.Linear(
            self.num_channels,
            initializer='relu',
            name='act_0')(
            act)
        act = relu(act)

        act = common_modules.Linear(
            self.num_channels,
            initializer='relu',
            name='act_1')(
            act)
        act = relu(act)

        logits = common_modules.Linear(
            self.num_bins,
            initializer=utils.final_init(self.zero_init),
            name='logits')(
            act)
        # Shape (batch_size, num_res, num_bins)
        return dict(logits=logits)

    def loss(self, value, batch):
        # Shape (num_res, 37, 3)
        pred_all_atom_pos = value['structure_module']['final_atom_positions']
        # Shape (num_res, 37, 3)
        true_all_atom_pos = batch['all_atom_positions']
        # Shape (num_res, 37)
        all_atom_mask = batch['all_atom_mask']

        # Shape (num_res,)
        lddt_ca = lddt.lddt(
            # Shape (batch_size, num_res, 3)
            predicted_points=pred_all_atom_pos[None, :, 1, :],
            # Shape (batch_size, num_res, 3)
            true_points=true_all_atom_pos[None, :, 1, :],
            # Shape (batch_size, num_res, 1)
            true_points_mask=all_atom_mask[None, :, 1:2].astype(np.float32),
            cutoff=15.,
            per_residue=True)
        # lddt_ca = jax.lax.stop_gradient(lddt_ca)

        bin_index = np.floor(lddt_ca * self.num_bins).astype(np.int32)

        # protect against out of range for lddt_ca == 1
        bin_index = np.minimum(bin_index, self.num_bins - 1)
        lddt_ca_one_hot = one_hot(bin_index, num_classes=self.num_bins)

        # Shape (num_res, num_channel)
        logits = value['predicted_lddt']['logits']
        errors = softmax_cross_entropy(labels=lddt_ca_one_hot, logits=logits)

        # Shape (num_res,)
        mask_ca = all_atom_mask[:, residue_constants.atom_order['CA']]
        mask_ca = mask_ca.astype(np.float32)
        loss = np.sum(errors * mask_ca) / (np.sum(mask_ca) + 1e-8)

        if self.filter_by_resolution:
            # NMR & distillation have resolution = 0
            loss *= ((batch['resolution'] >= self.min_resolution)
                     & (batch['resolution'] <= self.max_resolution)).astype(np.float32)

        output = {'loss': loss}
        return output


class PredictedAlignedErrorHead:
    """Head to predict the distance errors in the backbone alignment frames.
    Can be used to compute predicted TM-Score.
    Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
    """

    def __init__(self, num_bins=64, zero_init=True, max_error_bin=31.,
                 filter_by_resolution=True, min_resolution=0.1, max_resolution=3.0,
                 name='predicted_aligned_error_head'):
        super().__init__(name=name)
        self.num_bins = num_bins
        self.zero_init = zero_init
        self.max_error_bin = max_error_bin
        self.filter_by_resolution = filter_by_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def __call__(self, representations, batch, is_training):
        """Builds PredictedAlignedErrorHead module.
        Arguments:
            representations: Dictionary of representations, must contain:
                * 'pair': pair representation, shape [N_res, N_res, c_z].
            batch: Batch, unused.
            is_training: Whether the module is in training mode.
        Returns:
            Dictionary containing:
                * logits: logits for aligned error, shape [N_res, N_res, N_bins].
                * bin_breaks: array containing bin breaks, shape [N_bins - 1].
        """

        act = representations['pair']

        # Shape (num_res, num_res, num_bins)
        logits = common_modules.Linear(
            self.num_bins,
            initializer=utils.final_init(self.zero_init),
            name='logits')(act)
        # Shape (num_bins,)
        breaks = np.linspace(0., self.max_error_bin, self.num_bins - 1)
        return dict(logits=logits, breaks=breaks)

    def loss(self, value, batch):
        # Shape (num_res, 7)
        predicted_affine = quat_affine.QuatAffine.from_tensor(
            value['structure_module']['final_affines'])
        # Shape (num_res, 7)
        true_affine = quat_affine.QuatAffine.from_tensor(
            batch['backbone_affine_tensor'])
        # Shape (num_res)
        mask = batch['backbone_affine_mask']
        # Shape (num_res, num_res)
        square_mask = mask[:, None] * mask[None, :]
        # (1, num_bins - 1)
        breaks = value['predicted_aligned_error']['breaks']
        # (1, num_bins)
        logits = value['predicted_aligned_error']['logits']

        # Compute the squared error for each alignment.
        def _local_frame_points(affine):
            points = [np.expand_dims(x, axis=-2) for x in affine.translation]
            return affine.invert_point(points, extra_dims=1)

        error_dist2_xyz = [
            np.square(a - b)
            for a, b in zip(_local_frame_points(predicted_affine),
                            _local_frame_points(true_affine))]
        error_dist2 = sum(error_dist2_xyz)
        # Shape (num_res, num_res)
        # First num_res are alignment frames, second num_res are the residues.
        # error_dist2 = jax.lax.stop_gradient(error_dist2)

        sq_breaks = np.square(breaks)
        true_bins = np.sum((error_dist2[..., None] > sq_breaks).astype(np.int32), axis=-1)

        errors = softmax_cross_entropy(labels=one_hot(true_bins, self.num_bins, axis=-1), logits=logits)

        loss = (np.sum(errors * square_mask, axis=(-2, -1)) /
                (1e-8 + np.sum(square_mask, axis=(-2, -1))))

        if self.filter_by_resolution:
            # NMR & distillation have resolution = 0
            loss *= ((batch['resolution'] >= self.min_resolution)
                     & (batch['resolution'] <= self.max_resolution)).astype(np.float32)

        output = {'loss': loss}
        return output


class ExperimentallyResolvedHead:
    """Predicts if an atom is experimentally resolved in a high-res structure.
    Only trained on high-resolution X-ray crystals & cryo-EM.
    Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
    """

    def __init__(self, zero_init=True, filter_by_resolution=True,
                 min_resolution=0.1, max_resolution=3.0, name='experimentally_resolved_head'):
        super().__init__(name=name)
        self.zero_init = zero_init
        self.filter_by_resolution = filter_by_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def __call__(self, representations, batch, is_training):
        """Builds ExperimentallyResolvedHead module.
        Arguments:
            representations: Dictionary of representations, must contain:
                * 'single': Single representation, shape [N_res, c_s].
            batch: Batch, unused.
            is_training: Whether the module is in training mode.
        Returns:
            Dictionary containing:
                * 'logits': logits of shape [N_res, 37],
                    log probability that an atom is resolved in atom37 representation,
                    can be converted to probability by applying sigmoid.
        """
        logits = common_modules.Linear(
            37,  # atom_exists.shape[-1]
            initializer=utils.final_init(self.zero_init),
            name='logits')(representations['single'])
        return dict(logits=logits)

    def loss(self, value, batch):
        logits = value['logits']
        assert len(logits.shape) == 2

        # Does the atom appear in the amino acid?
        atom_exists = batch['atom37_atom_exists']
        # Is the atom resolved in the experiment? Subset of atom_exists,
        # *except for OXT*
        all_atom_mask = batch['all_atom_mask'].astype(np.float32)

        xent = sigmoid_cross_entropy(labels=all_atom_mask, logits=logits)
        loss = np.sum(xent * atom_exists) / (1e-8 + np.sum(atom_exists))

        if self.filter_by_resolution:
            # NMR & distillation examples have resolution = 0.
            loss *= ((batch['resolution'] >= self.min_resolution)
                     & (batch['resolution'] <= self.max_resolution)).astype(np.float32)

        output = {'loss': loss}
        return output

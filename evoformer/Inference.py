class EmbeddingsAndEvoformer(hk.Module):
    """Embeds the input data and runs Evoformer.
    Produces the MSA, single and pair representations.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5-18
    """

    def __init__(self, config, global_config, name='evoformer'):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, batch, is_training, safe_key=None):

        c = self.config
        gc = self.global_config

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        # Embed clustered MSA.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        preprocess_1d = common_modules.Linear(
            c.msa_channel, name='preprocess_1d')(
            batch['target_feat'])

        preprocess_msa = common_modules.Linear(
            c.msa_channel, name='preprocess_msa')(
            batch['msa_feat'])

        msa_activations = jnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa

        left_single = common_modules.Linear(
            c.pair_channel, name='left_single')(
            batch['target_feat'])
        right_single = common_modules.Linear(
            c.pair_channel, name='right_single')(
            batch['target_feat'])
        pair_activations = left_single[:, None] + right_single[None]
        mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if c.recycle_pos:
            prev_pseudo_beta = pseudo_beta_fn(
                batch['aatype'], batch['prev_pos'], None)
            dgram = dgram_from_positions(prev_pseudo_beta, **self.config.prev_pos)
            pair_activations += common_modules.Linear(
                c.pair_channel, name='prev_pos_linear')(
                dgram)

        if c.recycle_features:
            prev_msa_first_row = hk.LayerNorm(
                axis=[-1],
                create_scale=True,
                create_offset=True,
                name='prev_msa_first_row_norm')(
                batch['prev_msa_first_row'])
            msa_activations = msa_activations.at[0].add(prev_msa_first_row)

            pair_activations += hk.LayerNorm(
                axis=[-1],
                create_scale=True,
                create_offset=True,
                name='prev_pair_norm')(
                batch['prev_pair'])

        # Relative position encoding.
        # Jumper et al. (2021) Suppl. Alg. 4 "relpos"
        # Jumper et al. (2021) Suppl. Alg. 5 "one_hot"
        if c.max_relative_feature:
            # Add one-hot-encoded clipped residue distances to the pair activations.
            pos = batch['residue_index']
            offset = pos[:, None] - pos[None, :]
            rel_pos = jax.nn.one_hot(
                jnp.clip(
                    offset + c.max_relative_feature,
                    a_min=0,
                    a_max=2 * c.max_relative_feature),
                2 * c.max_relative_feature + 1)
            pair_activations += common_modules.Linear(
                c.pair_channel, name='pair_activiations')(
                rel_pos)

        # Embed templates into the pair activations.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-13
        if c.template.enabled:
            template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
            template_pair_representation = TemplateEmbedding(c.template, gc)(
                pair_activations,
                template_batch,
                mask_2d,
                is_training=is_training)

            pair_activations += template_pair_representation

        # Embed extra MSA features.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
        extra_msa_feat = create_extra_msa_feature(batch)
        extra_msa_activations = common_modules.Linear(
            c.extra_msa_channel,
            name='extra_msa_activations')(
            extra_msa_feat)

        # Extra MSA Stack.
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        extra_msa_stack_input = {
            'msa': extra_msa_activations,
            'pair': pair_activations,
        }

        extra_msa_stack_iteration = EvoformerIteration(
            c.evoformer, gc, is_extra_msa=True, name='extra_msa_stack')

        def extra_msa_stack_fn(x):
            act, safe_key = x
            safe_key, safe_subkey = safe_key.split()
            extra_evoformer_output = extra_msa_stack_iteration(
                activations=act,
                masks={
                    'msa': batch['extra_msa_mask'],
                    'pair': mask_2d
                },
                is_training=is_training,
                safe_key=safe_subkey)
            return (extra_evoformer_output, safe_key)

        if gc.use_remat:
            extra_msa_stack_fn = hk.remat(extra_msa_stack_fn)

        extra_msa_stack = layer_stack.layer_stack(
            c.extra_msa_stack_num_block)(
            extra_msa_stack_fn)
        extra_msa_output, safe_key = extra_msa_stack(
            (extra_msa_stack_input, safe_key))

        pair_activations = extra_msa_output['pair']

        evoformer_input = {
            'msa': msa_activations,
            'pair': pair_activations,
        }

        evoformer_masks = {'msa': batch['msa_mask'], 'pair': mask_2d}

        # Append num_templ rows to msa_activations with template embeddings.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 7-8
        if c.template.enabled and c.template.embed_torsion_angles:
            num_templ, num_res = batch['template_aatype'].shape

            # Embed the templates aatypes.
            aatype_one_hot = jax.nn.one_hot(batch['template_aatype'], 22, axis=-1)

            # Embed the templates aatype, torsion angles and masks.
            # Shape (templates, residues, msa_channels)
            ret = all_atom.atom37_to_torsion_angles(
                aatype=batch['template_aatype'],
                all_atom_pos=batch['template_all_atom_positions'],
                all_atom_mask=batch['template_all_atom_masks'],
                # Ensure consistent behaviour during testing:
                placeholder_for_undefined=not gc.zero_init)

            template_features = jnp.concatenate([
                aatype_one_hot,
                jnp.reshape(
                    ret['torsion_angles_sin_cos'], [num_templ, num_res, 14]),
                jnp.reshape(
                    ret['alt_torsion_angles_sin_cos'], [num_templ, num_res, 14]),
                ret['torsion_angles_mask']], axis=-1)

            template_activations = common_modules.Linear(
                c.msa_channel,
                initializer='relu',
                name='template_single_embedding')(
                template_features)
            template_activations = jax.nn.relu(template_activations)
            template_activations = common_modules.Linear(
                c.msa_channel,
                initializer='relu',
                name='template_projection')(
                template_activations)

            # Concatenate the templates to the msa.
            evoformer_input['msa'] = jnp.concatenate(
                [evoformer_input['msa'], template_activations], axis=0)
            # Concatenate templates masks to the msa masks.
            # Use mask from the psi angle, as it only depends on the backbone atoms
            # from a single residue.
            torsion_angle_mask = ret['torsion_angles_mask'][:, :, 2]
            torsion_angle_mask = torsion_angle_mask.astype(
                evoformer_masks['msa'].dtype)
            evoformer_masks['msa'] = jnp.concatenate(
                [evoformer_masks['msa'], torsion_angle_mask], axis=0)

        # Main trunk of the network
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        evoformer_iteration = EvoformerIteration(
            c.evoformer, gc, is_extra_msa=False, name='evoformer_iteration')

        def evoformer_fn(x):
            act, safe_key = x
            safe_key, safe_subkey = safe_key.split()
            evoformer_output = evoformer_iteration(
                activations=act,
                masks=evoformer_masks,
                is_training=is_training,
                safe_key=safe_subkey)
            return (evoformer_output, safe_key)

        if gc.use_remat:
            evoformer_fn = hk.remat(evoformer_fn)

        evoformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(
            evoformer_fn)
        evoformer_output, safe_key = evoformer_stack(
            (evoformer_input, safe_key))

        msa_activations = evoformer_output['msa']
        pair_activations = evoformer_output['pair']

        single_activations = common_modules.Linear(
            c.seq_channel, name='single_activations')(
            msa_activations[0])

        num_sequences = batch['msa_feat'].shape[0]
        output = {
            'single': single_activations,
            'pair': pair_activations,
            # Crop away template rows such that they are not used in MaskedMsaHead.
            'msa': msa_activations[:num_sequences, :, :],
            'msa_first_row': msa_activations[0],
        }

        return output


class AlphaFoldIteration(hk.Module):
    """A single recycling iteration of AlphaFold architecture.
    Computes ensembled (averaged) representations from the provided features.
    These representations are then passed to the various heads
    that have been requested by the configuration file. Each head also returns a
    loss which is combined as a weighted sum to produce the total loss.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
    """

    def __init__(self, config, global_config, name='alphafold_iteration'):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self,
                 ensembled_batch,
                 non_ensembled_batch,
                 is_training,
                 compute_loss=False,
                 ensemble_representations=False,
                 return_representations=False):

        num_ensemble = jnp.asarray(ensembled_batch['seq_length'].shape[0])

        if not ensemble_representations:
            assert ensembled_batch['seq_length'].shape[0] == 1

        def slice_batch(i):
            b = {k: v[i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b

        # Compute representations for each batch element and average.
        evoformer_module = EmbeddingsAndEvoformer(
            self.config.embeddings_and_evoformer, self.global_config)
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
                representations_update = evoformer_module(
                    feats, is_training)

                new_representations = {}
                for k in current_representations:
                    new_representations[k] = (
                            current_representations[k] + representations_update[k])
                return i + 1, new_representations

            if hk.running_init():
                # When initializing the Haiku module, run one iteration of the
                # while_loop to initialize the Haiku modules used in `body`.
                _, representations = body((1, representations))
            else:
                _, representations = hk.while_loop(
                    lambda x: x[0] < num_ensemble,
                    body,
                    (1, representations))

            for k in representations:
                if k != 'msa':
                    representations[k] /= num_ensemble.astype(representations[k].dtype)

        representations['msa'] = msa_representation
        batch = batch0  # We are not ensembled from here on.

        heads = {}
        for head_name, head_config in sorted(self.config.heads.items()):
            if not head_config.weight:
                continue  # Do not instantiate zero-weight heads.

            head_factory = {
                'masked_msa': MaskedMsaHead,
                'distogram': DistogramHead,
                'structure_module': functools.partial(
                    folding.StructureModule, compute_loss=compute_loss),
                'predicted_lddt': PredictedLDDTHead,
                'predicted_aligned_error': PredictedAlignedErrorHead,
                'experimentally_resolved': ExperimentallyResolvedHead,
            }[head_name]
            heads[head_name] = (head_config,
                                head_factory(head_config, self.global_config))

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


class AlphaFold(hk.Module):
    """AlphaFold model with recycling.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference"
    """

    def __init__(self, config, name='alphafold'):
        super().__init__(name=name)
        self.config = config
        self.global_config = config.global_config

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

        impl = AlphaFoldIteration(self.config, self.global_config)
        batch_size, num_residues = batch['aatype'].shape

        def get_prev(ret):
            new_prev = {
                'prev_pos':
                    ret['structure_module']['final_atom_positions'],
                'prev_msa_first_row': ret['representations']['msa_first_row'],
                'prev_pair': ret['representations']['pair'],
            }
            return jax.tree_map(jax.lax.stop_gradient, new_prev)

        def do_call(prev,
                    recycle_idx,
                    compute_loss=compute_loss):
            if self.config.resample_msa_in_recycling:
                num_ensemble = batch_size // (self.config.num_recycle + 1)

                def slice_recycle_idx(x):
                    start = recycle_idx * num_ensemble
                    size = num_ensemble
                    return jax.lax.dynamic_slice_in_dim(x, start, size, axis=0)

                ensembled_batch = jax.tree_map(slice_recycle_idx, batch)
            else:
                num_ensemble = batch_size
                ensembled_batch = batch

            non_ensembled_batch = jax.tree_map(lambda x: x, prev)

            return impl(
                ensembled_batch=ensembled_batch,
                non_ensembled_batch=non_ensembled_batch,
                is_training=is_training,
                compute_loss=compute_loss,
                ensemble_representations=ensemble_representations)

        prev = {}
        emb_config = self.config.embeddings_and_evoformer
        if emb_config.recycle_pos:
            prev['prev_pos'] = jnp.zeros(
                [num_residues, residue_constants.atom_type_num, 3])
        if emb_config.recycle_features:
            prev['prev_msa_first_row'] = jnp.zeros(
                [num_residues, emb_config.msa_channel])
            prev['prev_pair'] = jnp.zeros(
                [num_residues, num_residues, emb_config.pair_channel])

        if self.config.num_recycle:
            if 'num_iter_recycling' in batch:
                # Training time: num_iter_recycling is in batch.
                # The value for each ensemble batch is the same, so arbitrarily taking
                # 0-th.
                num_iter = batch['num_iter_recycling'][0]

                # Add insurance that we will not run more
                # recyclings than the model is configured to run.
                num_iter = jnp.minimum(num_iter, self.config.num_recycle)
            else:
                # Eval mode or tests: use the maximum number of iterations.
                num_iter = self.config.num_recycle

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

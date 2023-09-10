import numpy as np
from VarianceScaling import glorot_uniform


class Attention:
    """Multihead attention."""

    def __init__(self, output_dim, q_data, m_data, gating, nonbatched_bias=None, zero_init=True,
                 num_head=8, name='attention'):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.q_data = q_data
        self.m_data = m_data
        self.gating = gating
        self.nonbatched_bias = nonbatched_bias
        self.zero_init = zero_init
        self.num_head = num_head

    def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
        """Builds Attention module.
        Arguments:
            q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
            m_data: A tensor of memories from which the keys and values are
                projected, shape [batch_size, N_keys, m_channels].
            bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
            nonbatched_bias: Shared bias, shape [N_queries, N_keys].
        Returns:
            A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        # Sensible default for when the config keys are missing

        key_dim = int(self.q_data.shape[-1])
        value_dim = int(self.m_data.shape[-1])
        assert key_dim % self.num_head == 0
        assert value_dim % self.num_head == 0
        key_dim = key_dim // self.num_head
        value_dim = value_dim // self.num_head

        q_weights = glorot_uniform(shape=(self.q_data.shape[-1], self.num_head, key_dim))
        k_weights = glorot_uniform(shape=(self.m_data.shape[-1], self.num_head, key_dim))
        v_weights = glorot_uniform(shape=(self.m_data.shape[-1], self.num_head, key_dim))
        q = np.einsum('bqa,ahc->bqhc', self.q_data, np.array([q_weights])) * key_dim ** (-0.5)
        k = np.einsum('bka,ahc->bkhc', self.m_data, np.array([k_weights]))
        v = np.einsum('bka,ahc->bkhc', self.m_data, np.array([v_weights]))
        logits = np.einsum('bqhc,bkhc->bhqk', q, k) + bias

        if self.nonbatched_bias is not None:
            logits += np.expand_dims(self.nonbatched_bias, axis=0)
        weights = np.exp(logits) / np.sum(np.exp(logits))
        weighted_avg = np.einsum('bhqk,bkhc->bqhc', weights, v)

        if self.gating:
            gating_weights = np.zeros(shape=(self.q_data.shape[-1], self.num_head, value_dim))
            gating_bias = np.ones(shape=(self.num_head, value_dim))
            gate_values = np.einsum('bqc, chv->bqhv', self.q_data, gating_weights) + gating_bias
            gate_values = 1 / (1 + np.exp(-gate_values))
            weighted_avg *= gate_values

        if self.zero_init:
            o_weights = np.zeros(shape=(self.num_head, value_dim, self.output_dim))
        else:
            o_weights = glorot_uniform(shape=(self.num_head, value_dim, self.output_dim))
        o_bias = np.zeros(shape=(self.output_dim,))
        output = np.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

        return output

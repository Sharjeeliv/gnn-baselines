from . import tf, np
import math
import re
import warnings


class SparseDropout(tf.keras.layers.Layer):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    @tf.function
    def call(self, input: tf.sparse.SparseTensor):
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(input.values.shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(input, dropout_mask)
        return pre_out / keep_prob


class SparseDense(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_bias=False, activation=None,
                 kernel_regularizer=None):
        super().__init__()
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.output_dim],
            regularizer=self.kernel_regularizer
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.output_dim],
                initializer=tf.zeros_initializer
            )
        super().build(input_shape)

    @tf.function
    def call(self, input: tf.sparse.SparseTensor):
        if not isinstance(input, tf.SparseTensor):
            input = tf.sparse.from_dense(input)
        if input.shape[-1] != self.kernel.shape[0]:
            # Transpose self.kernel if inner dimensions don't match
            self.kernel = tf.transpose(self.kernel)
        input = tf.sparse.sparse_dense_matmul(input, self.kernel)
        if self.use_bias:
            input += self.bias
        if self.activation:
            input = self.activation(input)
        return input

class GCNLayer(tf.keras.layers.Layer):
    SIGNATURE = ["adjhops", "inputs"]

    def __init__(self, hops=None):
        super().__init__()
        self.hops = hops
        self.cpu_large_spmatmul = False

    @tf.function
    def sparse_dense_matmul(self, sp_a, b, ind: str):
        # ind argument is to force tensorflow to retrace for different hops
        if (sp_a.values.shape[0] * b.shape[1] > (2 ** 31) and
                len(tf.config.experimental.list_physical_devices('GPU')) > 0):
            numSplits = (sp_a.values.shape[0] * b.shape[1] // 2 ** 31) + 1
            splitSizes = np.arange(
                b.shape[1] + numSplits - 1, b.shape[1] - 1, -1) // numSplits
            print(
                f"Splitting tensor to {splitSizes} allow sparse tensor multiplication...")
            assert sum(splitSizes) == b.shape[1]
            b_splits = tf.split(b, splitSizes, axis=-1)
            return tf.concat([tf.sparse.sparse_dense_matmul(sp_a, x) for x in b_splits], axis=-1)
        else:
            return tf.sparse.sparse_dense_matmul(sp_a, b)

    @tf.function
    def call(self, adjhops, inputs):
        return tf.stack([self.sparse_dense_matmul(x, inputs, str(ind)) for ind, x in enumerate(adjhops)
                         if (self.hops is None or ind in self.hops)], axis=-2)

class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, multi_head=4, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.multi_head = multi_head

        # Define multi-head weights
        self.W = [self.add_weight(shape=(units, units), initializer='glorot_uniform', trainable=True) for _ in range(multi_head)]
        self.attention_kernels = [self.add_weight(shape=(2 * units, 1), initializer='glorot_uniform', trainable=True) for _ in range(multi_head)]

    def call(self, adj, inputs, training=False):
        N = inputs.shape[0]
        outputs = []
        
        for h in range(self.multi_head):
            # Linear transformation for this head
            transformed_features = tf.matmul(inputs, self.W[h])

            # Compute attention scores with sparse adjacency
            expanded_features_row = tf.tile(tf.expand_dims(transformed_features, 0), [N, 1, 1])
            expanded_features_col = tf.tile(tf.expand_dims(transformed_features, 1), [1, N, 1])
            concat_features = tf.concat([expanded_features_row, expanded_features_col], axis=-1)

            e = tf.nn.leaky_relu(tf.matmul(concat_features, self.attention_kernels[h]))
            e = tf.squeeze(e, -1)

            # Apply sparse masking for adjacency
            e = tf.where(adj > 0, e, -1e9 * tf.ones_like(e))
            attention_coefficients = tf.nn.softmax(e, axis=1)

            # Dropout for regularization
            attention_coefficients = tf.nn.dropout(attention_coefficients, rate=self.dropout_rate)

            # Aggregate with attention weights
            head_output = tf.matmul(attention_coefficients, transformed_features)
            outputs.append(head_output)
        
        # Concatenate heads
        output = tf.concat(outputs, axis=-1)
        return output

class ConcatLayer(tf.keras.layers.Layer):
    def __init__(self, tags, axis=-1, addInputs=True):
        super().__init__()
        self.tags = tags
        self.axis = axis
        self.addInputs = addInputs

    def call(self, *args, **kwargs):
        selected = [value for name, value in kwargs.items()
                    if name in self.tags]
        if self.addInputs:
            return tf.concat(list(args) + selected, self.axis)
        else:
            return tf.concat(selected, self.axis)

class SumLayer(tf.keras.layers.Layer):
    def __init__(self, dim=-2):
        super().__init__()
        self.dim = dim

    def call(self, inputs):
        return tf.reduce_sum(inputs, self.dim)


class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, loadTag, sliceObj, **kwargs):
        super().__init__()
        self.tag = loadTag
        self.sliceObj = sliceObj

    def call(self, inputs, **kwargs):
        if self.tag:
            inputs = kwargs[self.tag]
        return inputs[:, self.sliceObj]

experimentalDict = {
}

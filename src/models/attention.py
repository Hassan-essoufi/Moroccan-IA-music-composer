import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention with causal masking for sequential generation.
    Includes dropout on attention weights.
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout_rate

        self.qkv_dense = layers.Dense(embed_dim * 3)
        self.output_dense = layers.Dense(embed_dim)

        self.attn_dropout = layers.Dropout(dropout_rate)
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def _split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        x = tf.reshape(
            x, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _causal_mask(self, seq_len, dtype):
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=dtype), -1, 0
        )
        mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
        return mask

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # QKV projection
        qkv = self.qkv_dense(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        # Split heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale

        # Causal mask
        mask = self._causal_mask(seq_len, scores.dtype)
        scores = scores * mask + (1.0 - mask) * -1e9

        # Attention weights
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        attention = tf.matmul(weights, v)

        # Merge heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(
            attention, (batch_size, seq_len, self.embed_dim)
        )

        # Final projection
        return self.output_dense(attention)

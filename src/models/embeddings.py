import math
import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class TokenEmbedding(layers.Layer):
    """
    Token embedding.
    """

    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=False
        )

        self.positional_encoding = self._build_positional_encoding(
            max_seq_len, embed_dim
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_seq_len": self.max_seq_len,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _build_positional_encoding(self, max_len, embed_dim):
        positions = tf.cast(tf.range(max_len)[:, tf.newaxis], tf.float32)  
        dims = tf.cast(tf.range(embed_dim)[tf.newaxis, :], tf.float32)         
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(embed_dim, tf.float32))
        angle_rads = positions * angle_rates 

        # Apply sin to even indices; cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        # Interleave sin et cos
        pe_even_odd = tf.stack([sines, cosines], axis=-1)  
        pe_flat = tf.reshape(pe_even_odd, (max_len, embed_dim)) 

        # Add batch dimension
        pe = tf.expand_dims(pe_flat, axis=0) 
        return pe

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.token_embedding(x)
        x = x + self.positional_encoding[:, :seq_len, :]    
        return x 
import tensorflow as tf
from tensorflow.keras import layers, Model
from src.models.embeddings import TokenEmbedding
from src.models.attention import MultiHeadSelfAttention


@tf.keras.utils.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    """
    Single Transformer decoder block.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate=dropout)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training=False):
        attn_out = self.attention(x)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(x + ffn_out)

@tf.keras.utils.register_keras_serializable()
class TransformerDecoder(Model):
    """
    Autoregressive Transformer decoder for symbolic music generation.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dropout,**kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len
        )

        self.blocks = [
            TransformerDecoderBlock(
                embed_dim, num_heads, ff_dim, dropout
            )
            for _ in range(num_layers)
        ]

        self.output_layer = layers.Dense(vocab_size)
        
    def get_config(self):
        """Nécessaire pour la sérialisation"""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Nécessaire pour la désérialisation"""
        return cls(**config)


    def call(self, x, training=False):
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, training=training)

        return self.output_layer(x)

import tensorflow as tf
@tf.keras.utils.register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule.
    """
    def __init__(self, embed_dim, warmup_steps=4000):
        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(tf.cast(self.embed_dim, tf.float32)) * tf.math.minimum(arg1, arg2)
        
    def get_config(self):
        return {
            "embed_dim": self.embed_dim,
            "warmup_steps": self.warmup_steps
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_optimizer(scheduler, weight_decay=0.0):
    """
    Build AdamW optimizer with custom learning rate schedule.
    
    """    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=scheduler,
        weight_decay=weight_decay
    )
    return optimizer

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """
    Ignore padding tokens (0) in loss computation.
    """
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask

    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)
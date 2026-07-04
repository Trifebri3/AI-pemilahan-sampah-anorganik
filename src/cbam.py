import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    """
    Channel Attention Module (CAM)
    Mengekstrak hubungan antar-channel dengan Global Average Pooling (GAP) 
    dan Global Max Pooling (GMP) yang dilewatkan ke shared MLP.
    """
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.shared_mlp_1 = layers.Dense(
            self.channels // self.reduction_ratio,
            activation='relu',
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.shared_mlp_2 = layers.Dense(
            self.channels,
            use_bias=False,
            kernel_initializer='he_normal'
        )
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Global Average Pooling
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, self.channels))(avg_pool)
        avg_pool = self.shared_mlp_1(avg_pool)
        avg_pool = self.shared_mlp_2(avg_pool)
        
        # Global Max Pooling
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        max_pool = layers.Reshape((1, 1, self.channels))(max_pool)
        max_pool = self.shared_mlp_1(max_pool)
        max_pool = self.shared_mlp_2(max_pool)
        
        # Penjumlahan kedua representasi
        attention = layers.Add()([avg_pool, max_pool])
        attention = layers.Activation('sigmoid')(attention)
        
        # Mengalikan input dengan attention map
        return layers.Multiply()([inputs, attention])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config

class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module (SAM)
    Mengekstrak fitur spasial dari input dengan mengaplikasikan Average 
    dan Max Pooling sepanjang channel, lalu melakukan konvolusi 2D.
    """
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv2d = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            use_bias=False,
            kernel_initializer='he_normal'
        )
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Average pooling sepanjang axis channel
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # Max pooling sepanjang axis channel
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Konkatenasi rata-rata dan maksimum
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        
        # Operasi konvolusi spasial
        attention = self.conv2d(concat)
        
        # Mengalikan input dengan spatial attention map
        return layers.Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config

def cbam_block(inputs, reduction_ratio=16, kernel_size=7):
    """
    CBAM Block: menggabungkan Channel Attention dan Spatial Attention secara sekuensial.
    """
    channels = inputs.shape[-1]
    # 1. Terapkan Channel Attention
    x = ChannelAttention(channels, reduction_ratio)(inputs)
    # 2. Terapkan Spatial Attention
    x = SpatialAttention(kernel_size)(x)
    return x

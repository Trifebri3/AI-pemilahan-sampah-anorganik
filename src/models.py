import tensorflow as tf
from tensorflow.keras import layers, Model
from src.cbam import cbam_block

def build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=5):
    """
    Baseline Model: MobileNetV2 dengan transfer learning.
    Sangat ringan, efisien, dan cocok untuk perangkat Edge / IoT.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Bekukan layer dasar untuk transfer learning
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    
    # Pra-pemrosesan khusus MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="MobileNetV2_Baseline")
    return model

def build_efficientnet_b0(input_shape=(224, 224, 3), num_classes=5):
    """
    SOTA Model: EfficientNetB0 dengan transfer learning.
    Memiliki keseimbangan akurasi tinggi dan ukuran model yang relatif kecil.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    
    # Catatan: EfficientNetB0 memiliki layer rescale terintegrasi, 
    # namun kita tetap melampirkan inputs secara bersih.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="EfficientNetB0_SOTA")
    return model

def build_efficientnet_b0_cbam(input_shape=(224, 224, 3), num_classes=5):
    """
    Innovative Model: EfficientNetB0 + CBAM Attention.
    Mengintegrasikan modul perhatian CBAM di akhir ekstraksi fitur 
    untuk menonjolkan area citra yang paling relevan secara spasial dan channel.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    
    # Feature extractor
    x = base_model(inputs, training=False)  # Output shape: (7, 7, 1280)
    
    # Terapkan modul atensi CBAM
    x = cbam_block(x, reduction_ratio=16, kernel_size=7)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="EfficientNetB0_CBAM_Innovative")
    return model

if __name__ == "__main__":
    # Test model summaries
    m1 = build_mobilenet_v2()
    m1.summary()
    m2 = build_efficientnet_b0()
    m2.summary()
    m3 = build_efficientnet_b0_cbam()
    m3.summary()

import os
import tensorflow as tf
from src.config import MODEL_SAVE_DIR
from src.cbam import ChannelAttention, SpatialAttention

def convert_to_tflite(model_name):
    """
    Mengonversi model Keras (.keras) ke format TensorFlow Lite (.tflite).
    Menghasilkan versi float32 standar dan versi float16 teroptimasi.
    """
    keras_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.keras")
    tflite_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.tflite")
    tflite_opt_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best_quant16.tflite")
    
    if not os.path.exists(keras_path):
        print(f"Error: Model file {keras_path} not found.")
        return
        
    print(f"\nConverting {model_name} to TFLite...")
    
    # Registrasi custom objects CBAM saat memuat model
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention
    }
    
    # 1. Muat model Keras
    model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
    
    # 2. Konversi Standar (Float32)
    print("Generating standard Float32 TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Izinkan custom operations jika diperlukan
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # default ops
        tf.lite.OpsSet.SELECT_TF_OPS   # fallback ops
    ]
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Standard TFLite model saved at {tflite_path} ({os.path.getsize(tflite_path) / (1024*1024):.2f} MB)")
    
    # 3. Konversi dengan Optimasi Quantization (Float16)
    print("Generating optimized Float16 TFLite model...")
    converter_opt = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_opt.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_opt.target_spec.supported_types = [tf.float16]
    converter_opt.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_opt_model = converter_opt.convert()
    
    with open(tflite_opt_path, "wb") as f:
        f.write(tflite_opt_model)
    print(f"Quantized FP16 TFLite model saved at {tflite_opt_path} ({os.path.getsize(tflite_opt_path) / (1024*1024):.2f} MB)")

def main():
    models = ["mobilenet_v2", "efficientnet_b0", "efficientnet_b0_cbam"]
    for m in models:
        try:
            convert_to_tflite(m)
        except Exception as e:
            print(f"Failed to convert {m}: {str(e)}")

if __name__ == "__main__":
    main()

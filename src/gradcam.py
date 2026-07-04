import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from src.config import MODEL_SAVE_DIR, VISUALIZATION_DIR, CLASSES
from src.cbam import ChannelAttention, SpatialAttention

def get_gradcam_model(model, model_name):
    """
    Membangun model gradcam khusus yang mengembalikan output dari 
    layer konvolusi terakhir dan output prediksi kelas.
    """
    if "mobilenet_v2" in model_name:
        base_model = model.layers[1]
        target_layer = base_model.get_layer("out_relu")
        grad_base_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[target_layer.output, base_model.output]
        )
        return grad_base_model, "mobilenet_v2"
        
    elif "efficientnet_b0_cbam" in model_name:
        # Pada model CBAM, target kita adalah layer SpatialAttention ('spatial_attention')
        # yang terletak pada model luar (bukan di dalam backbone)
        target_layer = None
        for layer in model.layers:
            if "spatial_attention" in layer.name:
                target_layer = layer
                break
                
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
        return grad_model, "efficientnet_b0_cbam"
        
    elif "efficientnet_b0" in model_name:
        base_model = model.layers[1]
        target_layer = base_model.get_layer("top_activation")
        grad_base_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[target_layer.output, base_model.output]
        )
        return grad_base_model, "efficientnet_b0"
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def generate_gradcam(img_path, model, model_name, target_size=(224, 224)):
    """
    Menghasilkan heatmap Grad-CAM dan menumpukkannya ke citra asli.
    """
    # 1. Load dan preprocess gambar
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
    
    # 2. Dapatkan model Grad-CAM
    grad_model, mode = get_gradcam_model(model, model_name)
    
    # 3. Hitung gradien dengan GradientTape
    with tf.GradientTape() as tape:
        if mode == "mobilenet_v2":
            # MobileNetV2 has preprocessing
            x_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            conv_outputs, base_features = grad_model(x_preprocessed)
            # Tape needs to track operations through outer layers
            x = base_features
            for layer in model.layers[2:]:
                x = layer(x)
            preds = x
            
        elif mode == "efficientnet_b0":
            conv_outputs, base_features = grad_model(img_array)
            # Tape needs to track operations through outer layers
            x = base_features
            for layer in model.layers[2:]:
                x = layer(x)
            preds = x
            
        elif mode == "efficientnet_b0_cbam":
            conv_outputs, preds = grad_model(img_array)
            
        class_idx = np.argmax(preds[0])
        loss_value = preds[:, class_idx]
        
    # Ambil gradien dari loss terhadap output conv layer
    grads = tape.gradient(loss_value, conv_outputs)
    
    # Lakukan global average pooling pada gradien
    # shape dari conv_outputs: (1, H, W, C)
    # shape dari grads: (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Kalikan tiap channel pada feature map dengan bobot gradiennya
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Terapkan ReLU untuk mempertahankan fitur positif saja
    heatmap = tf.maximum(heatmap, 0.0)
    
    # Normalisasi ke [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val == 0.0:
        max_val = 1e-10
    heatmap = heatmap / max_val
    heatmap = heatmap.numpy()
    
    # 4. Resize heatmap ke ukuran asli gambar
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = np.uint8(255 * heatmap_resized)
    
    # Terapkan color map JET
    heatmap_colormap = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # Superimpose heatmap ke gambar asli
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colormap, 0.4, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb, heatmap_resized, CLASSES[class_idx]

def save_gradcam_comparison(img_path, models, output_name):
    """
    Menghasilkan perbandingan visual Grad-CAM untuk ketiga model side-by-side.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Tampilkan Gambar Asli
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    model_keys = ["mobilenet_v2", "efficientnet_b0", "efficientnet_b0_cbam"]
    titles = [
        "MobileNetV2 (Baseline)", 
        "EfficientNetB0 (SOTA)", 
        "EfficientNetB0 + CBAM (Innovative)"
    ]
    
    for i, (key, title) in enumerate(zip(model_keys, titles), start=1):
        if key in models:
            try:
                cam, _, pred_class = generate_gradcam(img_path, models[key], key)
                axes[i].imshow(cam)
                axes[i].set_title(f"{title}\nPred: {pred_class}")
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center')
                axes[i].set_title(title)
        else:
            axes[i].text(0.5, 0.5, "Model Not Loaded", ha='center', va='center')
            axes[i].set_title(title)
        axes[i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATION_DIR, f"gradcam_comparison_{output_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Grad-CAM comparison saved to {save_path}")

def run_gradcam_on_samples():
    """
    Memilih beberapa gambar sampel dari dataset_split/test dan menjalankan Grad-CAM.
    """
    # Muat model
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention
    }
    
    models = {}
    for name in ["mobilenet_v2", "efficientnet_b0", "efficientnet_b0_cbam"]:
        path = os.path.join(MODEL_SAVE_DIR, f"{name}_best.keras")
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path, custom_objects=custom_objects)
            
    # Pilih sampel gambar dari tiap kategori di folder test
    test_dir = "dataset_split/test"
    if not os.path.exists(test_dir):
        print(f"Error: test directory {test_dir} not found. Please split dataset first.")
        return
        
    for cls in CLASSES:
        cls_dir = os.path.join(test_dir, cls)
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                # Ambil gambar pertama sebagai perwakilan
                sample_img_path = os.path.join(cls_dir, files[0])
                print(f"Generating Grad-CAM for class {cls} using {files[0]}...")
                save_gradcam_comparison(sample_img_path, models, cls.lower())

if __name__ == "__main__":
    run_gradcam_on_samples()

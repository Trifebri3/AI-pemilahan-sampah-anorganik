import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, auc
)
from src.config import (
    MODEL_SAVE_DIR, HISTORY_SAVE_DIR, VISUALIZATION_DIR, CLASSES, NUM_CLASSES
)
from src.data_preprocessing import get_data_pipelines
from src.cbam import ChannelAttention, SpatialAttention

def load_all_models():
    """
    Memuat ketiga model dari penyimpanan lokal dengan registrasi custom objects CBAM.
    """
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention
    }
    
    models = {}
    model_files = {
        "mobilenet_v2": "mobilenet_v2_best.keras",
        "efficientnet_b0": "efficientnet_b0_best.keras",
        "efficientnet_b0_cbam": "efficientnet_b0_cbam_best.keras"
    }
    
    for key, name in model_files.items():
        path = os.path.join(MODEL_SAVE_DIR, name)
        if os.path.exists(path):
            print(f"Loading {name}...")
            models[key] = tf.keras.models.load_model(path, custom_objects=custom_objects)
        else:
            print(f"Warning: {name} not found at {path}")
            
    return models

def measure_inference_time(model, test_ds):
    """
    Mengukur rata-rata waktu inferensi per gambar (dalam milidetik).
    """
    # Ambil satu batch gambar
    for images, _ in test_ds.take(1):
        sample_images = images[:10]  # ambil 10 sampel gambar
        break
        
    # Warm-up run
    _ = model.predict(sample_images, verbose=0)
    
    # Measure time
    start_time = time.time()
    iterations = 5
    for _ in range(iterations):
        _ = model.predict(sample_images, verbose=0)
    total_time = time.time() - start_time
    
    avg_inference_time_ms = (total_time / (iterations * len(sample_images))) * 1000
    return avg_inference_time_ms

def evaluate_models():
    # 1. Muat data test
    _, _, test_ds, _, class_names = get_data_pipelines()
    
    # Kumpulkan label aktual
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    y_true_indices = np.argmax(y_true, axis=1)
    
    # 2. Muat model
    models = load_all_models()
    
    comparison_data = []
    
    # Untuk plotting ROC & PR Curve
    predictions_dict = {}
    
    for name, model in models.items():
        print(f"\nEvaluating Model: {name}...")
        
        # Prediksi probabilitas
        start_pred = time.time()
        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred_indices = np.argmax(y_pred_probs, axis=1)
        predictions_dict[name] = y_pred_probs
        
        # Hitung Metrik Evaluasi
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        
        # Classification Report (Precision, Recall, F1)
        report = classification_report(y_true_indices, y_pred_indices, target_names=class_names, output_dict=True)
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        
        # ROC-AUC (Multi-class One-vs-Rest)
        roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
        
        # Model Size (MB)
        model_path = os.path.join(MODEL_SAVE_DIR, f"{name}_best.keras")
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Waktu Training & History
        history_path = os.path.join(HISTORY_SAVE_DIR, f"{name}_history.json")
        training_time = 0.0
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
                training_time = history_data.get('training_time', 0.0)
                
        # Waktu Inferensi per Gambar
        avg_inf_time_ms = measure_inference_time(model, test_ds)
        
        comparison_data.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": macro_precision,
            "Recall": macro_recall,
            "F1-Score": macro_f1,
            "ROC-AUC": roc_auc,
            "Inference Time (ms/img)": avg_inf_time_ms,
            "Model Size (MB)": model_size_mb,
            "Training Time (sec)": training_time
        })
        
        # 3. Plot Confusion Matrix
        cm = confusion_matrix(y_true_indices, y_pred_indices)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, f'confusion_matrix_{name}.png'))
        plt.close()
        
        # 4. Save Classification Report Text
        with open(os.path.join(VISUALIZATION_DIR, f'classification_report_{name}.txt'), 'w') as f:
            f.write(classification_report(y_true_indices, y_pred_indices, target_names=class_names))
            
    # Simpan hasil perbandingan model ke DataFrame & Markdown
    df_comp = pd.DataFrame(comparison_data)
    print("\n=== Model Comparison Results ===")
    print(df_comp.to_string(index=False))
    
    comp_md_path = os.path.join(VISUALIZATION_DIR, "model_comparison.md")
    df_comp.to_markdown(comp_md_path, index=False)
    print(f"Comparison markdown saved to {comp_md_path}")
    
    # 5. Plot Training History (Accuracy & Loss Curves)
    plot_training_curves()
    
    # 6. Plot ROC Curve (Multi-Class)
    plot_roc_curves(y_true, predictions_dict, class_names)
    
    # 7. Plot Precision-Recall Curve (Multi-Class)
    plot_pr_curves(y_true, predictions_dict, class_names)
    
    print("\nAll evaluation figures generated and saved successfully!")

def plot_training_curves():
    """
    Menggambar grafik kurva akurasi dan loss untuk ketiga model.
    """
    model_names = ["mobilenet_v2", "efficientnet_b0", "efficientnet_b0_cbam"]
    display_names = ["MobileNetV2 (Baseline)", "EfficientNetB0 (SOTA)", "EfficientNetB0 + CBAM (Innovative)"]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for name, disp_name, color in zip(model_names, display_names, colors):
        history_path = os.path.join(HISTORY_SAVE_DIR, f"{name}_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                h = json.load(f)
            
            epochs_range = range(1, len(h['accuracy']) + 1)
            
            # Accuracy Curves
            ax1.plot(epochs_range, h['accuracy'], label=f'{disp_name} - Train', color=color, linestyle='--')
            ax1.plot(epochs_range, h['val_accuracy'], label=f'{disp_name} - Val', color=color, linestyle='-')
            
            # Loss Curves
            ax2.plot(epochs_range, h['loss'], label=f'{disp_name} - Train', color=color, linestyle='--')
            ax2.plot(epochs_range, h['val_loss'], label=f'{disp_name} - Val', color=color, linestyle='-')
            
    ax1.set_title('Training & Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Training & Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history_curves.png'))
    plt.close()

def plot_roc_curves(y_true, predictions_dict, class_names):
    """
    Menggambar ROC Curve multi-kelas untuk model terbaik (EfficientNetB0 + CBAM)
    dan membandingkan ROC makro untuk ketiga model.
    """
    plt.figure(figsize=(10, 8))
    
    # Gambarkan ROC Kurva per kelas hanya untuk model terbaik: EfficientNetB0 + CBAM
    best_model_key = "efficientnet_b0_cbam"
    if best_model_key in predictions_dict:
        y_pred = predictions_dict[best_model_key]
        for i, cls_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{cls_name} (AUC = {roc_auc:.3f})')
            
    # Gambarkan macro-average ROC untuk ketiga model sebagai pembanding
    for name, y_pred in predictions_dict.items():
        # Hitung ROC makro secara manual
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        tpr_list = []
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            tpr_list.append(np.interp(fpr_grid, fpr, tpr))
        mean_tpr = np.mean(tpr_list, axis=0)
        mean_auc = auc(fpr_grid, mean_tpr)
        
        disp_name = {
            "mobilenet_v2": "MobileNetV2 (Macro)",
            "efficientnet_b0": "EfficientNetB0 (Macro)",
            "efficientnet_b0_cbam": "EfficientNetB0+CBAM (Macro)"
        }.get(name, name)
        
        plt.plot(fpr_grid, mean_tpr, label=f'**{disp_name} (AUC = {mean_auc:.3f})**', linestyle=':', linewidth=3)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'roc_curves_comparison.png'))
    plt.close()

def plot_pr_curves(y_true, predictions_dict, class_names):
    """
    Menggambar Precision-Recall Curve multi-kelas untuk model terbaik.
    """
    plt.figure(figsize=(10, 8))
    
    best_model_key = "efficientnet_b0_cbam"
    if best_model_key in predictions_dict:
        y_pred = predictions_dict[best_model_key]
        for i, cls_name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{cls_name} (AP = {pr_auc:.3f})')
            
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curves - EfficientNetB0 + CBAM')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'precision_recall_curves.png'))
    plt.close()

if __name__ == "__main__":
    evaluate_models()

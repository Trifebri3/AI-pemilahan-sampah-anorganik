import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from src.config import DATASET_DIR, IMAGE_SIZE, BATCH_SIZE

# Pipeline Augmentasi Data (Hanya untuk Training)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1)
])

def calculate_class_weights(train_dir):
    """
    Menghitung class weights berdasarkan jumlah sampel di tiap kelas 
    untuk menyeimbangkan gradien saat model dilatih.
    """
    classes = sorted(os.listdir(train_dir))
    y = []
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        # Ambil semua file gambar
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        y.extend([class_indices[cls]] * len(files))
        
    unique_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y
    )
    class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    print(f"Calculated Class Weights: {class_weights}")
    return class_weights

def get_data_pipelines():
    """
    Memuat subset Train, Validation, dan Test, lalu mengonversinya 
    menjadi tf.data.Dataset pipeline yang efisien dengan prefetching.
    """
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    test_dir = os.path.join(DATASET_DIR, "test")
    
    # 1. Load datasets from directory
    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    test_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Ambil label kelas
    class_names = train_raw.class_names
    print(f"Class names found: {class_names}")
    
    # 2. Terapkan Data Augmentasi pada Train Dataset secara paralel
    train_ds = train_raw.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 3. Optimasi dataset pipeline dengan caching & prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_raw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_raw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 4. Hitung Class Weights
    class_weights = calculate_class_weights(train_dir)
    
    return train_ds, val_ds, test_ds, class_weights, class_names

if __name__ == "__main__":
    # Test pipeline loading
    train_ds, val_ds, test_ds, weights, names = get_data_pipelines()
    print("Dataset pipelines loaded successfully.")

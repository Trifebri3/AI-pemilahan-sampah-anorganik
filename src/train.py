import os
import json
import time
import tensorflow as tf
from src.config import (
    MODEL_SAVE_DIR, HISTORY_SAVE_DIR, INIT_LR, EPOCHS, INPUT_SHAPE, NUM_CLASSES
)
from src.data_preprocessing import get_data_pipelines
from src.models import (
    build_mobilenet_v2, build_efficientnet_b0, build_efficientnet_b0_cbam
)

# Batasi penggunaan memori GPU jika ada agar tidak crash (jika berjalan di CUDA)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth enabled.")
    except RuntimeError as e:
        print(f"Error enabling GPU growth: {e}")

def train_model(model_builder, model_name, train_ds, val_ds, class_weights):
    print(f"\n==================================================")
    print(f"Training Model: {model_name}")
    print(f"==================================================")
    
    # 1. Bangun model
    model = model_builder(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    
    # Phase 1: Train classification head with base model frozen
    print("Phase 1: Training Classification Head (Feature Extraction)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Path file penyimpanan model
    best_keras_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.keras")
    best_h5_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.h5")
    
    # Callback untuk menyimpan checkpoint terbaik dari Phase 1
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_keras_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    start_time = time.time()
    
    # Train head for 5 epochs
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        class_weight=class_weights,
        callbacks=[checkpoint_cb],
        verbose=1
    )
    
    # Phase 2: Fine-Tuning (Unfreeze only the top layers of the backbone)
    print("\nPhase 2: Fine-Tuning the top layers of the backbone...")
    unfrozen = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name or "mobilenet" in layer.name:
            layer.trainable = True
            
            # Freeze the lower layers, unfreeze the top layers
            # EfficientNetB0 has ~238 layers, MobileNetV2 has ~154 layers.
            num_freeze = 180 if "efficientnet" in layer.name else 110
            for sublayer in layer.layers[:num_freeze]:
                sublayer.trainable = False
            for sublayer in layer.layers[num_freeze:]:
                sublayer.trainable = True
                
            print(f"Backbone unfrozen (from layer index {num_freeze} onwards): {layer.name}")
            unfrozen = True
            
    if not unfrozen:
        print("Warning: Backbone layer not detected/unfrozen via Model class match. Unfreezing layer at index 1.")
        if len(model.layers) > 1:
            base_model_layer = model.layers[1]
            base_model_layer.trainable = True
            num_freeze = 180 if "efficientnet" in base_model_layer.name else 110
            for sublayer in base_model_layer.layers[:num_freeze]:
                sublayer.trainable = False
            for sublayer in base_model_layer.layers[num_freeze:]:
                sublayer.trainable = True
            print(f"Backbone unfrozen (from layer index {num_freeze} onwards): {base_model_layer.name}")
            
    # Kompilasi ulang dengan learning rate yang sangat rendah
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks untuk fine-tuning
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Latih seluruh model untuk 10 epochs tambahan
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb],
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Total Training Time for {model_name}: {training_time:.2f} seconds")
    
    # 4. Simpan model dalam format legacy .h5 juga
    print(f"Saving model to H5 format...")
    try:
        model.save(best_h5_path)
        print("H5 model saved successfully.")
    except Exception as e:
        print(f"Failed to save H5 model: {e}")
        
    # 5. Simpan training history ke JSON file
    history_dict = history.history
    history_dict['training_time'] = training_time
    
    history_path = os.path.join(HISTORY_SAVE_DIR, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"History saved to {history_path}")
    
    return model, history_dict

def main():
    # Load dataset pipelines
    train_ds, val_ds, _, class_weights, _ = get_data_pipelines()
    
    # Model 1: MobileNetV2 (Baseline)
    train_model(
        build_mobilenet_v2,
        "mobilenet_v2",
        train_ds,
        val_ds,
        class_weights
    )
    
    # Model 2: EfficientNetB0 (SOTA)
    train_model(
        build_efficientnet_b0,
        "efficientnet_b0",
        train_ds,
        val_ds,
        class_weights
    )
    
    # Model 3: EfficientNetB0 + CBAM (Innovative)
    train_model(
        build_efficientnet_b0_cbam,
        "efficientnet_b0_cbam",
        train_ds,
        val_ds,
        class_weights
    )
    
    print("\nAll models trained successfully! Running evaluation...")
    from src.evaluate import evaluate_models
    evaluate_models()
    print("\nEvaluation completed. All charts and reports have been updated.")

if __name__ == "__main__":
    main()

import os
import sys

# Tambahkan root directory ke sys.path agar modul 'src' dapat ditemukan
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import datetime
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from src.config import MODEL_SAVE_DIR, CLASSES, IMAGE_SIZE
from src.cbam import ChannelAttention, SpatialAttention

app = Flask(__name__)

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(BASE_DIR, "history", "classification_history.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "history", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load best model (EfficientNetB0+CBAM)
model_path = os.path.join(MODEL_SAVE_DIR, "efficientnet_b0_cbam_best.keras")
model = None

def get_model():
    """
    Memuat model terbaik dengan malas (lazy loading) ketika request pertama masuk 
    atau ketika server diinisialisasi.
    """
    global model
    if model is None:
        custom_objects = {
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention
        }
        if os.path.exists(model_path):
            print(f"Loading best model for API from {model_path}...")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model not found at {model_path}. Predict will fail until models are trained.")
    return model

def log_classification(pred_class, confidence, filename):
    """
    Mencatat data klasifikasi ke file CSV untuk sinkronisasi dengan Streamlit dashboard.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "timestamp,predicted_class,confidence,image_name\n"
    
    file_exists = os.path.exists(HISTORY_FILE)
    
    with open(HISTORY_FILE, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(f"{timestamp},{pred_class},{confidence:.4f},{filename}\n")

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "message": "AI Smart Bin API Server is running.",
        "model_loaded": get_model() is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Pastikan model termuat
    net = get_model()
    if net is None:
        return jsonify({"error": "Model not loaded on server."}), 500
        
    # 2. Periksa keberadaan file gambar
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form-data parameter 'image'."}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
        
    # 3. Simpan file sementara untuk dokumentasi dan Grad-CAM dashboard
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp_str}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(save_path)
    
    try:
        # 4. Baca dan preprocess gambar menggunakan OpenCV
        img = cv2.imread(save_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
        
        # 5. Jalankan inferensi
        preds = net.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = float(preds[class_idx])
        pred_class = CLASSES[class_idx]
        class_translation = {
            "Glass": "Kaca",
            "Paper": "Kertas",
            "Metal": "Logam",
            "Plastic": "Plastik",
            "Residu": "Residu"
        }
        pred_class = class_translation.get(pred_class, pred_class)
        if pred_class not in ["Kaca", "Kertas", "Logam", "Plastik", "Residu"]:
            pred_class = "Jenis Tidak Dikenali"
        
        # 6. Catat riwayat klasifikasi
        log_classification(pred_class, confidence, safe_filename)
        
        # 7. Respon ke ESP32-CAM
        # ESP32-CAM membutuhkan informasi kategori yang jelas untuk menggerakkan servo
        # Contoh sudut default servo:
        # Kertas -> 0 derajat, Kaca -> 45 derajat, Logam -> 90 derajat, Plastik -> 135 derajat, Residu -> 180 derajat
        servo_angles = {
            "Kertas": 0,
            "Kaca": 45,
            "Logam": 90,
            "Plastik": 135,
            "Residu": 180
        }
        
        return jsonify({
            "success": True,
            "category": pred_class,
            "confidence": confidence,
            "servo_angle": servo_angles.get(pred_class, 90),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    # Jalankan server Flask di port 5000 (mendengarkan semua antarmuka jaringan)
    app.run(host="0.0.0.0", port=5000, debug=False)

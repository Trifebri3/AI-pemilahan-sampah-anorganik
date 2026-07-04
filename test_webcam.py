import os
import sys
import time

# Tambahkan path root agar modul 'src' dapat diimpor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import cv2
import numpy as np
import tensorflow as tf
from src.config import MODEL_SAVE_DIR, CLASSES, IMAGE_SIZE
from src.cbam import ChannelAttention, SpatialAttention

def main():
    print("=== AI Smart Bin Real-Time Continuous Scanner ===")
    
    # 1. Load best model
    model_path = os.path.join(MODEL_SAVE_DIR, "efficientnet_b0_cbam_best.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}!")
        print("Silakan jalankan training terlebih dahulu.")
        return
        
    print(f"Memuat model dari {model_path}...")
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model berhasil dimuat.")
    
    # 2. Setup mapping sudut servo
    servo_angles = {
        "Kertas": 0,
        "Kaca": 45,
        "Logam": 90,
        "Plastik": 135,
        "Residu": 180
    }
    
    # Kategori dalam bahasa Indonesia sesuai training labels
    class_translation = {
        "Glass": "Kaca",
        "Paper": "Kertas",
        "Metal": "Logam",
        "Plastic": "Plastik",
        "Residu": "Residu"
    }

    # 3. Akses Webcam
    print("Membuka webcam (port 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera/webcam!")
        return
        
    print("Kamera berhasil dibuka. Berjalan dalam mode pemindaian kontinu. Tekan tombol 'q' untuk keluar.")
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame.")
            break
            
        display_frame = frame.copy()
        h, w, _ = display_frame.shape
        
        # Hitung area kontur terluar untuk mendeteksi apakah objek dekat dengan kamera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        if contours:
            max_area = max(cv2.contourArea(c) for c in contours)
            
        # Ambang batas kontur relatif: jika lebih dari 1.5% dari area frame, objek dianggap dekat
        frame_area = w * h
        detected_area_percent = (max_area / frame_area) * 100.0
        is_close = detected_area_percent > 1.5
        
        if is_close:
            # 4. Preprocessing untuk model (hanya jika ada objek dekat)
            input_img = cv2.resize(frame, IMAGE_SIZE)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_batch = np.expand_dims(input_img, axis=0).astype(np.float32)
            
            # Jalankan prediksi kontinu secara otomatis
            preds = model.predict(input_batch, verbose=0)[0]
            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx]
            
            # Terjemahkan label kelas
            raw_class = CLASSES[pred_idx]
            pred_class = class_translation.get(raw_class, raw_class)
            servo_angle = servo_angles.get(pred_class, 90)
            
            status_text = f"TARGET: {pred_class.upper()} | CONF: {confidence:.0%} | AREA: {detected_area_percent:.1f}%"
            text_color = (0, 255, 0) # Hijau untuk target dekat terdeteksi kuat
        else:
            # Jika objek terlalu jauh atau tidak ada benda di depan kamera
            status_text = f"STANDBY | AREA: {detected_area_percent:.1f}% / MIN: 1.5%"
            text_color = (0, 255, 255) # Kuning-cyan untuk status standby
            servo_angle = 90
            confidence = 0.0
            
        # Hitung FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # 5. Gambarkan HUD Pemindaian Futuristik
        color_cyan = (255, 255, 0) # Cyan
        color_laser = (0, 0, 255) # Merah laser
        
        # Gambarkan Siku Pembatas Target (Viewfinder Brackets)
        length = 35
        thickness = 3
        # Top Left Corner
        cv2.line(display_frame, (50, 50), (50 + length, 50), color_cyan, thickness)
        cv2.line(display_frame, (50, 50), (50, 50 + length), color_cyan, thickness)
        # Top Right Corner
        cv2.line(display_frame, (w - 50, 50), (w - 50 - length, 50), color_cyan, thickness)
        cv2.line(display_frame, (w - 50, 50), (w - 50, 50 + length), color_cyan, thickness)
        # Bottom Left Corner
        cv2.line(display_frame, (50, h - 50), (50 + length, h - 50), color_cyan, thickness)
        cv2.line(display_frame, (50, h - 50), (50, h - 50 - length), color_cyan, thickness)
        # Bottom Right Corner
        cv2.line(display_frame, (w - 50, h - 50), (w - 50 - length, h - 50), color_cyan, thickness)
        cv2.line(display_frame, (w - 50, h - 50), (w - 50, h - 50 - length), color_cyan, thickness)
        
        # Gambarkan Garis Laser Pemindai Bergerak (Animated Scanning Line)
        scan_y = int((time.time() * 200) % (h - 100)) + 50
        # Laser line
        cv2.line(display_frame, (50, scan_y), (w - 50, scan_y), color_laser, 2)
        # Laser glow effect
        cv2.line(display_frame, (50, scan_y), (w - 50, scan_y), color_laser, 4, cv2.LINE_AA)
        
        # Kotak HUD Atas untuk Menampilkan Hasil Prediksi
        cv2.rectangle(display_frame, (50, 15), (w - 50, 75), (20, 20, 20), -1) # Latar gelap
        cv2.rectangle(display_frame, (50, 15), (w - 50, 75), color_cyan, 1) # Border
        
        cv2.putText(display_frame, status_text, (70, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        
        # Info FPS & Tombol Keluar di bagian bawah
        cv2.rectangle(display_frame, (50, h - 35), (w - 50, h - 10), (10, 10, 10), -1)
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Frame Resolution: {w}x{h}", (60, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "Tekan 'q' to exit", (w - 230, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Tampilkan frame
        cv2.imshow("AI Smart Bin Real-Time Classifier", display_frame)
        
        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Pemindaian selesai.")

if __name__ == "__main__":
    main()

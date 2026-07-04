import os
import sys

# Tambahkan root directory ke sys.path agar modul 'src' dapat ditemukan
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import json
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from src.config import MODEL_SAVE_DIR, VISUALIZATION_DIR, CLASSES, IMAGE_SIZE
from src.cbam import ChannelAttention, SpatialAttention
from src.gradcam import generate_gradcam

# Set page configuration with premium dark theme
st.set_page_config(
    page_title="AI Smart Bin - Dashboard Monitoring",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and clean typography
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #2E7D32;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f1f8e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(BASE_DIR, "history", "classification_history.csv")
UPLOADS_DIR = os.path.join(BASE_DIR, "history", "uploads")

@st.cache_resource
def load_model_cached(model_name):
    """
    Memuat model secara cached agar tidak membebani memori setiap interaksi.
    """
    path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.keras")
    if os.path.exists(path):
        custom_objects = {
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention
        }
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    return None

def get_history_df():
    """
    Membaca riwayat deteksi dari CSV API.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def render_html(placeholder, html_code):
    cleaned = "\n".join([line.lstrip() for line in html_code.splitlines()])
    placeholder.markdown(cleaned, unsafe_allow_html=True)

# Navigation
st.sidebar.image("https://img.icons8.com/clouds/200/garbage-truck.png", width=120)
st.sidebar.markdown("<h2 style='text-align: center; color: #2E7D32;'>AI Smart Bin</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigasi Menu",
    ["Dashboard Real-Time", "Simulasi Fisik Smart Bin", "Evaluasi & Komparasi Model", "Interpretasi Grad-CAM (XAI)"]
)

# ----------------- PAGE 1: DASHBOARD REAL-TIME -----------------
if page == "Dashboard Real-Time":
    st.markdown("<h1 class='main-title'>Dashboard Monitoring Real-Time</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Sistem Deteksi Jenis Sampah Otomatis Menggunakan Computer Vision</p>", unsafe_allow_html=True)
    
    # KPIs
    df_history = get_history_df()
    col1, col2, col3 = st.columns(3)
    
    if not df_history.empty:
        total_items = len(df_history)
        avg_conf = df_history["confidence"].mean() * 100
        most_common = df_history["predicted_class"].mode().iloc[0] if not df_history["predicted_class"].empty else "N/A"
    else:
        total_items = 0
        avg_conf = 0.0
        most_common = "Belum Ada"
        
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total_items}</div>
            <div class='metric-label'>Total Sampah Terdeteksi</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 5px solid #2196f3;'>
            <div class='metric-value'>{avg_conf:.1f}%</div>
            <div class='metric-label'>Rata-rata Akurasi Keyakinan</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 5px solid #ff9800;'>
            <div class='metric-value'>{most_common}</div>
            <div class='metric-label'>Kategori Sampah Dominan</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Main content split: Detector vs Stats
    d_col1, d_col2 = st.columns([1, 1])
    
    with d_col1:
        st.markdown("### Uji Deteksi Citra")
        model_choice = st.selectbox(
            "Pilih Model AI",
            ["efficientnet_b0_cbam", "efficientnet_b0", "mobilenet_v2"],
            format_func=lambda x: {
                "mobilenet_v2": "MobileNetV2 (Baseline)",
                "efficientnet_b0": "EfficientNetB0 (SOTA)",
                "efficientnet_b0_cbam": "EfficientNetB0 + CBAM (Innovative)"
            }[x]
        )
        
        uploaded_file = st.file_uploader("Unggah Gambar Sampah...", type=["png", "jpg", "jpeg", "webp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Terunggah", use_container_width=True)
            
            # Button to trigger prediction
            if st.button("Jalankan Klasifikasi"):
                # Load model
                model = load_model_cached(model_choice)
                if model is None:
                    st.error(f"Gagal memuat model '{model_choice}'. Pastikan model sudah ditraining.")
                else:
                    with st.spinner("Mengklasifikasi dan menghitung Grad-CAM..."):
                        # Simpan file sementara untuk Grad-CAM
                        temp_path = os.path.join(UPLOADS_DIR, f"temp_{uploaded_file.name}")
                        image.save(temp_path)
                        
                        try:
                            # Hitung Grad-CAM
                            cam, _, pred_class = generate_gradcam(temp_path, model, model_choice)
                            
                            # Jalankan prediksi probabilitas untuk grafik bar
                            img_cv = cv2.imread(temp_path)
                            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                            img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
                            img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
                            
                            preds = model.predict(img_array, verbose=0)[0]
                            
                            # Hapus file temp
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                
                            # Tampilkan Hasil
                            st.success(f"**Prediksi Kategori: {pred_class}** (Confidence: {preds[np.argmax(preds)]*100:.2f}%)")
                            
                            # Tampilkan Grad-CAM side-by-side
                            cam_col1, cam_col2 = st.columns(2)
                            with cam_col1:
                                st.image(image, caption="Citra Asli", use_container_width=True)
                            with cam_col2:
                                st.image(cam, caption="Explainable AI - Grad-CAM Activation Map", use_container_width=True)
                                
                            # Probabilities Bar Chart
                            st.markdown("#### Distribusi Keyakinan Kelas:")
                            chart_data = pd.DataFrame({
                                "Kategori": CLASSES,
                                "Keyakinan (%)": [float(p)*100 for p in preds]
                            })
                            st.bar_chart(chart_data.set_index("Kategori"))
                            
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat memproses gambar: {str(e)}")
                            
    with d_col2:
        st.markdown("### Statistik Deteksi & Riwayat")
        if not df_history.empty:
            # Bar Chart: Counts by Category
            st.markdown("#### Frekuensi Deteksi per Kategori")
            cls_counts = df_history["predicted_class"].value_counts().reset_index()
            cls_counts.columns = ["Kategori", "Jumlah"]
            # Pad categories that have zero counts
            all_classes = pd.DataFrame({"Kategori": CLASSES})
            cls_counts = pd.merge(all_classes, cls_counts, on="Kategori", how="left").fillna(0)
            st.bar_chart(cls_counts.set_index("Kategori"))
            
            # Line Chart: Confidence Over Time
            st.markdown("#### Tren Keyakinan Deteksi Terakhir")
            df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
            df_time = df_history.sort_values("timestamp").tail(20)
            st.line_chart(df_time.set_index("timestamp")["confidence"])
            
            # History Table
            st.markdown("#### Tabel Riwayat Klasifikasi Terbaru")
            st.dataframe(
                df_history.sort_values("timestamp", ascending=False).head(10),
                column_config={
                    "timestamp": "Waktu",
                    "predicted_class": "Hasil Prediksi",
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
                    "image_name": "File Gambar"
                },
                use_container_width=True
            )
        else:
            st.info("Belum ada riwayat deteksi. Jalankan klasifikasi secara manual atau hubungkan dengan ESP32-CAM.")

# ----------------- PAGE: SIMULASI FISIK SMART BIN -----------------
elif page == "Simulasi Fisik Smart Bin":
    st.markdown("<h1 class='main-title'>Simulasi Fisik Smart Bin</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Uji Coba Deteksi Kamera dan Simulasi Gerakan Motor Servo Pemilah Sampah</p>", unsafe_allow_html=True)
    
    st.info("Simulasi ini meniru cara kerja fisik tempat sampah pintar. Ketika kamera mendeteksi sampah, sistem akan menentukan kategori sampah, mengirim sinyal sudut servo, dan membuka pintu tempat sampah yang sesuai secara otomatis.")

    s_col1, s_col2 = st.columns([1, 1])
    
    with s_col1:
        st.markdown("### Kontrol Simulasi")
        model_choice = st.selectbox(
            "Pilih Model Deteksi",
            ["efficientnet_b0_cbam", "efficientnet_b0", "mobilenet_v2"],
            key="sim_model",
            format_func=lambda x: {
                "mobilenet_v2": "MobileNetV2 (Baseline)",
                "efficientnet_b0": "EfficientNetB0 (SOTA)",
                "efficientnet_b0_cbam": "EfficientNetB0 + CBAM (Innovative)"
            }[x]
        )
        
        input_mode = st.radio("Metode Pengambilan Citra", ["Kamera Web Live (Webcam)", "Unggah Citra Sampah"], horizontal=True)
        
        img_file = None
        run_auto = False
        lock_close_only = True
        close_sensitivity = 1.5
        if input_mode == "Kamera Web Live (Webcam)":
            run_auto = st.checkbox("Aktifkan Pemindaian Otomatis (Scan Kontinu)", value=False)
            if run_auto:
                st.markdown("---")
                st.markdown("**Konfigurasi Jarak Pindai:**")
                lock_close_only = st.checkbox("Kunci Hanya Objek Dekat", value=True)
                if lock_close_only:
                    close_sensitivity = st.slider("Sensitivitas Jarak (Min Area %)", min_value=0.2, max_value=8.0, value=1.5, step=0.1)
        else:
            img_file = st.file_uploader("Pilih file gambar sampah untuk simulasi...", type=["png", "jpg", "jpeg", "webp"])
            
    with s_col2:
        st.markdown("### Visualisasi Respon Mekanis")
        
        if input_mode == "Kamera Web Live (Webcam)" and run_auto:
            model = load_model_cached(model_choice)
            if model is None:
                st.error(f"Gagal memuat model '{model_choice}'. Selesaikan proses training terlebih dahulu.")
            else:
                camera_placeholder = s_col1.empty()
                visualizer_placeholder = st.empty()
                
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Kamera tidak dapat diakses atau sedang digunakan oleh aplikasi lain.")
                else:
                    try:
                        import time
                        while run_auto:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Hitung area kontur untuk mendeteksi apakah objek dekat dengan kamera
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                            edges = cv2.Canny(blurred, 50, 150)
                            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            max_area = 0
                            if contours:
                                max_area = max(cv2.contourArea(c) for c in contours)
                            
                            h_f, w_f, _ = frame.shape
                            frame_area = w_f * h_f
                            detected_area_percent = (max_area / frame_area) * 100.0
                            
                            if lock_close_only:
                                min_area_pixels = (close_sensitivity / 100.0) * frame_area
                                is_close = max_area > min_area_pixels
                            else:
                                is_close = True
                            
                            if is_close:
                                # Preprocess frame
                                img_resized = cv2.resize(frame, IMAGE_SIZE)
                                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                                img_array = np.expand_dims(img_rgb, axis=0).astype(np.float32)
                                
                                # Predict
                                preds = model.predict(img_array, verbose=0)[0]
                                class_idx = np.argmax(preds)
                                confidence = float(preds[class_idx])
                                raw_class = CLASSES[class_idx]
                                
                                class_translation = {
                                    "Glass": "Kaca",
                                    "Paper": "Kertas",
                                    "Metal": "Logam",
                                    "Plastic": "Plastik",
                                    "Residu": "Residu"
                                }
                                pred_class = class_translation.get(raw_class, raw_class)
                                if pred_class not in ["Kaca", "Kertas", "Logam", "Plastik", "Residu"] or confidence < 0.50:
                                    pred_class = "TIDAK DIKETAHUI"
                                    
                                if pred_class == "TIDAK DIKETAHUI":
                                    servo_angle = 90
                                    status_text = f"TIDAK DIKETAHUI | CONF: {confidence*100:.1f}%"
                                    text_color = (0, 0, 255)  # Merah
                                    title_status = "TIDAK DIKETAHUI"
                                    desc_status = f"Gambar tidak dikenali atau tidak identik dengan model (Confidence: {confidence*100:.2f}%)."
                                    icon_text = "TIDAK ADA"
                                    color = "#757575"
                                else:
                                    servo_angle = {
                                        "Kertas": 0,
                                        "Kaca": 45,
                                        "Logam": 90,
                                        "Plastik": 135,
                                        "Residu": 180
                                    }.get(pred_class, 90)
                                    status_text = f"TARGET: {pred_class.upper()} | AREA: {detected_area_percent:.1f}% | SERVO: {servo_angle} DEG"
                                    text_color = (0, 255, 0)
                                    title_status = f"TONG SAMPAH {pred_class.upper()} DIBUKA"
                                    desc_status = f"Hasil Deteksi: {pred_class} (Confidence: {confidence*100:.2f}%)"
                                    icon_text = "SAMPAH"
                                    color = {
                                        "Kertas": "#FBC02D",   # Kuning
                                        "Kaca": "#00ACC1",     # Cyan
                                        "Logam": "#757575",    # Abu-abu
                                        "Plastik": "#1E88E5",  # Biru
                                        "Residu": "#795548"    # Cokelat
                                    }.get(pred_class, "#333")
                            else:
                                pred_class = "Mencari..."
                                confidence = 0.0
                                servo_angle = 90
                                status_text = f"STANDBY | AREA: {detected_area_percent:.1f}% / MIN: {close_sensitivity:.1f}%"
                                text_color = (0, 255, 255)
                                title_status = "TIDAK ADA OBJEK DEKAT"
                                desc_status = f"Dekatkan objek sampah ke kamera (min {close_sensitivity:.1f}% area kamera)."
                                icon_text = "TIDAK ADA"
                                color = "#757575"
                            
                            # HUD overlay on frame (No emojis)
                            display_frame = frame.copy()
                            h_f, w_f, _ = display_frame.shape
                            
                            # Viewfinder corners
                            length = 35
                            thickness = 3
                            color_cyan = (0, 255, 255) # Cyan (BGR: Yellow in OpenCV)
                            cv2.line(display_frame, (40, 40), (40 + length, 40), color_cyan, thickness)
                            cv2.line(display_frame, (40, 40), (40, 40 + length), color_cyan, thickness)
                            cv2.line(display_frame, (w_f - 40, 40), (w_f - 40 - length, 40), color_cyan, thickness)
                            cv2.line(display_frame, (w_f - 40, 40), (w_f - 40, 40 + length), color_cyan, thickness)
                            cv2.line(display_frame, (40, h_f - 40), (40 + length, h_f - 40), color_cyan, thickness)
                            cv2.line(display_frame, (40, h_f - 40), (40, h_f - 40 - length), color_cyan, thickness)
                            cv2.line(display_frame, (w_f - 40, h_f - 40), (w_f - 40 - length, h_f - 40), color_cyan, thickness)
                            cv2.line(display_frame, (w_f - 40, h_f - 40), (w_f - 40, h_f - 40 - length), color_cyan, thickness)
                            
                            # Scanning laser line
                            scan_y = int((time.time() * 200) % (h_f - 80)) + 40
                            cv2.line(display_frame, (40, scan_y), (w_f - 40, scan_y), (0, 0, 255), 2)
                            
                            # Tulis status teks pemindaian ke frame video
                            cv2.putText(display_frame, status_text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                            
                            # Show frame in streamlit (converting BGR to RGB)
                            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                            
                            # Render visualizer HTML
                            render_html(visualizer_placeholder, f"""
                            <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid {color}; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 1rem;">
                                <h3 style="color: {color}; margin-top: 0; font-weight: 700;">{title_status}</h3>
                                <p style="font-size: 1.1rem; color: #333; margin-bottom: 1.5rem;">{desc_status}</p>
                                
                                <div style="display: flex; justify-content: center; align-items: center; gap: 3rem; flex-wrap: wrap;">
                                    <!-- Servo dial animation -->
                                    <div style="text-align: center;">
                                        <p style="font-weight: 600; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Posisi Servo Motor</p>
                                        <div style="position: relative; width: 140px; height: 70px; border-bottom: 3px solid #ccc; overflow: hidden; margin: 0 auto;">
                                            <div style="position: absolute; bottom: 0; left: 50%; transform: translate(-50%, 0) rotate({servo_angle}deg); transform-origin: center bottom; width: 4px; height: 60px; background: #d32f2f; transition: all 0.5s ease-out; border-radius: 4px;">
                                                <div style="position: absolute; top: 0; left: -4px; width: 12px; height: 12px; border-radius: 50%; background: #d32f2f;"></div>
                                            </div>
                                            <div style="position: absolute; bottom: -10px; left: 50%; transform: translate(-50%, 0); width: 24px; height: 24px; border-radius: 50%; background: #333; border: 2px solid #fff;"></div>
                                        </div>
                                        <p style="font-size: 1.3rem; font-weight: 700; color: #333; margin-top: 0.5rem;">{servo_angle}°</p>
                                    </div>
                                    
                                    <!-- Trash bin animation -->
                                    <div style="text-align: center;">
                                        <p style="font-weight: 600; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Wadah Pemilah</p>
                                        <div style="width: 110px; height: 140px; position: relative; margin: 0 auto;">
                                            <!-- Animated lid -->
                                            <div style="
                                                width: 100px; 
                                                height: 12px; 
                                                background: {color}; 
                                                border-radius: 6px 6px 0 0; 
                                                position: absolute; 
                                                top: 15px; 
                                                left: 5px; 
                                                transform-origin: left bottom; 
                                                transform: rotate(-60deg); 
                                                transition: transform 0.5s ease-out;
                                                box-shadow: 0 -1px 3px rgba(0,0,0,0.1);
                                                z-index: 10;
                                            "></div>
                                            <!-- Trash falling down animation -->
                                            <div style="
                                                width: 60px; 
                                                height: 20px; 
                                                font-size: 0.75rem;
                                                font-weight: bold;
                                                color: #fff;
                                                background: #d32f2f;
                                                border-radius: 3px;
                                                position: absolute; 
                                                top: -30px; 
                                                left: 25px; 
                                                animation: fallTrashSim 2s infinite ease-in;
                                                z-index: 5;
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                            ">
                                                {icon_text}
                                            </div>
                                            <!-- Bin body -->
                                            <div style="
                                                width: 90px; 
                                                height: 95px; 
                                                background: #f5f5f5; 
                                                border: 4px solid {color};
                                                border-radius: 0 0 12px 12px; 
                                                position: absolute; 
                                                bottom: 10px; 
                                                left: 10px;
                                                box-shadow: inset 0 3px 6px rgba(0,0,0,0.1);
                                                z-index: 2;
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                            ">
                                                <span style="font-weight: 800; font-size: 0.85rem; color: #555; margin-top: 15px;">{pred_class.upper()}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div style="margin-top: 1.5rem; padding: 0.5rem; background: #fafafa; border: 1px solid #eee; border-radius: 6px; font-size: 0.85rem; color: #666;">
                                    Sudut Servo Dikirim: {servo_angle} derajat. Katup penutup tong otomatis ditutup setelah sampah masuk.
                                </div>
                            </div>
                            
                            <style>
                                @keyframes fallTrashSim {{
                                    0% {{ transform: translateY(0) rotate(0deg); opacity: 0; }}
                                    15% {{ opacity: 1; }}
                                    50% {{ transform: translateY(85px) rotate(180deg); opacity: 0.8; }}
                                    60%, 100% {{ transform: translateY(85px) rotate(180deg); opacity: 0; }}
                                }}
                            </style>
                            """)
                            
                            time.sleep(0.03)
                    finally:
                        cap.release()
                        
        elif input_mode == "Kamera Web Live (Webcam)" and not run_auto:
            st.info("Centang kotak 'Aktifkan Pemindaian Otomatis' di panel kiri untuk memulai kamera dan memindai secara otomatis.")
            
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px dashed #ccc; border-radius: 15px; margin-top: 1rem;">
                <h3 style="color: #888; margin-top: 0; font-weight: 700;">Kamera Non-Aktif</h3>
                <p style="font-size: 1rem; color: #666;">Silakan aktifkan pemindaian untuk melihat respon mekanisme tempat sampah.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif img_file is not None:
            # Load model
            model = load_model_cached(model_choice)
            if model is None:
                st.error(f"Gagal memuat model '{model_choice}'. Selesaikan proses training terlebih dahulu.")
            else:
                with st.spinner("Menganalisis gambar..."):
                    # Save image temporarily
                    temp_name = f"sim_{img_file.name if hasattr(img_file, 'name') else 'webcam.jpg'}"
                    temp_path = os.path.join(UPLOADS_DIR, temp_name)
                    
                    # Save stream to file
                    with open(temp_path, "wb") as f:
                        f.write(img_file.getvalue())
                        
                    try:
                        # Process image
                        img_cv = cv2.imread(temp_path)
                        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
                        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
                        
                        # Run predict
                        preds = model.predict(img_array, verbose=0)[0]
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
                        if pred_class not in ["Kaca", "Kertas", "Logam", "Plastik", "Residu"] or confidence < 0.50:
                            pred_class = "TIDAK DIKETAHUI"
                        
                        # Log classification into history for unified stats
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        header = "timestamp,predicted_class,confidence,image_name\n"
                        file_exists = os.path.exists(HISTORY_FILE)
                        with open(HISTORY_FILE, "a") as f:
                            if not file_exists:
                                f.write(header)
                            f.write(f"{timestamp},{pred_class},{confidence:.4f},{temp_name}\n")
                        
                        if pred_class == "TIDAK DIKETAHUI":
                            st.error(f"**Hasil Deteksi: TIDAK DIKETAHUI** (Confidence: {confidence*100:.2f}%)")
                            render_html(st, f"""
                            <div style="text-align: center; padding: 2rem; background: #ffffff; border: 2px solid #757575; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 1rem;">
                                <h3 style="color: #757575; margin-top: 0; font-weight: 700;">TIDAK DIKETAHUI</h3>
                                <p style="font-size: 1.1rem; color: #333; margin-bottom: 0;">Gambar sampah tidak dikenali atau tidak identik dengan kategori yang didukung model (Confidence: <b>{confidence*100:.2f}%</b>).</p>
                            </div>
                            """)
                        else:
                            st.success(f"**Hasil Deteksi: {pred_class}** (Confidence: {confidence*100:.2f}%)")
                            
                            # Define colors and servo angles
                            colors = {
                                "Kertas": "#FBC02D",   # Kuning
                                "Kaca": "#00ACC1",     # Cyan
                                "Logam": "#757575",    # Abu-abu
                                "Plastik": "#1E88E5",  # Biru
                                "Residu": "#795548"    # Cokelat
                            }
                            
                            servo_angles = {
                                "Kertas": 0,
                                "Kaca": 45,
                                "Logam": 90,
                                "Plastik": 135,
                                "Residu": 180
                            }
                            
                            icons = {
                                "Kertas": "SAMPAH",
                                "Kaca": "SAMPAH",
                                "Logam": "SAMPAH",
                                "Plastik": "SAMPAH",
                                "Residu": "SAMPAH"
                            }
                            
                            color = colors.get(pred_class, "#333")
                            angle = servo_angles.get(pred_class, 90)
                            icon = icons.get(pred_class, "SAMPAH")
                            
                            # Custom animated HTML for Servo and Bin Opening (No emojis)
                            render_html(st, f"""
                            <div style="text-align: center; padding: 1.5rem; background: #ffffff; border: 2px solid {color}; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 1rem;">
                                <h3 style="color: {color}; margin-top: 0; font-weight: 700;">TONG SAMPAH {pred_class.upper()} DIBUKA</h3>
                                <p style="font-size: 1.1rem; color: #333; margin-bottom: 1.5rem;">Hasil Deteksi: <b>{pred_class}</b> (Confidence: <b>{confidence*100:.2f}%</b>)</p>
                                
                                <div style="display: flex; justify-content: center; align-items: center; gap: 3rem; flex-wrap: wrap;">
                                    <!-- Servo dial animation -->
                                    <div style="text-align: center;">
                                        <p style="font-weight: 600; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Posisi Servo Motor</p>
                                        <div style="position: relative; width: 140px; height: 70px; border-bottom: 3px solid #ccc; overflow: hidden; margin: 0 auto;">
                                            <div style="position: absolute; bottom: 0; left: 50%; transform: translate(-50%, 0) rotate({angle}deg); transform-origin: center bottom; width: 4px; height: 60px; background: #d32f2f; transition: all 1.5s ease-out; border-radius: 4px;">
                                                <div style="position: absolute; top: 0; left: -4px; width: 12px; height: 12px; border-radius: 50%; background: #d32f2f;"></div>
                                            </div>
                                            <div style="position: absolute; bottom: -10px; left: 50%; transform: translate(-50%, 0); width: 24px; height: 24px; border-radius: 50%; background: #333; border: 2px solid #fff;"></div>
                                        </div>
                                        <p style="font-size: 1.3rem; font-weight: 700; color: #333; margin-top: 0.5rem;">{angle}°</p>
                                    </div>
                                    
                                    <!-- Trash bin animation -->
                                    <div style="text-align: center;">
                                        <p style="font-weight: 600; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Wadah Pemilah</p>
                                        <div style="width: 110px; height: 140px; position: relative; margin: 0 auto;">
                                            <!-- Animated lid -->
                                            <div style="
                                                width: 100px; 
                                                height: 12px; 
                                                background: {color}; 
                                                border-radius: 6px 6px 0 0; 
                                                position: absolute; 
                                                top: 15px; 
                                                left: 5px; 
                                                transform-origin: left bottom; 
                                                transform: rotate(-60deg); 
                                                transition: transform 1.2s ease-out;
                                                box-shadow: 0 -1px 3px rgba(0,0,0,0.1);
                                                z-index: 10;
                                            "></div>
                                            <!-- Trash falling down animation -->
                                            <div style="
                                                width: 60px; 
                                                height: 20px; 
                                                font-size: 0.75rem;
                                                font-weight: bold;
                                                color: #fff;
                                                background: #d32f2f;
                                                border-radius: 3px;
                                                position: absolute; 
                                                top: -30px; 
                                                left: 25px; 
                                                animation: fallTrashSim 2s infinite ease-in;
                                                z-index: 5;
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                            ">
                                                {icon}
                                            </div>
                                            <!-- Bin body -->
                                            <div style="
                                                width: 90px; 
                                                height: 95px; 
                                                background: #f5f5f5; 
                                                border: 4px solid {color};
                                                border-radius: 0 0 12px 12px; 
                                                position: absolute; 
                                                bottom: 10px; 
                                                left: 10px;
                                                box-shadow: inset 0 3px 6px rgba(0,0,0,0.1);
                                                z-index: 2;
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                            ">
                                                <span style="font-weight: 800; font-size: 0.85rem; color: #555; margin-top: 15px;">{pred_class.upper()}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div style="margin-top: 1.5rem; padding: 0.5rem; background: #fafafa; border: 1px solid #eee; border-radius: 6px; font-size: 0.85rem; color: #666;">
                                    Sudut Servo Dikirim: {angle}° dan katup penutup tong otomatis ditutup setelah sampah masuk.
                                </div>
                            </div>
                            
                            <style>
                                @keyframes fallTrashSim {{
                                    0% {{ transform: translateY(0) rotate(0deg); opacity: 0; }}
                                    15% {{ opacity: 1; }}
                                    50% {{ transform: translateY(85px) rotate(180deg); opacity: 0.8; }}
                                    60%, 100% {{ transform: translateY(85px) rotate(180deg); opacity: 0; }}
                                }}
                            </style>
                            """)
                        
                        # Clean temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        st.error(f"Gagal melakukan klasifikasi: {str(e)}")
        else:
            st.write("Arahkan sampah Anda ke kamera webcam atau unggah foto sampah di sebelah kiri untuk melihat simulasi gerakan mekanis Smart Bin secara otomatis.")

# ----------------- PAGE 2: EVALUASI & KOMPARASI MODEL -----------------
elif page == "Evaluasi & Komparasi Model":
    st.markdown("<h1 class='main-title'>Evaluasi Ilmiah & Komparasi Model</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Laporan Perbandingan Performa antara MobileNetV2, EfficientNetB0, dan EfficientNetB0+CBAM</p>", unsafe_allow_html=True)
    
    # Section 1: Visualisasi Distribusi Dataset Citra Sampah
    st.markdown("### 📊 1. Rekapan Distribusi Data Citra Sampah")
    st.markdown("""
    Sebelum melatih kecerdasan buatan, kita mengumpulkan gambar sampah untuk masing-masing jenis kategori. 
    Total dataset berjumlah **4.785 gambar** yang dibagi menjadi **70% data pelatihan**, **15% data validasi**, dan **15% data pengujian**.
    Berikut adalah grafik persebaran jumlah gambar yang digunakan dalam penelitian ini:
    """)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    dist_fig, dist_ax = plt.subplots(figsize=(8, 4))
    dataset_distribution = {
        "Kaca": 1404,
        "Kertas": 1050,
        "Logam": 769,
        "Plastik": 865,
        "Residu": 697
    }
    colors = ["#00ACC1", "#FBC02D", "#757575", "#1E88E5", "#795548"]
    bars = dist_ax.bar(dataset_distribution.keys(), dataset_distribution.values(), color=colors, edgecolor='black', alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        dist_ax.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f'{yval}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    dist_ax.set_ylabel("Jumlah Gambar")
    dist_ax.set_ylim(0, 1600)
    dist_ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(dist_fig)
    
    st.markdown("---")
    
    # Section 2: Tampilkan tabel komparasi model
    st.markdown("### 📋 2. Tabel Perbandingan Kinerja Algoritma")
    st.markdown("""
    Tabel berikut merangkum hasil pengujian ilmiah dari tiga arsitektur model Deep Learning.
    """)
    comp_md_path = os.path.join(VISUALIZATION_DIR, "model_comparison.md")
    if os.path.exists(comp_md_path):
        with open(comp_md_path, 'r') as f:
            st.markdown(f.read())
    else:
        st.warning("Data tabel perbandingan model belum tersedia. Silakan jalankan training dan evaluasi terlebih dahulu.")
        
    st.markdown("---")
    
    # Section 3: Grafik Rekapan Kinerja Algoritma
    st.markdown("### 📈 3. Grafik Rekapan Performa: Akurasi & F1-Score")
    st.markdown("""
    Grafik batang di bawah ini mempermudah kita membandingkan akurasi akhir dan F1-Score (keseimbangan tebakan benar) antara ketiga model:
    """)
    
    compare_fig, compare_ax = plt.subplots(figsize=(9, 4.5))
    compare_data = {
        "Model": ["MobileNetV2", "EfficientNetB0", "EfficientNetB0 + CBAM"],
        "Akurasi (%)": [88.11, 91.56, 90.73],
        "F1-Score (%)": [88.27, 91.61, 90.71]
    }
    df_compare = pd.DataFrame(compare_data)
    df_melted = df_compare.melt(id_vars="Model", value_vars=["Akurasi (%)", "F1-Score (%)"], var_name="Metrik", value_name="Nilai")
    
    sns.barplot(data=df_melted, x="Model", y="Nilai", hue="Metrik", palette="Set2", ax=compare_ax)
    compare_ax.set_ylim(80, 100)
    compare_ax.set_ylabel("Persentase (%)")
    compare_ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for p in compare_ax.patches:
        height = p.get_height()
        if height > 0:
            compare_ax.annotate(f'{height:.2f}%', (p.get_x() + p.get_width() / 2., height + 0.3), ha='center', va='bottom', fontweight='bold', fontsize=8)
            
    st.pyplot(compare_fig)
    st.markdown("---")
    
    # 2. Kurva Training (Accuracy & Loss)
    history_curve_path = os.path.join(VISUALIZATION_DIR, "training_history_curves.png")
    if os.path.exists(history_curve_path):
        st.markdown("### Kurva Pembelajaran (Learning Curves)")
        st.image(history_curve_path, caption="Perbandingan Akurasi & Loss antara Train dan Validation untuk 3 Model", use_container_width=True)
    
    st.markdown("---")
    
    # 3. Confusion Matrix
    st.markdown("### Confusion Matrix per Model")
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    
    with cm_col1:
        path = os.path.join(VISUALIZATION_DIR, "confusion_matrix_mobilenet_v2.png")
        if os.path.exists(path):
            st.image(path, caption="MobileNetV2 (Baseline)", use_container_width=True)
        else:
            st.info("Confusion Matrix MobileNetV2 belum siap.")
            
    with cm_col2:
        path = os.path.join(VISUALIZATION_DIR, "confusion_matrix_efficientnet_b0.png")
        if os.path.exists(path):
            st.image(path, caption="EfficientNetB0 (SOTA)", use_container_width=True)
        else:
            st.info("Confusion Matrix EfficientNetB0 belum siap.")
            
    with cm_col3:
        path = os.path.join(VISUALIZATION_DIR, "confusion_matrix_efficientnet_b0_cbam.png")
        if os.path.exists(path):
            st.image(path, caption="EfficientNetB0 + CBAM (Innovative)", use_container_width=True)
        else:
            st.info("Confusion Matrix CBAM belum siap.")
            
    st.markdown("---")
    
    # 4. ROC & Precision-Recall Curves
    st.markdown("### Kurva ROC-AUC & Precision-Recall")
    curve_col1, curve_col2 = st.columns(2)
    
    with curve_col1:
        path = os.path.join(VISUALIZATION_DIR, "roc_curves_comparison.png")
        if os.path.exists(path):
            st.image(path, caption="Perbandingan Kurva ROC & Nilai AUC", use_container_width=True)
        else:
            st.info("Kurva ROC belum siap.")
            
    with curve_col2:
        path = os.path.join(VISUALIZATION_DIR, "precision_recall_curves.png")
        if os.path.exists(path):
            st.image(path, caption="Kurva Precision-Recall Model Terbaik (EfficientNetB0 + CBAM)", use_container_width=True)
        else:
            st.info("Kurva PR belum siap.")
 
# ----------------- PAGE 3: INTERPRETASI GRAD-CAM -----------------
elif page == "Interpretasi Grad-CAM (XAI)":
    st.markdown("<h1 class='main-title'>Explainable AI (XAI) - Grad-CAM</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Visualisasi Bagian Gambar yang Menjadi Fokus Model saat Mengambil Keputusan Klasifikasi</p>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Apa itu Grad-CAM?
    **Grad-CAM (Gradient-weighted Class Activation Mapping)** adalah teknik AI Terbuka (Explainable AI) yang menggunakan gradien dari target kelas 
    yang mengalir ke layer konvolusional terakhir. Teknik ini menghasilkan heatmap kasar yang menyoroti area penting di dalam gambar yang paling 
    berkontribusi terhadap hasil prediksi klasifikasi kelas sampah.
    
    Pada penelitian ini:
    - **MobileNetV2** divisualisasikan pada layer `out_relu`.
    - **EfficientNetB0** divisualisasikan pada layer `top_activation`.
    - **EfficientNetB0 + CBAM** divisualisasikan langsung pada layer `spatial_attention` untuk melihat efek fokus dari mekanisme CBAM Attention.
    """)
    
    st.markdown("---")
    
    st.markdown("###  Galeri Perbandingan Visualisasi Grad-CAM")
    
    cls_to_show = st.selectbox(
        "Pilih Kategori Sampah untuk Menampilkan Visualisasi Grad-CAM:",
        ["Kaca", "Kertas", "Logam", "Plastik", "Residu"]
    )
    
    img_name = f"gradcam_comparison_{cls_to_show.lower()}.png"
    img_path = os.path.join(VISUALIZATION_DIR, img_name)
    
    if os.path.exists(img_path):
        st.image(img_path, caption=f"Perbandingan Grad-CAM untuk Kelas {cls_to_show} (Kiri ke Kanan: Original, MobileNetV2, EfficientNetB0, EfficientNetB0+CBAM)", use_container_width=True)
    else:
        st.info(f"Visualisasi Grad-CAM untuk kelas {cls_to_show} belum di-generate. Gambar perbandingan akan tersedia setelah model selesai dievaluasi.")

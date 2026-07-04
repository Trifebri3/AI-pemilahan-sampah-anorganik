# Panduan Deployment AI Smart Bin (CloudPanel & GitHub)

Dokumen ini menjelaskan langkah-langkah untuk melakukan publikasi kode ke GitHub dan mendeploy aplikasi **AI Smart Bin** (Streamlit Dashboard & Flask API) pada server Linux menggunakan **CloudPanel**.

---

## Bagian 1: Publikasi Project ke GitHub

### 1. Inisialisasi Git Lokal
Buka terminal (Git Bash atau PowerShell) di folder project Anda (`c:\web project\JULI 2027\AI SAMPAH`), lalu jalankan perintah berikut:
```bash
git init
```

### 2. Memeriksa Berkas yang Diabaikan (.gitignore)
Berkas `.gitignore` sudah dibuat untuk mencegah file besar terunggah ke GitHub (seperti folder virtual environment `.venv/` dan dataset citra). Anda bisa memeriksa statusnya dengan:
```bash
git status
```

### 3. Membuat Commit Pertama
Tambahkan berkas dan lakukan commit:
```bash
git add .
git commit -m "Initial commit: AI Smart Bin dengan model EfficientNetB0+CBAM dan Webcam otomatis"
```

### 4. Menghubungkan ke GitHub
1. Buka [GitHub](https://github.com/) dan buat repository baru bernama `ai-smart-bin`.
2. Salin URL repository Anda (contoh: `https://github.com/Trifebri3/AI-pemilahan-sampah-anorganik.git`).
3. Jalankan perintah berikut untuk menghubungkan dan mengunggah kode:
```bash
git branch -M main
git remote add origin https://github.com/Trifebri3/AI-pemilahan-sampah-anorganik.git
git push -u origin main
```

---

## Bagian 2: Deployment pada CloudPanel (Ubuntu Server)

CloudPanel menyediakan fitur **Python Site** yang berjalan di belakang reverse-proxy Nginx.

### Arsitektur Deployment:
1. **Streamlit Dashboard (Frontend & Simulasi):** Berjalan di port `8090` (diakses pengguna melalui browser).
2. **Flask API Server (Backend IoT):** Berjalan di port `5000` (menerima request klasifikasi gambar dari hardware ESP32/Raspberry Pi).

---

### Langkah 1: Setup Situs Python Baru di CloudPanel

1. Masuk ke **CloudPanel Admin Area**.
2. Klik **Add Site** > **Create a Python Site**.
3. Isi detail berikut:
   * **Domain Name:** `www.domainanda.com` (atau subdomain seperti `dashboard.domainanda.com`)
   * **Python Version:** `Python 3.12`
   * **App Port:** `8090` (port default untuk aplikasi Streamlit kita)
   * **Site User:** `site-user`
4. Klik **Create** untuk mengonfigurasi situs.

---

### Langkah 2: Kloning Kode dan Persiapan Berkas di Server

1. Masuk ke server Anda via SSH menggunakan user situs (`site-user`):
   ```bash
   ssh site-user@ip-server-anda
   ```
2. Pindah ke direktori aplikasi:
   ```bash
   cd htdocs/www.domainanda.com/
   ```
3. Kloning repository GitHub Anda ke folder saat ini (atau upload file via SFTP):
   ```bash
   git clone https://github.com/username/ai-smart-bin.git .
   ```
4. Pastikan file model terbaik Anda (`models/efficientnet_b0_cbam_best.keras` dan file `.tflite`) berada di folder `models/` di server.

---

### Langkah 3: Instalasi Dependencies di Virtual Environment Server

CloudPanel secara otomatis membuat virtual environment di folder `/home/site-user/htdocs/www.domainanda.com/venv`. Aktifkan venv tersebut dan instal dependensi:

```bash
# Aktifkan virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Instal dependensi dari requirements.txt
pip install -r requirements.txt
```

---

### Langkah 4: Membuat Daemon Systemd untuk Streamlit Dashboard (Port 8090)

Agar aplikasi Streamlit terus berjalan di latar belakang (background) dan otomatis restart jika server reboot, buatlah service systemd:

1. Buat file service baru sebagai root/sudo user:
   ```bash
   sudo nano /etc/systemd/system/smartbin-dashboard.service
   ```
2. Tempel konfigurasi berikut (sesuaikan username dan path jika berbeda):
   ```ini
   [Unit]
   Description=AI Smart Bin Streamlit Dashboard
   After=network.target

   [Service]
   User=site-user
   WorkingDirectory=/home/site-user/htdocs/www.domainanda.com
   ExecStart=/home/site-user/htdocs/www.domainanda.com/venv/bin/streamlit run dashboard/app.py --server.port 8090 --server.address 0.0.0.0 --browser.gatherUsageStats=false
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
3. Simpan dan keluar (Ctrl+O, Enter, Ctrl+X).
4. Aktifkan dan jalankan service tersebut:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable smartbin-dashboard
   sudo systemctl start smartbin-dashboard
   ```

---

### Langkah 5: Membuat Daemon Systemd untuk Flask API (Port 5000)

Jika Anda menghubungkan alat IoT (ESP32/Raspberry Pi) ke server untuk pemilahan otomatis secara fisik, Anda perlu menjalankan Flask API di port `5000`:

1. Buat file service Flask:
   ```bash
   sudo nano /etc/systemd/system/smartbin-api.service
   ```
2. Tempel konfigurasi berikut:
   ```ini
   [Unit]
   Description=AI Smart Bin Flask API Server
   After=network.target

   [Service]
   User=site-user
   WorkingDirectory=/home/site-user/htdocs/www.domainanda.com
   ExecStart=/home/site-user/htdocs/www.domainanda.com/venv/bin/python -m api.server
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
3. Simpan, aktifkan, dan jalankan service API:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable smartbin-api
   sudo systemctl start smartbin-api
   ```

---

### Langkah 6: Verifikasi Status Layanan

Jalankan perintah berikut untuk memastikan kedua backend berjalan dengan status **active (running)**:
```bash
sudo systemctl status smartbin-dashboard
sudo systemctl status smartbin-api
```

Aplikasi Streamlit kini dapat diakses secara publik melalui domain `www.domainanda.com` (CloudPanel Nginx mengarahkan traffic eksternal secara otomatis ke port internal `8090`), dan API IoT dapat diakses pada `http://ip-server-anda:5000/predict`.

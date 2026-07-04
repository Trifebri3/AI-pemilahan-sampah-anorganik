#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ESP32Servo.h> // Pastikan library ESP32Servo terinstal di Arduino IDE

// ==========================================
// KREDENSIAL WIFI & CONFIG SERVER
// ==========================================
const char* ssid = "NAMA_WIFI_ANDA";
const char* password = "PASSWORD_WIFI_ANDA";


// Contoh: "http://192.168.1.100:5000/predict"
const char* serverUrl = "http://IP_SERVER_ANDA:5000/predict";

// ==========================================
// PIN DEFINITIONS UNTUK ESP32-CAM (AI-THINKER)
// ==========================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==========================================
// PIN DEFINITIONS UNTUK SENSOR & ACTUATOR
// ==========================================
#define TRIG_PIN          15  // Pin Trig Sensor Ultrasonik
#define ECHO_PIN          14  // Pin Echo Sensor Ultrasonik
#define SERVO_PIN         13  // Pin Signal Motor Servo

Servo compartmentServo;
const int distanceThreshold = 10; // Jarak sensor untuk memicu kamera (dalam cm)
bool objectDetected = false;

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // Inisialisasi pin Ultrasonik
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Inisialisasi Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  compartmentServo.setPeriodHertz(50); // Servo standar 50Hz
  compartmentServo.attach(SERVO_PIN, 500, 2400); // Servo SG90 min/max pulse width
  compartmentServo.write(90); // Sudut netral/tutup (90 derajat)
  delay(500);

  // 1. Inisialisasi Kamera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Gunakan resolusi menengah (SVGA) agar transfer gambar lebih cepat
  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_CIF;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Mulai Kamera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Inisialisasi kamera gagal dengan error 0x%x", err);
    return;
  }

  // 2. Hubungkan ke WiFi
  WiFi.begin(ssid, password);
  Serial.print("Menghubungkan ke Wi-Fi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi terhubung!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

long readUltrasonicDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH);
  long distance = duration * 0.034 / 2; // Konversi ke cm
  return distance;
}

void loop() {
  // Hubungkan ulang jika WiFi terputus
  if (WiFi.status() != WL_CONNECTED) {
    setup();
  }

  long distance = readUltrasonicDistance();
  Serial.printf("Jarak Objek: %ld cm\n", distance);

  // Jika objek terdeteksi berada di dekat pintu masuk tempat sampah
  if (distance > 0 && distance < distanceThreshold) {
    if (!objectDetected) {
      Serial.println("Sampah terdeteksi! Mengambil gambar...");
      objectDetected = true;
      
      // Ambil foto dan kirim ke server
      captureAndSendPhoto();
      
      // Delay untuk menghindari pemicuan ganda yang terlalu cepat
      delay(5000); 
    }
  } else {
    objectDetected = false;
  }

  delay(200); // Polling ultrasonik setiap 200ms
}

void captureAndSendPhoto() {
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Gagal mengambil gambar dari kamera!");
    return;
  }

  Serial.println("Mengirim gambar ke API Server...");
  
  HTTPClient http;
  http.begin(serverUrl);
  
  // Siapkan header multipart
  String boundary = "----ESP32CAMBoundary";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  
  // Membangun body multipart/form-data secara manual untuk stream buffer JPEG
  String head = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";
  
  uint32_t extraLen = head.length() + tail.length();
  uint32_t totalLen = fb->len + extraLen;
  
  // Alokasikan buffer payload
  uint8_t * reqBuffer = (uint8_t *)malloc(totalLen);
  if (!reqBuffer) {
    Serial.println("Gagal mengalokasikan memori untuk request payload!");
    esp_camera_fb_return(fb);
    http.end();
    return;
  }
  
  // Salin bagian header, data gambar, dan footer ke buffer request
  memcpy(reqBuffer, head.c_str(), head.length());
  memcpy(reqBuffer + head.length(), fb->buf, fb->len);
  memcpy(reqBuffer + head.length() + fb->len, tail.c_str(), tail.length());
  
  // Kirim POST Request
  int httpResponseCode = http.POST(reqBuffer, totalLen);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("Response Code: %d\n", httpResponseCode);
    Serial.println("Response: " + response);
    
    // Parse JSON response sederhana untuk mengambil sudut servo
    // Format JSON respon server: {"category":"Residu", "servo_angle":180, ...}
    int angleIndex = response.indexOf("\"servo_angle\":");
    if (angleIndex != -1) {
      int start = angleIndex + 14;
      int end = response.indexOf(",", start);
      if (end == -1) {
        end = response.indexOf("}", start);
      }
      String angleStr = response.substring(start, end);
      int targetAngle = angleStr.toInt();
      
      Serial.printf("Menggerakkan servo ke sudut: %d derajat\n", targetAngle);
      
      // Gerakkan servo untuk mengarahkan sampah ke wadah yang tepat
      compartmentServo.write(targetAngle);
      delay(3000); // Beri waktu 3 detik agar sampah masuk
      
      // Kembalikan servo ke posisi awal/tutup
      Serial.println("Menutup wadah kembali...");
      compartmentServo.write(90);
    }
  } else {
    Serial.printf("Koneksi gagal, error: %s\n", http.errorToString(httpResponseCode).c_str());
  }
  
  // Bersihkan memory buffer
  free(reqBuffer);
  esp_camera_fb_return(fb);
  http.end();
}

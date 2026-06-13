# 🏢 SadarHR - Attrition Predictor

**End-to-End Employee Retention Analytics Platform**  
🚀 **Live Demo:** [https://sadarhr.vercel.app/](https://sadarhr.vercel.app/)  
📊 **Pitch Deck:** [Buka Pitch Deck Presentasi (Google Drive)](https://drive.google.com/file/d/1TuMHthhftTpuhLbTHyG4pIdiLYn86ueq/view?usp=sharing)

*Solusi cerdas berbasis Artificial Intelligence untuk mengidentifikasi potensi pengunduran diri (attrition) karyawan dengan arsitektur serverless yang mengedepankan keamanan dan privasi data.*

---

## 🎯 Gambaran Umum Proyek

Pergantian karyawan (turnover) adalah tantangan strategis yang berdampak langsung pada produktivitas dan finansial perusahaan. Proyek ini menghadirkan sebuah solusi analitik yang komprehensif, cepat, dan presisi tinggi dengan menggunakan pendekatan **Serverless Machine Learning**.

Sistem ini tidak hanya menyediakan pipeline *Machine Learning* yang kuat (menggunakan XGBoost dan TensorFlow/Keras) untuk keperluan batch processing dan kompetisi data, tetapi juga menyajikan **Aplikasi Web Interaktif** yang memanfaatkan TensorFlow.js. Model AI dieksekusi secara instan di dalam peramban web (*browser*) pengguna (Client-Side), memastikan **100% Privasi Data** tanpa perlu mengirimkan informasi sensitif karyawan ke server eksternal mana pun.

---

## 🏗️ Arsitektur Solusi (End-to-End)

Pengembangan solusi ini dibagi menjadi dua kapabilitas utama:

1. **Kompetisi & Batch Pipeline (XGBoost)**
   - Menggunakan model *Gradient Boosting* (XGBoost) dengan validasi silang (Stratified 5-Fold CV) yang secara khusus dioptimalkan untuk metrik **F1-Score**.
   - Penanganan otomatis untuk ketidakseimbangan kelas (*class imbalance*) dan penanganan nilai kosong (*missing values imputation*) menggunakan *Mode* (data kategorikal) dan *Median* (data numerik).
   - Menghasilkan file prediksi akhir `HR_sample_submission.csv` yang sudah siap diserahkan kepada dewan juri kompetisi.

2. **Serverless AI Web Application (TensorFlow.js + React)**
   - Model **Deep Learning (Neural Network)** dibangun dengan Keras/TensorFlow dan diekspor menggunakan format yang dioptimalkan untuk web.
   - **Front-end Modern:** Dibangun menggunakan React, TypeScript, Vite, dan Tailwind CSS dengan desain antarmuka (*UI/UX*) yang premium, responsif, dan *mobile-friendly*.
   - **TensorFlow.js:** Memungkinkan inferensi seketika di *browser*. Sangat cocok untuk enterprise yang memiliki regulasi privasi dan tata kelola data yang sangat ketat.

---

## 📂 Struktur Direktori Proyek

Demi kerapian dan standarisasi tingkat *enterprise*, kami telah mengkategorikan *source code* ke dalam struktur berikut:

```text
📁 / (Root Directory)
├── 📁 machine-learning/               # Ekosistem Data Science & Model Training
│   ├── competition_pipeline.py      # Script pembuat 'HR_sample_submission.csv' menggunakan XGBoost
│   ├── train_and_convert.py         # Script training Neural Network & export ke model TFJS
│   └── *.csv                        # Dataset mentah (train, test, sample submission)
├── 📁 web-app/                        # Source Code Aplikasi Web (Front-End Serverless AI)
│   ├── public/model/                # Model TFJS yang sudah dikonversi (model.json, .bin)
│   ├── src/                         # Logika utama UI React dan integrasi '@tensorflow/tfjs'
│   └── package.json                 # Konfigurasi dependensi Node.js
├── HR_sample_submission.csv         # [HASIL AKHIR] File hasil prediksi untuk di-submit
└── README.md                        # Dokumentasi standar teknis ini
```

---

## 🚀 Panduan Eksekusi (How to Run)

### 1. Menjalankan Aplikasi Web AI (Prediksi Real-time)
Aplikasi ini tidak memerlukan konfigurasi server atau backend *database*. Anda hanya memerlukan Node.js untuk menjalankan server pengembangan lokal.

```bash
# Masuk ke direktori web-app
cd web-app

# Lakukan instalasi dependensi
npm install

# Jalankan development server
npm run dev
```
Setelah server berjalan, buka peramban Anda dan akses **http://localhost:5173**. 
Desain sudah disesuaikan agar **Multiplatform** (tampil sempurna di Laptop/Desktop maupun Perangkat Mobile/HP).

### 2. Melatih Ulang Model (Retraining Pipeline)
Bila diperlukan untuk memperbarui model atau mengevaluasi *pipeline*, Anda dapat mengeksekusi *script* Python yang tersedia di dalam direktori `machine-learning`.

```bash
cd machine-learning

# Menjalankan pipeline kompetisi (XGBoost) -> menghasilkan HR_sample_submission.csv
python competition_pipeline.py

# Melatih ulang Neural Network dan melakukan konversi ke TFJS -> update web-app/public/model/
python train_and_convert.py
```
*(Catatan: Dibutuhkan Python 3.9+ beserta library standar seperti pandas, scikit-learn, xgboost, tensorflow, dan tensorflowjs).*

---

## 🛡️ Komitmen Keamanan & Kepatuhan Data

Kami sangat memahami bahwa data sumber daya manusia bersifat konfidensial. Oleh karena itu, pendekatan **Edge-Computing** melalui TensorFlow.js di peramban menjamin bahwa tidak ada satu pun *payload* yang meninggalkan jaringan komputer perusahaan saat HRD menggunakan platform ini untuk menganalisis retensi karyawan.

---
*Didesain dan dikembangkan sebagai solusi cerdas berbasis AI yang siap untuk diintegrasikan di tingkat Enterprise.*

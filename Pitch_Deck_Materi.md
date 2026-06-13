---
title: "HR Analytics Attrition Predictor"
author: "Tim Inovator"
date: "Hackathon Pitch Deck"
---

# SLIDE 1: Judul
**HR Analytics Attrition Predictor**
*Mengubah Data Karyawan Menjadi Wawasan Retensi Proaktif Melalui Serverless AI*

---

# SLIDE 2: Problem Statement (Latar Belakang)
**Masalah Turnover Karyawan yang Mahal**
- Pengunduran diri sukarela (*voluntary attrition*) merugikan perusahaan miliaran rupiah setiap tahunnya (biaya rekrutmen ulang, pelatihan, dan hilangnya *knowledge*).
- Metode tradisional seperti *Exit Interview* atau *Survei Tahunan* sifatnya terlalu terlambat (reaktif). Saat HRD menyadari ketidakpuasan, karyawan sudah berada di ambang pintu keluar.
- Sinyal-sinyal ketidakpuasan seringkali kompleks, non-linear, dan sulit dibaca hanya dari sekadar melihat riwayat promosi atau gaji.

---

# SLIDE 3: Solusi Kami (Proactive Analytics)
**Dari Reaktif Menjadi Prediktif**
- Kami membangun platform analitik yang dapat mengkalkulasi probabilitas seorang karyawan untuk *resign* secara langsung (real-time).
- Berdasarkan 34 titik data (demografi, evaluasi performa, tingkat keterlibatan, jarak ke kantor, dll).
- Memberikan peringatan dini (*early warning*) bagi divisi HR untuk mengambil tindakan retensi yang tepat *sebelum* surat pengunduran diri diajukan.

---

# SLIDE 4: End-to-End ML Pipeline (Akurasi Tinggi)
**Arsitektur Model Machine Learning Utama**
- Algoritma Utama (Kompetisi): *Gradient Boosting* (XGBoost) dengan *Stratified 5-Fold Cross-Validation*.
- Menggunakan parameter `scale_pos_weight` untuk mengatasi ketidakseimbangan kelas (*imbalanced data*), karena jumlah karyawan yang resign jauh lebih sedikit daripada yang bertahan.
- Proses pra-pemrosesan (*imputation* otomatis untuk nilai kosong: *Mode* untuk kategori, *Median* untuk numerik) dan normalisasi standar.
- Hasil: Model yang kokoh dan memiliki *F1-Score* tinggi untuk diserahkan ke *leaderboard* kompetisi.

---

# SLIDE 5: Inovasi Teknologi - Serverless AI
**Deep Learning di Sisi Klien (TensorFlow.js)**
- Masalah: Banyak perusahaan enggan mengadopsi AI pihak ketiga karena takut data rahasia karyawannya bocor ke server luar.
- Inovasi Kami: Kami mengekspor model *Neural Network* (Keras) ke dalam format **TensorFlow.js**.
- Artinya, proses inferensi/prediksi AI dieksekusi 100% menggunakan *resource* *browser* (Client-Side). Tidak ada infrastruktur *backend* berbasis *cloud* yang menyimpan atau memproses data input form.

---

# SLIDE 6: Privasi Data Mutlak (Keunggulan Kompetitif)
**Keamanan Setara Enterprise**
- **Zero Data Transfer**: Saat tim HR mengetikkan profil karyawan dan memprediksi, tidak ada satu bait data pun yang dikirim melalui internet.
- Sesuai dengan hukum privasi data yang ketat.
- **Ultra Cepat**: Karena tidak membutuhkan koneksi API ke server luar, hasil prediksi (*probability circle*) muncul dalam hitungan milidetik tanpa latensi.

---

# SLIDE 7: Desain & Kemudahan Penggunaan
**Antarmuka Premium Merah-Putih**
- *Landing Page* interaktif dan dasbor prediksi yang didesain secara modern (SaaS-grade).
- Responsif di seluruh perangkat (Multiplatform: Desktop, Tablet, dan Mobile/HP).
- Layar prediksi menampilkan "Probabilitas Attrition" menggunakan representasi visual persentase berwarna (Hijau untuk aman, Merah untuk waspada), dilengkapi kesimpulan rekomendasi tindakan.

---

# SLIDE 8: Dampak Bisnis & Penutup
**Mengapa Solusi Kami Penting untuk Perusahaan?**
- **Efisiensi Finansial**: Menurunkan anggaran rekrutmen hingga 30% dengan melakukan retensi pada target yang tepat.
- **Produktivitas Berkelanjutan**: Menjaga dinamika tim dengan mencegah kehilangan talenta terbaik (Top Performers).
- Solusi siap pakai (*production-ready*), ultra-aman, tanpa biaya infrastruktur server AI (*Serverless*).

*Terima kasih.*

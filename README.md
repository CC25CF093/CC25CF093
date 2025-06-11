# 🧠 Machine Learning - Real-time Sign Language Detection

Model ini dikembangkan untuk mengenali bahasa isyarat tangan secara **real-time** melalui kamera laptop. Fokus utama pengembangan adalah pada efisiensi, kecepatan inferensi, dan kompatibilitas di browser.

---

## 📦 Dataset

Dataset yang digunakan bisa diakses melalui Google Drive berikut:  
👉 [Sign Language Dataset - Google Drive](https://drive.google.com/drive/folders/1g88HrZxbEy0bHLLBPuzLSj6SVXkDZqfp?usp=sharing)

Dataset ini berisi gambar gesture tangan dalam berbagai posisi dan label, yang telah diproses dan dibersihkan untuk keperluan pelatihan model.

---

## 🧠 Model Arsitektur

Model ini dibangun menggunakan **MobileNetV2** sebagai feature extractor, karena:

- Ringan dan cepat, cocok untuk aplikasi berbasis web.
- Performa bagus untuk klasifikasi gambar sederhana.
- Bisa langsung diintegrasikan ke **TensorFlow.js** tanpa konversi berat.

Model dilatih dengan TensorFlow dan dikonversi ke **TensorFlow.js format (.json + .bin)** agar bisa langsung digunakan di browser.

---

# ♻️ Proyek Klasifikasi Gambar: Garbage Classification (12 classes) ♻️

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" width="60" alt="TensorFlow Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/2560px-NumPy_logo_2020.svg.png" width="110" alt="NumPy Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/1200px-Created_with_Matplotlib-logo.svg.png" width="60" alt="Matplotlib Logo">
  <img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" width="160" alt="Seaborn Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width="90" alt="Scikit-learn Logo">
</p>

## Ringkasan Proyek

Proyek ini bertujuan untuk mengatasi tantangan pengelolaan sampah modern dengan membangun sebuah model **Deep Learning** yang canggih. Model ini mampu mengklasifikasikan gambar sampah ke dalam **12 kategori berbeda** secara otomatis, mulai dari kardus, kaca, hingga berbagai jenis sampah lainnya.

Dengan memanfaatkan pendekatan **Hybrid Transfer Learning** menggunakan arsitektur **MobileNetV2**, proyek ini berhasil mencapai **akurasi validasi di atas 95%**, menghasilkan sebuah solusi yang kuat dan efisien untuk sistem pemilahan sampah otomatis. Model akhir juga diekspor ke dalam tiga format standar industri: `SavedModel`, `TFLite`, dan `TensorFlow.js` untuk fleksibilitas penerapan di berbagai platform.

---

## Dataset & Kategori Sampah

- **Nama Dataset**: [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Total Gambar**: >11.000 gambar
- **Jumlah Kategori**: 12

Model ini dilatih untuk mengenali 12 kategori sampah berikut:

- `battery` (baterai)
- `biological` (sampah organik)
- `brown-glass` (kaca cokelat)
- `cardboard` (kardus)
- `clothes` (pakaian)
- `green-glass` (kaca hijau)
- `metal` (logam)
- `paper` (kertas)
- `plastic` (plastik)
- `shoes` (sepatu)
- `trash` (sampah umum)
- `white-glass` (kaca putih)

---

## 🧠 Arsitektur Model & Metodologi

Untuk mencapai akurasi tinggi, proyek ini mengimplementasikan beberapa teknik canggih dalam perancangan dan pelatihan model.

### 1. Pendekatan Hybrid Transfer Learning

Kami tidak membangun model dari nol. Sebaliknya, kami menggunakan pendekatan **Hybrid Transfer Learning** yang cerdas:

- **Base Model**: Menggunakan **MobileNetV2** yang telah dilatih pada dataset raksasa ImageNet sebagai pengekstrak fitur dasar. Ini memberikan "pengetahuan" awal tentang bentuk, tekstur, dan pola visual secara umum.
- **Lapisan Kustom**: Di atas MobileNetV2, kami menambahkan **blok konvolusi kustom** (`Conv2D` -> `MaxPooling2D` -> `BatchNormalization`). Tujuannya adalah untuk mempelajari fitur-fitur yang lebih spesifik dan relevan dengan dataset sampah kita, yang tidak ada di ImageNet.

### 2. Struktur Model Rinci

```
Input Layer (224, 224, 3)
       |
Data Augmentation Layers (RandomFlip, RandomRotation, RandomZoom)
       |
Base Model: MobileNetV2 (sebagian lapisan di-unfreeze saat fine-tuning)
       |
Custom Convolutional Block:
  -> Conv2D (64 filter, kernel 3x3, aktivasi ReLU)
  -> MaxPooling2D (2x2)
  -> BatchNormalization
       |
Classifier Head:
  -> GlobalAveragePooling2D
  -> Dropout (rate 0.5 untuk mencegah overfitting)
  -> Dense (12 neuron, aktivasi Softmax untuk klasifikasi multi-kelas)
```

### 3. Strategi Pelatihan 3 Fase

Pelatihan tidak dilakukan sekali jalan, melainkan melalui 3 fase strategis untuk stabilitas dan performa maksimal:

1.  **Fase 1: Warm-up**: Hanya melatih lapisan kustom dan _classifier head_. Seluruh _base model_ MobileNetV2 dibekukan (`trainable=False`). Tujuannya agar _head_ yang baru belajar beradaptasi tanpa merusak bobot berharga dari _base model_.
2.  **Fase 2: Partial Fine-tuning**: Membuka (unfreeze) beberapa lapisan teratas dari MobileNetV2 dan melatihnya kembali bersama _head_ dengan _learning rate_ yang rendah. Ini memungkinkan model untuk menyesuaikan fitur tingkat tinggi (lebih abstrak) dengan dataset sampah.
3.  **Fase 3: Full Fine-tuning**: Melatih hampir seluruh model dengan _learning rate_ yang sangat kecil untuk penyesuaian akhir yang halus dan konvergensi optimal.

### 4. Callback Cerdas untuk Pelatihan Efisien

Untuk mengontrol proses pelatihan, beberapa _callback_ cerdas digunakan:

- `ModelCheckpoint`: Menyimpan model terbaik secara otomatis selama pelatihan.
- `EarlyStopping`: Menghentikan pelatihan jika tidak ada peningkatan performa pada data validasi untuk mencegah pemborosan waktu dan _overfitting_.
- `ReduceLROnPlateau`: Mengurangi _learning rate_ secara otomatis ketika performa model stagnan.
- `ImprovedTargetCallback`: _Callback_ kustom yang akan menghentikan pelatihan **segera** setelah target akurasi validasi **>95%** tercapai, memastikan kriteria proyek terpenuhi secara efisien.

---

## 📊 Hasil & Performa Model

Setelah melalui strategi pelatihan 3 fase, model menunjukkan performa yang sangat baik dan berhasil memenuhi target proyek.

- **Akurasi Training Akhir**: **~98.86%**
- **Akurasi Validasi Akhir**: **~95.01%**

![Grafik Akurasi & Loss](https://i.imgur.com/your-accuracy-loss-plot-image.png)
_(Catatan: Harap ganti tautan gambar ini dengan screenshot plot akurasi dan loss dari notebook Anda)_

Plot di atas menunjukkan bahwa kurva akurasi dan loss untuk data latih dan validasi bergerak secara harmonis. Kurva validasi naik secara konsisten dan mendatar di tingkat performa yang tinggi, menandakan model berhasil melakukan generalisasi dengan baik tanpa mengalami _overfitting_ yang signifikan.

---

## 🚀 Hasil Akhir: Model Siap Pakai

Sebagai hasil akhir, model yang telah dilatih secara optimal diekspor ke dalam tiga format berbeda untuk mendukung berbagai skenario penerapan:

1.  **SavedModel (`/saved_model`)**: Format standar TensorFlow, ideal untuk deployment di server atau cloud.
2.  **TensorFlow Lite (`/tflite`)**: Format yang ringan dan teroptimalkan, dirancang khusus untuk aplikasi mobile (Android/iOS) dan perangkat IoT.
3.  **TensorFlow.js (`/tfjs_model`)**: Format untuk menjalankan model secara langsung di browser web, memungkinkan aplikasi AI berbasis web yang interaktif.

---

## 📁 Struktur Direktori Repositori

Struktur berkas dalam repositori ini diatur sebagai berikut untuk kemudahan navigasi dan penggunaan.

```
submission/
├── tfjs_model/                # Model untuk deployment web
│   ├── group1-shard1of1.bin
│   └── model.json
├── tflite/                    # Model untuk deployment mobile/IoT
│   ├── model.tflite
│   └── labels.txt
├── saved_model/               # Model format standar TensorFlow
│   ├── saved_model.pb
│   └── variables/
├── garbage-classification.ipynb # Notebook utama berisi semua proses
├── README.md                    # Dokumentasi proyek (file ini)
└── requirements.txt             # Daftar dependensi Python
```

---

## ⚙️ Instalasi & Penggunaan

Untuk menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut:

1.  **Clone Repositori**

    ```bash
    git clone [https://github.com/NAMA_USER/NAMA_REPO.git](https://github.com/NAMA_USER/NAMA_REPO.git)
    cd NAMA_REPO
    ```

2.  **Buat Lingkungan Virtual (Direkomendasikan)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instal Dependensi**
    Instal semua _library_ yang dibutuhkan menggunakan file `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Jupyter Notebook**
    Buka dan jalankan file `garbage-classification.ipynb` untuk melihat seluruh alur kerja proyek, mulai dari pemuatan data, pelatihan model, hingga evaluasi.
    ```bash
    jupyter notebook garbage-classification.ipynb
    ```

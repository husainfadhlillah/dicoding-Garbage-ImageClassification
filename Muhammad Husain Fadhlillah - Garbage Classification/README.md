# Proyek Klasifikasi Gambar: Garbage Classification (12 classes)

## Detail Proyek
- *Nama:* Muhammad Husain Fadhlillah
- *Email Student:* mc006d5y2343@student.devacademy.id
- *Cohort ID:* MC006D5Y2343

## Detail Model
- **Dataset**: Garbage Classification (11,000 gambar, 12 kelas, resolusi tidak seragam)
- **Arsitektur Utama**: `tf.keras.Sequential`
- **Base Model**: `MobileNetV2` (Transfer Learning, sebagian di-fine-tune)
- **Lapisan Kustom**: 1 blok `Conv2D` + `MaxPooling2D` + `BatchNormalization` setelah *base model*.
- **Callback yang Digunakan**: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `ImprovedTargetCallback` (custom callback untuk akurasi >95%).
- **Strategi Training**: 3 fase (Warm-up, Partial Fine-tuning, Full Fine-tuning).
- **Akurasi Training Akhir**: 98.86%
- **Akurasi Validasi Akhir**: 95.01%
- **Akurasi Test Aktual**: **95.01%** (sesuai kriteria >95% untuk bintang 5)

## Struktur Direktori Submission
submission/
├── tfjs_model/
| ├── group1-shard1of1.bin
| └── model.json
├── tflite_model/
| ├── model.tflite
| └── labels.txt
├── saved_model/
| ├── saved_model.pb
| └── variables/
├── notebook.ipynb
├── README.md
└── requirements.txt
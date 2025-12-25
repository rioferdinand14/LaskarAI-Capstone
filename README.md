# Deteksi Resep Berdasarkan Gambar Makanan

Proyek ini mengimplementasikan model klasifikasi citra makanan menggunakan arsitektur MobileNetV2 dengan teknik transfer learning. Model dilatih untuk mengenali 11 kategori makanan dari gambar.

**Sumber Dataset**
[Indonesian Foods Dataset](https://www.kaggle.com/datasets/raihanrizkidwiputra/indonesian-foods-dataset)

## Cara Menjalankan

**Clone repository dan masuk ke direktori project**
   ```bash
   git clone https://github.com/rioferdinand14/LaskarAI-Capstone
   cd LaskarAI-Capstone
   ```

**Buat dan aktifkan virtual environment**
   ```bash
    python -m venv venv
    source venv/bin/activate  # atau `venv\Scripts\activate` di Windows
    pip install -r requirements.txt
   ```

**Buka dan jalankan notebook**
Anda hanya perlu menjalankan cell **import libary**, **load dataset**, dan **inference** di `notebook.ipynb`
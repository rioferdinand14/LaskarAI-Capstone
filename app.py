import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os

st.set_page_config(
    page_title="Deteksi Resep Makanan",
    page_icon="🍳",
    layout="wide"
)

DAFTAR_MAKANAN_DISPLAY = [
    "Ayam Goreng", "Ayam Pop", "Bakso", "Dendeng Batokok", 
    "Gado-gado", "Grontol", "Gudeg", "Gulai Ikan", 
    "Gulai Tambusu", "Gulai Tunjang", "Kue Ape", "Kue Bika Ambon", 
    "Kue Cenil", "Kue Dadar Gulung", "Kue Gethuk Lindri", "Kue Kastangel", 
    "Kue Klepon", "Kue Lapis", "Kue Lemper", "Kue Lumpur", 
    "Kue Nagasari", "Kue Pastel", "Kue Putri Salju", "Kue Risoles", 
    "Lanting", "Lumpia", "Putu Ayu", "Rendang", 
    "Serabi Solo", "Telur Balado", "Telur Dadar", "Wajik"
]

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3480/3480823.png", width=100)
    st.title("👨‍🍳 AI Chef")
    st.write("Aplikasi ini menggunakan Deep Learning untuk mengenali gambar makanan Indonesia.")
    
    st.divider()
    
    st.subheader("📋 Daftar Kemampuan")
    st.write(f"Model dilatih untuk mengenali **{len(DAFTAR_MAKANAN_DISPLAY)} jenis** makanan:")
    
    search_query = st.text_input("Cari menu di list...", "").lower()
    
    filtered_list = [m for m in DAFTAR_MAKANAN_DISPLAY if search_query in m.lower()]
    
    st.dataframe(
        pd.DataFrame(filtered_list, columns=["Nama Menu"]),
        hide_index=True,
        use_container_width=True,
        height=300 
    )
    
    st.info("Pastikan upload gambar yang jelas dan fokus pada makanannya.")

@st.cache_resource
def load_model_inference():
    """
    Memuat model menggunakan Low-Level SavedModel API.
    Sama persis dengan logika kode inference manual Anda.
    """
    BASE_DIR = "." 
    SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model_tl")
    
    if not os.path.exists(SAVED_MODEL_DIR):
        return None, None, None, "Folder 'saved_model_tl' tidak ditemukan."

    try:
        loaded = tf.saved_model.load(SAVED_MODEL_DIR)
        infer = loaded.signatures["serving_default"]
        
        _, input_kwargs = infer.structured_input_signature
        input_key = list(input_kwargs.keys())[0]
        output_key = list(infer.structured_outputs.keys())[0]
        
        return infer, input_key, output_key, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_csv_data():
    try:
        df = pd.read_excel('Dataset_resep.xlsx', engine='openpyxl')
        
        class_names = df['Nama'].values 
        return df, class_names, None
    except Exception as e:
        return None, None, str(e)

infer_func, input_key, output_key, err_model = load_model_inference()
df_resep, class_names, err_csv = load_csv_data()

def run_prediction(image_input, infer_func, input_key, output_key, class_names):

    size = (224, 224)
    image = ImageOps.fit(image_input, size, Image.Resampling.LANCZOS)
    
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    
    batch_input = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    input_tensor = tf.constant(batch_input)
    
    outputs = infer_func(**{input_key: input_tensor})
    preds = outputs[output_key].numpy()[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])
    label = class_names[idx]
    
    return label, confidence

col_upload, col_result = st.columns([1, 1.5])

with col_upload:
    st.title("Deteksi Resep Makanan")
    st.markdown("Upload foto makanan, dapatkan resepnya!")

    if err_model:
        st.error(f"❌ Error Model: {err_model}")
        st.stop()
    if err_csv:
        st.error(f"❌ Error CSV: {err_csv}")
        st.stop()

    file = st.file_uploader("Pilih gambar makanan (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

    if file is not None:
        # Tampilkan Gambar
        image = Image.open(file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        if st.button("🔍 Deteksi Resep", type="primary"):
            with st.spinner('Sedang menganalisis gambar...'):

                with col_result:                               
                                try:                                    
                                    label_result, conf_result = run_prediction(
                                        image, infer_func, input_key, output_key, class_names
                                    )

                                    if conf_result < 0.60:
                                        st.warning("Maaf, gambar kurang jelas atau objek tidak dikenali.")                
                                        
                                    else:
                                        st.success(f"Hasil Deteksi: **{label_result.replace('_', ' ')}**")
                                        st.caption(f"Confidence: {conf_result*100:.2f}%")
                                                                          
                                        row_data = df_resep[df_resep['Nama'] == label_result]
                                        
                                        if not row_data.empty:
                                            data = row_data.iloc[0]
                                            
                                            st.divider()
                                            st.subheader(f"📖 Resep {data['Nama'].replace('_', ' ')}")
                                            
                                            if 'Deskripsi' in data:
                                                st.info(data['Deskripsi'])
                                            
                                            if 'Resep' in data:
                                                st.markdown("### Cara Memasak & Bahan")
                                                st.text_area("Detail Resep", data['Resep'], height=300)
                                            else:
                                                st.warning("Kolom 'Resep' tidak ditemukan di CSV.")
                                        else:
                                            st.warning(f"Resep untuk '{label_result}' belum tersedia di database CSV.")
                                        
                                except Exception as e:
                                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
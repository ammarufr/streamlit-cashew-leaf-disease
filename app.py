import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
from streamlit_option_menu import option_menu

# Konstanta
IMG_SIZE = 224
THRESHOLD = 0.6
CLASS_LABELS = ['Cashew anthracnose', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust', 'non-leaf']

# Load model dengan caching
@st.cache_resource
def load_model_cached():
    return load_model("model_cashew_disease.h5")

model = load_model_cached()

# Sidebar Navigasi
with st.sidebar:
    selected = option_menu(
        menu_title="Navigasi",
        options=["Welcome", "Diagnosis", "Penyakit"],
        icons=["house", "activity", "book"],
        default_index=0
    )

# --- Welcome Page ---
if selected == "Welcome":
    st.markdown("<h1>Selamat Datang di Aplikasi Diagnosis Daun Jambu Mete</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 18px; text-align: justify;'>
        Aplikasi ini membantu mendeteksi penyakit pada daun jambu mete.
        <br><br>
        Upload atau foto daun untuk mendapatkan diagnosis secara cepat dan akurat.
        <br><br>
        Baca informasi tiap penyakit dan cara penanganannya.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

    st.markdown("<h3>Cara Menggunakan Aplikasi</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 16px; line-height: 1.8; text-align: justify;'>
        1. Klik menu <b>Diagnosis</b> di sebelah kiri untuk memulai pemeriksaan.
        <br>
        2. Upload gambar daun jambu mete dari galeri <b>atau</b> ambil foto langsung menggunakan kamera.
        <br>
        3. Sistem akan memproses gambar dan menampilkan hasil diagnosis beserta informasi penyakit.
        <br>
        4. Jika ingin membaca penjelasan lengkap tentang tiap penyakit, klik menu <b>Penyakit</b>.
    </div>
    """, unsafe_allow_html=True)

# --- Diagnosis Page ---
elif selected == "Diagnosis":
    st.title("Diagnosis Penyakit Daun Jambu Mete")

    disease_info = {
        "Cashew anthracnose": {
            "deskripsi": "Penyakit jamur yang menyebabkan bercak cokelat hingga hitam pada daun dan ranting.",
            "penanganan": "- Gunakan fungisida berbahan aktif tembaga.\n- Pangkas bagian yang terinfeksi.\n- Jaga sirkulasi udara tanaman.",
            "gambar": "images/anthracnose.jpg"
        },
        "Cashew leaf miner": {
            "deskripsi": "Hama penggerek daun yang membuat lorong-lorong putih di permukaan daun.",
            "penanganan": "- Gunakan insektisida sistemik.\n- Pangkas daun yang terinfeksi parah.\n- Jaga kebersihan area kebun.",
            "gambar": "images/leaf_miner.jpg"
        },
        "Cashew red rust": {
            "deskripsi": "Penyakit jamur yang menyebabkan bercak karat merah/oranye di permukaan daun.",
            "penanganan": "- Semprotkan fungisida secara berkala.\n- Singkirkan daun-daun terinfeksi.\n- Hindari kelembapan tinggi di sekitar tanaman.",
            "gambar": "images/red_rust.jpg"
        }
    }

    uploaded_file = st.file_uploader("Upload gambar daun atau objek lainnya", type=['jpg', 'png', 'jpeg'])
    camera_image = st.camera_input("Atau ambil gambar langsung")

    input_image = uploaded_file if uploaded_file is not None else camera_image

    if input_image is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(input_image, caption="Gambar yang diproses", use_container_width=True)

        with col2:
            img = PILImage.open(input_image).convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("Mendiagnosis..."):
            pred = model.predict(img_array)
            confidence = float(np.max(pred))
            class_idx = int(np.argmax(pred))
            predicted_class = CLASS_LABELS[class_idx]

            pred = model.predict(img_array)
            confidence = float(np.max(pred))
            class_idx = int(np.argmax(pred))
            predicted_class = CLASS_LABELS[class_idx]

            st.subheader("🔍 Hasil Prediksi:")
            if predicted_class == 'non-leaf' or confidence < THRESHOLD:
                st.error(f"⚠️ Gambar ini kemungkinan **bukan daun jambu mete**.\n\nPrediksi: `{predicted_class}`\nKeyakinan: `{confidence*100:.2f}%`")
            else:
                st.success(f"✅ Gambar dikenali sebagai: `{predicted_class}`\nKeyakinan: `{confidence*100:.2f}%`")

                if predicted_class in disease_info:
                    detail = disease_info[predicted_class]
                    st.image(detail["gambar"], caption=f"Gambar: {predicted_class}", use_container_width=True)
                    st.markdown(f"### ℹ️ Informasi Penyakit: {predicted_class}")
                    st.markdown(f"**Deskripsi:** {detail['deskripsi']}")
                    st.markdown(f"**Penanganan:**\n{detail['penanganan']}")

# --- Informasi Penyakit Page ---
elif selected == "Penyakit":
    st.title("Informasi Penyakit Daun Jambu Mete")

    info = {
        "Cashew anthracnose": {
            "deskripsi": "Penyakit jamur yang menyebabkan bercak cokelat hingga hitam pada daun dan ranting.",
            "penanganan": "- Gunakan fungisida berbahan aktif tembaga.\n- Pangkas bagian yang terinfeksi.\n- Jaga sirkulasi udara tanaman.",
            "gambar": "images/anthracnose.jpg"
        },
        "Cashew leaf miner": {
            "deskripsi": "Hama penggerek daun yang membuat lorong-lorong putih di permukaan daun.",
            "penanganan": "- Gunakan insektisida sistemik.\n- Pangkas daun yang terinfeksi parah.\n- Jaga kebersihan area kebun.",
            "gambar": "images/leaf_miner.jpg"
        },
        "Cashew red rust": {
            "deskripsi": "Penyakit jamur yang menyebabkan bercak karat merah/oranye di permukaan daun.",
            "penanganan": "- Semprotkan fungisida secara berkala.\n- Singkirkan daun-daun terinfeksi.\n- Hindari kelembapan tinggi di sekitar tanaman.",
            "gambar": "images/red_rust.jpg"
        }
    }

    for disease, detail in info.items():
        with st.expander(f"📌 {disease}"):
            st.image(detail["gambar"], caption=f"Gambar: {disease}", use_container_width=True)
            st.markdown(f"**Deskripsi:** {detail['deskripsi']}")
            st.markdown(f"**Penanganan:**\n{detail['penanganan']}")

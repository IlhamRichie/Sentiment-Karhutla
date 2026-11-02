import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Analisis Sentimen Karhutla", layout="centered")

st.title("Analisis Sentimen Opini Publik Karhutla")
st.caption("Klasifikasi sentimen masyarakat terkait isu kebakaran hutan dan lahan berbasis IndoBERT.")

MODEL_REPO = "forti2026/IndoBERT-Sentimen-Karhutla"

@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model=MODEL_REPO)

classifier = load_classifier()

# Contoh kalimat uji
preset_options = [
    "Ketik sendiri...",
    "Kabut asap di Pontianak makin tebal, anak-anak terpaksa libur sekolah dan sesak nafas.",
    "BMKG memantau adanya 18 titik panas di wilayah Kalimantan hari ini.",
    "Apresiasi setinggi-tingginya untuk tim Manggala Agni yang cepat tanggap memadamkan api."
]

selected_preset = st.selectbox("Gunakan contoh kalimat:", preset_options)

default_text = "" if selected_preset == "Ketik sendiri..." else selected_preset
user_input = st.text_area("Teks Komentar:", value=default_text, height=120)

if st.button("Analisis Sentimen", type="primary"):
    if not user_input.strip():
        st.warning("Masukkan teks komentar terlebih dahulu.")
    else:
        with st.spinner("Memproses..."):
            pred = classifier(user_input)[0]
            label = pred["label"]
            score = pred["score"]

        st.subheader("Hasil Prediksi")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Prediksi", value=label)
        with col2:
            st.metric(label="Tingkat Keyakinan", value=f"{score * 100:.2f}%")

        st.progress(float(score))

import streamlit as st
from transformers import pipeline

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen Karhutla", page_icon="🌲", layout="centered")

st.title("🌲🔥 Analisis Sentimen Isu Karhutla")
st.write("Aplikasi NLP klasifikasi opini masyarakat terkait kebakaran hutan dan lahan berbasis **IndoBERT**.")

# Cache model agar loadingnya cepat dan hemat RAM
@st.cache_resource
def muat_model():
    # Ambil model publik yang sudah kamu upload di Hugging Face
    return pipeline("text-classification", model="forti2026/IndoBERT-Sentimen-Karhutla")

with st.spinner("Sedang menyiapkan model IndoBERT di server..."):
    pipe = muat_model()

# Area Input Teks
teks_input = st.text_area(
    "Masukkan Komentar / Opini Publik:",
    placeholder="Contoh: Kabut asap di Pontianak makin tebal, anak-anak tidak bisa sekolah...",
    height=120
)

# Tombol Cepat Contoh
col1, col2, col3 = st.columns(3)
if col1.button("📌 Contoh Negatif"):
    teks_input = "Kabut asap makin parah dan menyesakkan dada, tolong pemerintah tangkap pelakunya!"
if col2.button("📌 Contoh Netral"):
    teks_input = "BMKG melaporkan adanya 18 titik api terpantau lewat satelit hari ini."
if col3.button("📌 Contoh Positif"):
    teks_input = "Terima kasih untuk tim Manggala Agni dan relawan yang sigap padamkan api!"

if st.button("🔍 Analisis Sentimen", type="primary"):
    if not teks_input.strip():
        st.warning("Silakan ketik kalimat komentarnya terlebih dahulu!")
    else:
        with st.spinner("Menganalisis..."):
            hasil = pipe(teks_input)[0]
            label = hasil['label']
            skor = hasil['score']
            
            st.divider()
            st.subheader("Hasil Klasifikasi:")
            
            # Tampilan Badge Sesuai Sentimen
            if label == "Negatif":
                st.error(f"Sentimen Dominan: **{label}** (Keyakinan: {skor*100:.1f}%)")
            elif label == "Positif":
                st.success(f"Sentimen Dominan: **{label}** (Keyakinan: {skor*100:.1f}%)")
            else:
                st.warning(f"Sentimen Dominan: **{label}** (Keyakinan: {skor*100:.1f}%)")
                
            st.progress(float(skor))

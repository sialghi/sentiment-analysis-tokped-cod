import streamlit as st
import pandas as pd
import joblib
import re
import json
import ast
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def inject_ui():
        st.markdown(
                """
                <style>
                    /* Page spacing */
                    .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; }

                    /* Sidebar polish */
                    section[data-testid="stSidebar"] { border-right: 1px solid rgba(15, 23, 42, 0.08); }
                    section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }

                    /* Inputs */
                    textarea, input, [data-baseweb="select"] > div {
                        border-radius: 14px !important;
                    }
                    textarea { line-height: 1.5; }

                    /* Buttons */
                    .stButton > button {
                        border-radius: 999px;
                        padding: 0.65rem 1.05rem;
                        font-weight: 650;
                        border: 1px solid rgba(37, 99, 235, 0.25);
                    }

                    /* Alerts */
                    [data-testid="stAlert"] { border-radius: 16px; }

                    /* Reduce visual noise */
                    header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
                </style>
                """,
                unsafe_allow_html=True,
        )

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia (COD)",
    page_icon="🛍️",
    layout="wide"
)

# --- FUNGSI LOAD ASSETS (CACHED) ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model_svm_tokopedia.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

@st.cache_data
def load_slang_dict():
    try:
        with open('slangwords.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Cara 1: Coba baca sebagai JSON murni
        try:
            slang_dict = json.loads(content)
        except json.JSONDecodeError:
            # Cara 2: Jika gagal, coba baca sebagai struktur data Python (misal kutip satu)
            try:
                slang_dict = ast.literal_eval(content)
            except:
                # Cara 3: Fallback manual jika formatnya baris per baris (opsional)
                slang_dict = {}
                # Logika manual bisa ditambahkan disini jika formatnya benar-benar beda
                
        return slang_dict
        
    except Exception as e:
        st.error(f"Gagal memuat slangwords.txt: {e}")
        return {}
    
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('Data ulasan tokopedia tentang COD.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# --- FUNGSI PREPROCESSING ---
# Inisialisasi Sastrawi di luar fungsi agar tidak dimuat ulang setiap kali
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

factory_stop = StopWordRemoverFactory()
stopword_remover = factory_stop.create_stop_word_remover()

def preprocess_text(text, slang_dict):
    if not isinstance(text, str):
        return ""
    
    # 1. Cleaning & Case Folding
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus angka & simbol
    
    # --- TAMBAHAN PENTING (Handling huruf berulang >3) ---
    # Mengubah "kereeeeen" menjadi "keren"
    text = re.sub(r'(.)\1{2,}', r'\1', text) 
    # -----------------------------------------------------

    # 2. Formalisasi (Slang)
    words = text.split()
    formalized_words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(formalized_words)
    
    # 3. Stopword Removal
    text = stopword_remover.remove(text)
    
    # 4. Stemming
    text = stemmer.stem(text)
    
    return text

# --- MAIN APPLICATION ---
def main():
    inject_ui()

    st.title("Analisis Sentimen Tokopedia - Fitur COD")
    st.caption("Demo klasifikasi sentimen ulasan COD Tokopedia menggunakan **Support Vector Machine (SVM)**.")

    # Load semua asset
    model, vectorizer = load_models()
    slang_dict = load_slang_dict()
    df = load_dataset()

    # Sidebar Menu
    with st.sidebar:
        st.markdown("### Navigasi")
        st.caption("Pilih menu untuk prediksi atau melihat pembagian data.")
    menu = st.sidebar.selectbox(
        "Pilih Menu",
        ["Prediksi Sentimen", "Data Latih", "Data Uji"]
    )

    # Logika Pembagian Data (Split)
    # Kita bagi data secara on-the-fly untuk ditampilkan di menu Data Latih/Uji
    if not df.empty:
        # Asumsi kolom label bernama 'score' atau sentimen perlu dibuat
        # Berdasarkan snippet file csv, ada kolom 'score'. Kita perlu mapping score ke label jika belum ada.
        # Biasanya: 1-2 Negatif, 3 Netral, 4-5 Positif. Atau sesuai notebook Anda.
        
        # Simple mapping untuk demo (sesuaikan dengan logika notebook Anda)
        def map_sentiment(score):
            if score >= 4: return 'Positif'
            elif score <= 2: return 'Negatif'
            else: return 'Netral'
        
        if 'Label' not in df.columns and 'score' in df.columns:
            df['Label'] = df['score'].apply(map_sentiment)
        
        # Split Data (80% Latih, 20% Uji - Standar umum)
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # --- MENU 1: PREDIKSI SENTIMEN ---
    if menu == "Prediksi Sentimen":
        st.header("Uji Coba Model")
        st.caption("Masukkan ulasan tentang fitur COD Tokopedia untuk diprediksi sentimennya.")

        left, right = st.columns([1.2, 0.8], gap="large")
        with left:
            user_input = st.text_area(
                "Teks Ulasan",
                height=170,
                placeholder="Contoh: Fitur COD sangat membantu tapi kurirnya lama...",
            )
            predict_clicked = st.button("Prediksi Sentimen", use_container_width=True)

        with right:
            st.subheader("Hasil")
            if not predict_clicked:
                st.info("Klik **Prediksi Sentimen** untuk melihat hasil.")

        if predict_clicked:
            if user_input and model and vectorizer:
                with st.spinner("Sedang memproses teks (Cleaning, Stemming, dll)..."):
                    # 1. Preprocessing
                    clean_text = preprocess_text(user_input, slang_dict)

                    # 2. Vectorizing
                    tfidf_matrix = vectorizer.transform([clean_text])

                    # 3. Prediction
                    prediction = model.predict(tfidf_matrix)[0]

                # Styling hasil
                if prediction == 2 or str(prediction).lower() in ["positif", "positive"]:
                    label_name = "Positif"
                    emoji = "😄"
                    color = "success"
                elif prediction == 0 or str(prediction).lower() in ["negatif", "negative"]:
                    label_name = "Negatif"
                    emoji = "😡"
                    color = "error"
                else:  # prediction == 1 atau 'Netral'
                    label_name = "Netral"
                    emoji = "😐"
                    color = "warning"

                with right:
                    if color == "success":
                        st.success(f"Sentimen: **{label_name}** {emoji}")
                    elif color == "error":
                        st.error(f"Sentimen: **{label_name}** {emoji}")
                    else:
                        st.warning(f"Sentimen: **{label_name}** {emoji}")

                    with st.expander("Lihat Hasil Preprocessing"):
                        st.write(f"**Original:** {user_input}")
                        st.write(f"**Processed:** {clean_text}")

            elif not user_input:
                st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
            else:
                st.error("Model belum dimuat dengan benar.")

    # --- MENU 2: DATA LATIH ---
    elif menu == "Data Latih":
        st.header("Data Latih (Training Data)")
        
        if not df.empty:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("Jumlah Data Latih", f"{len(train_data)}")
                st.caption("80% dari total data")
            with c2:
                st.metric("Jumlah Total Data", f"{len(df)}")

            st.dataframe(train_data.head(100)) # Menampilkan 100 data pertama agar ringan
            
            # Visualisasi Distribusi
            st.subheader("Distribusi Sentimen Data Latih")
            if 'Label' in train_data.columns:
                st.bar_chart(train_data['Label'].value_counts())
        else:
            st.error("Gagal memuat dataset.")

    # --- MENU 3: DATA UJI ---
    # ... (Bagian atas kode tetap sama) ...

    # --- MENU 3: DATA UJI & EVALUASI ---
    elif menu == "Data Uji":
        st.header("🧪 Evaluasi Model (Data Uji)")
        
        if not df.empty:
            # Tampilkan info dasar
            st.info(f"Jumlah Data Uji: **{len(test_data)}** baris (20% dari total data).")
            
            # Tombol untuk menjalankan Evaluasi (karena agak berat, lebih baik pakai tombol)
            if st.button("Jalankan Evaluasi Akurasi"):
                
                # Progress bar karena preprocessing ratusan data butuh waktu
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. Siapkan list untuk menampung hasil prediksi
                y_true = []
                y_pred_class = []
                
                # 2. Loop untuk Preprocessing & Prediksi satu per satu (agar progress terlihat)
                # Catatan: Ini cara lambat tapi aman. Untuk produksi bisa pakai batch processing.
                total_test = len(test_data)
                correct_count = 0
                
                for i, (index, row) in enumerate(test_data.iterrows()):
                    # Update progress
                    progress = (i + 1) / total_test
                    progress_bar.progress(progress)
                    status_text.text(f"Memproses data ke-{i+1} dari {total_test}...")
                    
                    # Ambil teks & label asli
                    text_raw = row['content']
                    
                    # Mapping Label Asli dari Score (Pastikan logika ini sama dengan notebook!)
                    # Asumsi: 1-2=Negatif, 3=Netral, 4-5=Positif
                    score = row['score']
                    if score <= 2: actual_label = 'Negatif'
                    elif score >= 4: actual_label = 'Positif'
                    else: actual_label = 'Netral'
                    
                    # --- PREDIKSI MODEL ---
                    # Preprocess
                    clean_text = preprocess_text(text_raw, slang_dict)
                    clean_text = re.sub(r'(.)\1{2,}', r'\1', clean_text) # Fix Tom
                    
                    # Vectorize
                    vector = vectorizer.transform([clean_text])
                    
                    # Predict
                    pred_code = model.predict(vector)[0]
                    
                    # Mapping Hasil Prediksi (0,1,2 -> Label)
                    # Sesuai temuan: 0=Negatif, 1=Netral, 2=Positif
                    if pred_code == 2: pred_label = 'Positif'
                    elif pred_code == 0: pred_label = 'Negatif'
                    else: pred_label = 'Netral'
                    
                    # Simpan untuk hitungan
                    y_true.append(actual_label)
                    y_pred_class.append(pred_label)
                    
                    if actual_label == pred_label:
                        correct_count += 1

                progress_bar.empty()
                status_text.empty()
                
                # --- TAMPILKAN HASIL EVALUASI ---
                accuracy = (correct_count / total_test) * 100
                st.success(f"### Akurasi pada Data Uji: {accuracy:.2f}%")
                
                # Hitung Confusion Matrix Manual Sederhana
                st.subheader("Rincian Detail (Confusion Matrix)")
                
                # Buat DataFrame komparasi
                df_result = pd.DataFrame({
                    'Ulasan Asli': test_data['content'].values,
                    'Label Sebenarnya': y_true,
                    'Prediksi Model': y_pred_class
                })
                
                # Filter berdasarkan Label
                labels = ['Positif', 'Negatif', 'Netral']
                
                # Tampilkan Metrics per Kelas
                cols = st.columns(3)
                
                for idx, label in enumerate(labels):
                    with cols[idx]:
                        # Berapa total data asli yg labelnya X
                        total_real = df_result[df_result['Label Sebenarnya'] == label].shape[0]
                        
                        # Berapa yang TEPAT diprediksi X
                        correct = df_result[
                            (df_result['Label Sebenarnya'] == label) & 
                            (df_result['Prediksi Model'] == label)
                        ].shape[0]
                        
                        # Berapa yang SALAH (Meleset)
                        missed = total_real - correct
                        
                        st.markdown(f"#### Kelas {label}")
                        st.write(f"Total Data: **{total_real}**")
                        st.write(f"✅ Benar: **{correct}**")
                        st.write(f"❌ Salah: **{missed}**")
                        
                        # Jika ada yang salah, salah nebak jadi apa?
                        if missed > 0:
                            wrong_guesses = df_result[
                                (df_result['Label Sebenarnya'] == label) & 
                                (df_result['Prediksi Model'] != label)
                            ]['Prediksi Model'].value_counts()
                            st.write("Salah diprediksi jadi:")
                            st.caption(wrong_guesses.to_dict())

                # Tampilkan Tabel Data agar user bisa cek mana yang salah
                st.write("---")
                st.subheader("Tabel Perbandingan Asli vs Prediksi")
                
                # Highlight yang salah agar mudah dilihat
                def highlight_error(row):
                    color = '#ffcccc' if row['Label Sebenarnya'] != row['Prediksi Model'] else '#ccffcc'
                    return [f'background-color: {color}' for _ in row]

                st.dataframe(df_result.style.apply(highlight_error, axis=1))

            else:
                st.write("Klik tombol di atas untuk memulai evaluasi (Mungkin butuh waktu 10-20 detik untuk preprocessing).")
                st.dataframe(test_data.head(10))

        else:
            st.error("Gagal memuat dataset.")

if __name__ == "__main__":
    main()
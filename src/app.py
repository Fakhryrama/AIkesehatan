import streamlit as st
import google.generativeai as genai
from datetime import datetime
from rag import RAGEngine
import os

# =========================================================
# KONFIGURASI & SETUP
# =========================================================
st.set_page_config(page_title="Konsultasi Gejala (Agentic)", page_icon="🩺", layout="centered")

# API KEY SETUP
# Pastikan API Key aman. Di production gunakan st.secrets
api_key = "AIzaSyDx0Qw_jQTAUpAlWrpj9yiuGQC30lErkr0" 
genai.configure(api_key=api_key)

# =========================================================
# 1. INISIALISASI TOOLS (FUNGSI PYTHON)
# =========================================================

# Kita gunakan @st.cache_resource agar RAG Engine hanya dimuat sekali
@st.cache_resource
def get_rag_engine():
    # Mendapatkan lokasi absolut dari file app.py saat ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Mencari folder 'data' yang berada satu level di atas folder 'src'
    # Struktur: AIkesehatan/src/app.py -> naik ke AIkesehatan -> masuk ke data/
    index_path = os.path.join(current_dir, "..", "data", "faiss.index")
    meta_path = os.path.join(current_dir, "..", "data", "meta.pkl")

    # Debugging: Cek path (opsional, akan muncul di terminal)
    print(f"[INFO] Loading RAG index from: {index_path}")
    
    # Cek apakah file benar-benar ada
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"File index tidak ditemukan di: {index_path}. Pastikan struktur folder benar.")

    return RAGEngine(
        index_path=index_path,
        meta_path=meta_path,
        use_local=True,
        min_score=0.40
    )
# Wrapper function untuk RAG agar bisa dipanggil oleh Agent
def cari_info_medis(keluhan: str):
    """
    Gunakan alat ini untuk mencari informasi penyakit yang mirip dengan keluhan pengguna
    dari database medis (knowledge base).
    Args:
        keluhan: Deskripsi gejala atau keluhan pasien.
    Returns:
        String berisi daftar kemungkinan penyakit dan deskripsinya.
    """
    engine = get_rag_engine()
    results = engine.retrieve(keluhan, top_k=3)
    
    if not results:
        return "Tidak ditemukan kecocokan penyakit yang spesifik di database."
    
    output = []
    for r in results:
        info = f"- Penyakit: {r['Disease']}\n  Gejala: {r['Symptoms']}\n  Deskripsi: {r['Description'][:200]}..."
        output.append(info)
    
    return "\n\n".join(output)

# Wrapper function untuk Severity check
def cek_tingkat_keparahan(teks_gejala: str):
    """
    Menganalisis tingkat keparahan dan tanda bahaya (red flags) dari gejala.
    Gunakan alat ini untuk menentukan urgensi medis.
    """
    t = teks_gejala.lower()
    red = []

    if "sesak" in t or "sulit bernapas" in t: red.append("Sesak napas")
    if "nyeri dada" in t: red.append("Nyeri dada")
    if "pingsan" in t or "tidak sadar" in t: red.append("Penurunan kesadaran")
    if "kejang" in t: red.append("Kejang")
    if "demam" in t and ("39" in t or "40" in t or "tinggi" in t): red.append("Demam tinggi")
    if "lemah sebelah" in t or "bicara pelo" in t: red.append("Gejala stroke mendadak")

    if red:
        level = "BERAT"
    elif any(k in t for k in ["nyeri", "sakit", "mual", "pusing", "demam"]):
        level = "SEDANG"
    else:
        level = "RINGAN"

    return {"level": level, "tanda_bahaya": red}

# Wrapper function untuk Klasifikasi Sistem Tubuh
def klasifikasi_sistem_tubuh(teks_gejala: str):
    """
    Mengklasifikasikan gejala ke area sistem tubuh (Pernapasan, Pencernaan, Saraf, dll).
    """
    t = teks_gejala.lower()
    systems = []
    
    if any(w in t for w in ["batuk", "pilek", "flu", "sesak", "napas", "tenggorokan"]): systems.append("Pernapasan")
    if any(w in t for w in ["mual", "muntah", "diare", "perut", "maag", "kembung"]): systems.append("Pencernaan")
    if any(w in t for w in ["pusing", "sakit kepala", "migren", "kejang", "kesemutan"]): systems.append("Saraf / Kepala")
    if any(w in t for w in ["bak", "kencing", "anyang", "urin"]): systems.append("Kemih")
    if any(w in t for w in ["ruam", "gatal", "bintik", "kulit"]): systems.append("Kulit")

    if not systems:
        return "Tidak terklasifikasi spesifik."
    return ", ".join(systems)

# Daftar tools yang akan diberikan ke Agent
my_tools = [cari_info_medis, cek_tingkat_keparahan, klasifikasi_sistem_tubuh]

# =========================================================
# 2. DEFINISI AGENT (SYSTEM INSTRUCTION)
# =========================================================
system_instruction = """
Anda adalah 'MediSense AI', asisten kesehatan virtual yang cerdas, ramah, dan berhati-hati.

TUGAS ANDA:
1. Menerima keluhan dari pengguna.
2. SECARA OTOMATIS menggunakan tools yang tersedia (`cari_info_medis`, `cek_tingkat_keparahan`, `klasifikasi_sistem_tubuh`) jika pengguna menyampaikan gejala penyakit.
3. Jika pengguna hanya menyapa (contoh: "Halo", "Selamat pagi"), balas dengan ramah tanpa menggunakan tools medis.

ATURAN PENTING DALAM MENJAWAB KELUHAN MEDIS:
Setelah mendapatkan informasi dari tools, susun jawabanmu dengan struktur berikut (WAJIB):

1. **Interpretasi Awal**: Ringkas keluhan user dan sebutkan sistem tubuh yang mungkin terdampak (hasil tool klasifikasi).
2. **Analisis Keparahan**: Sebutkan Level (Ringan/Sedang/Berat) dan Warning jika ada tanda bahaya (hasil tool cek_tingkat_keparahan).
3. **Kemungkinan Penyebab**: Jelaskan 2-3 kemungkinan kondisi berdasarkan hasil `cari_info_medis`. Ingatkan ini HANYA kemungkinan.
4. **Saran Perawatan di Rumah**: Berikan tips praktis yang aman.
5. **Kapan ke Dokter**: Jelaskan tanda-tanda kapan harus segera berobat.

DISCLAIMER:
Selalu akhiri dengan: "Informasi ini hanya untuk edukasi awal dan bukan pengganti diagnosis dokter profesional."
Jangan pernah mendiagnosis pasti (misal: "Anda terkena X"), gunakan kata "kemungkinan" atau "suspek".
"""

# =========================================================
# 3. STREAMLIT UI
# =========================================================
# CSS Styles
st.markdown("""
<style>
body { background-color: #f3fbf7; }
.stChatMessage { border-radius: 10px; }
.stChatMessage[data-testid="stChatMessageUser"] { background-color: #E8F5E9; border: 1px solid #2E7D32; }
.stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #FFFFFF; border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🩺 MediSense Agent")
    st.info("Agen ini dapat menggunakan tools (RAG, Severity Checker) secara otomatis saat dibutuhkan.")
    if st.button("Hapus Riwayat Chat"):
        st.session_state.chat_session = None
        st.session_state.messages = []
        st.rerun()

# Inisialisasi Chat Session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state or st.session_state.chat_session is None:
    # Inisialisasi Model dengan Tools
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=my_tools,
        system_instruction=system_instruction
    )
    # Aktifkan automatic function calling agar library yang mengurus eksekusi tools
    st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)

# =========================================================
# 4. CHAT INTERFACE
# =========================================================
st.title("💬 Konsultasi Gejala dengan AI Agent")
st.caption("Ceritakan keluhan Anda, AI akan menganalisisnya secara komprehensif.")

# Tampilkan Riwayat Chat di UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input User
if prompt := st.chat_input("Contoh: Saya sakit kepala belakang berdenyut sejak kemarin..."):
    # 1. Tampilkan pesan user
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Proses di Agent
    with st.chat_message("assistant"):
        with st.spinner("Sedang menganalisis (mencari data medis & cek keparahan)..."):
            try:
                # Kirim pesan ke Agent (Gemini menangani pemanggilan tool secara internal)
                response = st.session_state.chat_session.send_message(prompt)
                
                # Tampilkan jawaban
                st.markdown(response.text)
                
                # Simpan ke riwayat
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

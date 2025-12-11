# src/app.py

import streamlit as st
import google.generativeai as genai
from datetime import datetime
from rag import RAGEngine


# API KEY GEMINI
genai.configure(api_key="AIzaSyBKA7UHgKU_qeGCZ_Zt-50siPcdXEuXlC4")  

def call_gemini(prompt: str) -> str:
    """Pemanggilan model Gemini yang sederhana dan aman."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text or ""
    except Exception as e:
        return f"Saat ini ringkasan otomatis tidak tersedia.\n\nError: {e}"


# =========================================================
# RULE-BASED SEVERITY (TOOL 2)
# =========================================================
def evaluate_severity(text: str):
    t = text.lower()
    red = []

    if "sesak" in t or "sulit bernapas" in t:
        red.append("Sesak napas")
    if "nyeri dada" in t:
        red.append("Nyeri dada")
    if "pingsan" in t or "tidak sadar" in t:
        red.append("Penurunan kesadaran")
    if "kejang" in t:
        red.append("Kejang")
    if "demam" in t and ("39" in t or "40" in t or "tinggi" in t):
        red.append("Demam tinggi")
    if "lemah sebelah" in t or "bicara pelo" in t:
        red.append("Gejala stroke mendadak")

    if red:
        level = "berat"
    elif any(k in t for k in ["nyeri", "sakit", "mual", "pusing", "demam"]):
        level = "sedang"
    else:
        level = "ringan"

    return {"level": level, "red": red}


# =========================================================
# TOOL 3: KLASIFIKASI SISTEM TUBUH
# =========================================================
def classify_body_system(text: str):
    """
    Mengklasifikasikan gejala ke dalam sistem tubuh:
    - Pernapasan
    - Pencernaan
    - Saraf / Kepala
    - Kemih
    - Kulit
    """
    t = text.lower()
    systems = []

    # Pernapasan
    if any(w in t for w in ["batuk", "pilek", "flu", "sesak", "napas", "nafas", "tenggorokan", "dahak"]):
        systems.append("Pernapasan")

    # Pencernaan
    if any(w in t for w in ["mual", "muntah", "diare", "perut", "ulu hati", "kembung", "asam lambung", "maag"]):
        systems.append("Pencernaan")

    # Saraf / Kepala
    if any(w in t for w in ["pusing", "sakit kepala", "kepala berat", "migren", "migraine", "kejang", "baal", "kesemutan"]):
        systems.append("Saraf / Kepala")

    # Kemih
    if any(w in t for w in ["bak", "kencing", "anyang", "anyang-anyangan", "air kecil", "urine", "urin", "pipis"]):
        systems.append("Kemih / Saluran Kemih")

    # Kulit
    if any(w in t for w in ["ruam", "bintik", "gatal", "kemerahan", "benjolan kulit", "bercak kulit", "lentingan"]):
        systems.append("Kulit")

    return systems


# =========================================================
# STREAMLIT UI – THEME WARNA MEDIS
# =========================================================
st.set_page_config(page_title="Konsultasi Gejala", page_icon="🩺", layout="centered")

st.markdown(
    """
<style>
body { background-color: #f3fbf7; }
.main { padding-top: 10px; }

.card {
    padding: 16px;
    background: #E8F5E9;
    border-left: 6px solid #2E7D32;
    border-radius: 12px;
    margin-bottom: 12px;
}

.section-title {
    color: #1B5E20;
    font-size: 22px;
    font-weight: 700;
}

.sub-title {
    color: #4E6C50;
    font-size: 14px;
    margin-bottom: 10px;
}

.stButton > button {
    background-color: #2E7D32;
    color: white;
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    border: none;
}
.stButton > button:hover {
    background-color: #1B5E20;
}

[data-testid="stSidebar"] {
    background-color: #E0F2F1;
}
</style>
""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state["history"] = []


# =========================================================
# SIDEBAR SIMPLE
# =========================================================
with st.sidebar:
    st.markdown("### 🩺 MediSense")
    st.markdown(
        "Tuliskan keluhanmu, dan aplikasi akan membantu memberikan gambaran awal "
        "serta saran langkah berikutnya."
    )

    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("#### Riwayat Singkat")
        for h in st.session_state["history"][:5]:
            st.markdown(f"- {h['waktu']} · _{h['gejala'][:30]}…_")


# =========================================================
# TAB UTAMA
# =========================================================
tab1, tab2 = st.tabs(["💬 Konsultasi Gejala", "ℹ️ Informasi"])

# ========================= TAB 1 =========================
with tab1:
    st.markdown("<div class='section-title'>Konsultasi Gejala</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Ceritakan keluhan: lokasi, durasi, tingkat nyeri, dan gejala lain.</div>",
        unsafe_allow_html=True,
    )

    user_input = st.text_area(
        "Tuliskan keluhanmu:",
        placeholder="Contoh: sakit kepala belakang 2 hari, nyeri sedang, makin berat saat stres, kadang mual.",
        height=140,
    )

    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown("### 💡 Tips")
        st.markdown(
            """
- Lokasi keluhan  
- Sudah berapa lama  
- Tingkat keparahan
- Gejala tambahan  
        """
        )

    run = st.button("🔍 Analisis Gejala", use_container_width=True)

    if run:
        with col_left:
            if not user_input.strip():
                st.warning("Tolong isi keluhan terlebih dahulu.")
            else:
                with st.spinner("Sedang menganalisis..."):

                    # Tool 1 — RAG
                    engine = RAGEngine(use_local=True, min_score=0.40)
                    results = engine.retrieve(user_input, top_k=3)

                    st.markdown("### 📌 Kemungkinan Penyakit")

                    if not results:
                        st.info(
                            "Belum ada kecocokan kuat dari data.\n"
                            "Lengkapi lokasi, durasi, tingkat nyeri, dan gejala tambahan."
                        )
                    else:
                        for r in results:
                            st.markdown(
                                f"""
<div class="card">
  <h4 style="margin:0; color:#1B5E20;">{r['Disease']}</h4>
  <p><b>Skor:</b> {r['score']:.3f}</p>
  <p><b>Gejala umum:</b> {r['Symptoms']}</p>
  <p><b>Ringkasan:</b> {r['Description'][:240]}...</p>
</div>
""",
                                unsafe_allow_html=True,
                            )

                    # Tool 2 — Severity
                    sev = evaluate_severity(user_input)
                    st.markdown("### ⚠️ Tingkat Keparahan")

                    if sev["level"] == "berat":
                        st.error("Tingkat keparahan: **BERAT**. Segera periksa ke fasilitas kesehatan.")
                    elif sev["level"] == "sedang":
                        st.warning("Tingkat keparahan: **SEDANG**. Pantau perkembangan gejala.")
                    else:
                        st.success("Tingkat keparahan: **RINGAN**.")

                    if sev["red"]:
                        st.markdown("**Tanda Bahaya:**")
                        for rf in sev["red"]:
                            st.markdown(f"- {rf}")
                    else:
                        st.markdown("_Tidak ditemukan tanda bahaya dari teks._")

                    # Tool 3 — Klasifikasi Sistem Tubuh
                    systems = classify_body_system(user_input)
                    st.markdown("### 🧭 Perkiraan Sistem Tubuh yang Terkait")
                    if systems:
                        st.markdown(
                            "Gejala yang kamu sampaikan kemungkinan berkaitan dengan sistem berikut:"
                        )
                        for s in systems:
                            st.markdown(f"- **{s}**")
                    else:
                        st.markdown(
                            "_Belum dapat diidentifikasi jelas dari teks. "
                            "Coba jelaskan lebih rinci lokasi, jenis keluhan, dan gejala lain._"
                        )

                    # Ringkasan AI (Agent menggabungkan Tool1 + Tool2 + Tool3)
                    if results:
                        ctx = "\n".join(
                            f"- {r['Disease']}: {r['Description'][:200]}"
                            for r in results
                        )
                    else:
                        ctx = "Belum ada kecocokan penyakit yang kuat dari data."

                    systems_str = ", ".join(systems) if systems else "Belum terklasifikasi jelas dari teks."

                    prompt = f"""
Kamu adalah asisten kesehatan yang ramah dan berhati-hati.
Jawabanmu hanya untuk edukasi, BUKAN diagnosis pasti.

# Ringkasan hasil pencarian penyakit (Tool 1 - RAG):
{ctx}

# Analisis tingkat keparahan (Tool 2 - Severity):
- Level: {sev["level"]}
- Tanda bahaya terdeteksi: {", ".join(sev["red"]) if sev["red"] else "Tidak ada"}

# Klasifikasi sistem tubuh (Tool 3 - Body System Classifier):
- Sistem terkait (perkiraan): {systems_str}

# Keluhan pengguna (apa adanya dari user):
\"\"\"{user_input}\"\"\"

TUGASMU (IKUTI FORMAT DAN URUTAN BERIKUT):

1. Buat bagian berjudul **"Interpretasi Awal Berdasarkan Keluhan"**.
   - Ringkas kembali keluhan pengguna dengan bahasamu sendiri.
   - Jelaskan pola gejala (misalnya: sudah berapa lama, lokasi nyeri, hal yang memicu atau memperberat).
   - Jangan menyebut satu diagnosis pasti di bagian ini.

2. Buat bagian **"Kemungkinan Kondisi yang Perlu Dipertimbangkan"**.
   - Gunakan informasi dari daftar penyakit (Tool 1).
   - Sebutkan 1–3 kondisi/penyakit yang MUNGKIN berhubungan dengan keluhan,
     dengan kalimat seperti: 
     "Salah satu kemungkinan yang dapat dipertimbangkan adalah ...",
     "Kemungkinan lain yang kadang memiliki gejala serupa adalah ...".
   - Jelaskan gejala khas masing-masing secara umum.
   - Tekankan bahwa ini HANYA kemungkinan, bukan kepastian.

3. Buat bagian **"Tindakan Awal yang Bisa Dilakukan di Rumah"**.
   - Berikan 3–5 saran yang aman, sesuai dengan tingkat keparahan ({sev["level"]}).
   - Contoh: istirahat, kompres, minum air, menghindari pemicu, dan sebagainya.
   - Jangan menyarankan penggunaan obat resep secara spesifik.

4. Buat bagian **"Tanda Bahaya yang Harus Diwaspadai"**.
   - Jelaskan tanda-tanda bahaya yang perlu perhatian segera,
     terutama jika ada tanda bahaya dari Tool 2 ({", ".join(sev["red"]) if sev["red"] else "Tidak ada red flag jelas dari teks"}).
   - Jelaskan bahwa jika tanda-tanda ini muncul atau memburuk, pengguna harus segera ke IGD/dokter.

5. Buat bagian **"Informasi yang Perlu Disampaikan ke Dokter"**.
   - Berikan daftar poin yang sebaiknya disiapkan pengguna saat konsultasi, misalnya:
     - durasi gejala,
     - lokasi keluhan,
     - tingkat keparahan (ringan/sedang/berat),
     - gejala tambahan yang dirasakan,
     - riwayat penyakit dan obat yang sedang dikonsumsi,
     - hal-hal yang memperburuk atau meringankan keluhan.

6. Buat bagian **"Catatan Penting"**.
   - Tegaskan bahwa informasi yang kamu berikan hanya gambaran umum,
     tidak bisa menggantikan pemeriksaan langsung oleh tenaga kesehatan.
   - Tekankan bahwa hanya dokter yang dapat menegakkan diagnosis pasti.

Gunakan bahasa Indonesia yang sopan, jelas, dan mudah dipahami.
"""

                    answer = call_gemini(prompt)

                    st.markdown("### 💬 Rangkuman")
                    st.write(answer)

                    # Simpan riwayat
                    st.session_state["history"].insert(
                        0,
                        {
                            "waktu": datetime.now().strftime("%d/%m %H:%M"),
                            "gejala": user_input,
                        },
                    )


# ========================= TAB 2 =========================
with tab2:
    st.markdown("### ℹ️ Informasi")
    st.markdown(
        """
Aplikasi ini membantu memberikan gambaran awal mengenai keluhan yang kamu tuliskan.

Gunakan aplikasi ini untuk:
- pemantauan gejala,
- bahan diskusi saat konsultasi,
- edukasi kesehatan.

**Catatan:** Ini bukan pengganti pemeriksaan tenaga kesehatan.
        """
    )
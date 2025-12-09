# src/app.py

import streamlit as st
import google.generativeai as genai
from datetime import datetime
from rag import RAGEngine

# =========================================================
# API KEY GEMINI
# =========================================================
genai.configure(api_key="AIzaSyBk8cOT29S2JGlpvwh_WbyRWsUUPwz6bW8")


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

                    # Ringkasan AI (Agent menggabungkan Tool1 + Tool2)
                    if results:
                        ctx = "\n".join(
                            f"- {r['Disease']}: {r['Description'][:200]}"
                            for r in results
                        )
                    else:
                        ctx = "Belum ada kecocokan penyakit yang kuat dari data."

                    prompt = f"""
Kamu adalah asisten kesehatan yang ramah dan berhati-hati.
Gunakan informasi berikut untuk memberikan penjelasan edukatif, bukan diagnosis pasti.

# Ringkasan hasil pencarian penyakit (Tool 1):
{ctx}

# Analisis tingkat keparahan (Tool 2):
- Level: {sev["level"]}
- Tanda bahaya terdeteksi: {", ".join(sev["red"]) if sev["red"] else "Tidak ada"}

# Keluhan pengguna:
\"\"\"{user_input}\"\"\"

# TUGASMU:
1. Jelaskan kemungkinan penyebab atau kondisi umum yang sesuai dengan gejala pengguna 
   berdasarkan hasil pencarian penyakit (tanpa menyebut 1 diagnosis pasti).
2. Berikan 3–5 saran awal yang aman dilakukan di rumah.
3. Jelaskan tanda bahaya yang harus membuat pengguna segera ke IGD/dokter.
4. Berikan saran apa yang perlu disampaikan pengguna saat berkonsultasi dengan dokter:
   - durasi gejala,
   - lokasi nyeri,
   - tingkat keparahan,
   - gejala lain yang menyertai,
   - riwayat penyakit bila ada.

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

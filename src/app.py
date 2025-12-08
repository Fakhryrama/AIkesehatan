# src/app.py
import streamlit as st
from rag import RAGEngine
import google.generativeai as genai
genai.configure(api_key="AIzaSyAVw3sSrc16XJ3RBOIT-iLybK-_LSFKKms")
def call_gemini(prompt):
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return response.text


st.set_page_config(page_title="Symptom Chatbot", layout="centered")

engine = RAGEngine(use_local=True)  # use_local=False di production

st.title("Chatbot Pendeteksi Penyakit (Symptom Checker)")

user_input = st.text_area("Jelaskan gejala kamu (bahasa Indonesia):", height=120)

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Cek"):
        if not user_input.strip():
            st.warning("Tolong masukkan gejala.")
        else:
            # 1) retrieve top matches
            results = engine.retrieve(user_input, top_k=3)
            st.subheader("Kemungkinan penyakit (top 3)")
            for r in results:
                st.markdown(f"**{r['Disease']}** (score: {r['score']:.3f})")
                st.markdown(f"- Gejala: {r.get('Symptoms','')}")
                st.markdown(f"- Ringkasan: {r.get('Description','')[:200]}...")

            # 2) buat prompt & panggil Gemini
            context_text = "\n".join([f"{i+1}. {r['Disease']} - {r['Symptoms']} - {r['Description']}" for i,r in enumerate(results)])
            prompt = f"""Kamu adalah asisten kesehatan...
Context: {context_text}
Gejala user: {user_input}
(Tugas: ...)"""
            # panggil Gemini
            answer = call_gemini(prompt)
            st.subheader("Jawaban Asisten")
            st.write(answer)
with col2:
    st.markdown("**Tip**")
    st.markdown("- Jelaskan durasi gejala.")
    st.markdown("- Sebutkan usia & kondisi kronis jika ada.")

results = engine.retrieve(user_input, top_k=3)

context_text = "\n".join([
    f"{i+1}) {r['Disease']} — Symptoms: {r.get('Symptoms','')} \n   Description: {r.get('Description','')} \n   Precautions: {r.get('Precautions','')}"
    for i, r in enumerate(results)
])

prompt = f"""
Kamu adalah asisten kesehatan yang ramah dan hati-hati.
Jangan memberikan diagnosis definitif. Sarankan saran awal & kapan harus periksa ke dokter.

Context (top matches):
{context_text}

Gejala user: {user_input}

Tugasmu:
1) Jelaskan kemungkinan penyakit berdasar context (singkat, 2-3 kalimat)
2) Berikan 3 langkah tindakan awal yang aman
3) Jelaskan tanda bahaya (red flags) yang harus langsung ke UGD
4) Beri saran kata-kata untuk user jika mau konsultasi ke dokter (ceklist gejala, durasi, severity)

Jawab dalam bahasa Indonesia.
"""

answer = call_gemini(prompt)
st.write(answer)


import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from docx import Document
import os
import re
import uuid

import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills_db import skills_db

import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart ATS AI Dashboard",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)
st.title("🤖 Smart ATS AI Dashboard")
st.markdown("AI-powered Resume Ranking + Skill Intelligence System")

nlp = spacy.load("en_core_web_sm")


# -----------------------------
# CSS LOAD
# -----------------------------
def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css("style.css")

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).lower()

# -----------------------------
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    all_skills = []
    for cat in skills_db.values():
        all_skills.extend(cat)

    return list(set([s for s in all_skills if s.lower() in text.lower()]))

# -----------------------------
# ROLE DETECTION
# -----------------------------
def detect_role(skills):
    scores = {}
    for role, role_skills in skills_db.items():
        scores[role] = len(set(skills) & set(role_skills))
    return max(scores, key=scores.get)

# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text(file_path, file_type):
    text = ""

    if file_type == "pdf":
        try:
            with pdfplumber.open(file_path) as pdf:
                for p in pdf.pages:
                    if p.extract_text():
                        text += p.extract_text()
        except:
            pass

        if len(text) < 50:
            try:
                images = convert_from_path(file_path)
                for img in images:
                    text += pytesseract.image_to_string(img)
            except:
                pass

    elif file_type == "docx":
        doc = Document(file_path)
        for p in doc.paragraphs:
            text += p.text + " "

    elif file_type in ["png", "jpg", "jpeg"]:
        text = pytesseract.image_to_string(file_path)

    return clean_text(text)

# -----------------------------
# SCORE ENGINE (FIXED)
# -----------------------------
def calculate_score(resume_text, job_text, skill_w, sem_w):

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_text, job_text])

    semantic = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched = list(set(resume_skills) & set(job_skills))
    missing = list(set(job_skills) - set(resume_skills))

    skill_score = (len(matched) / len(job_skills)) * 100 if job_skills else 0

    final = (skill_score * skill_w) + (semantic * sem_w)

    final = max(0, min(final, 100))

    return int(round(final)), matched, missing, int(round(skill_score)), int(round(semantic))

# -----------------------------
# GAUGE (3D STYLE)
# -----------------------------
def gauge(score, name):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(score),

        number={'suffix': "%", 'font': {'size': 38, 'color': "white"}},

        title={'text': name, 'font': {'color': "#dbeafe"}},

        gauge={
            'axis': {'range': [0, 100], 'visible': False},

            'bar': {'color': "#60a5fa", 'thickness': 0.25},

            'bgcolor': "rgba(255,255,255,0.03)",

            'steps': [
                {'range': [0, 40], 'color': "rgba(239,68,68,0.5)"},
                {'range': [40, 70], 'color': "rgba(245,158,11,0.5)"},
                {'range': [70, 100], 'color': "rgba(34,197,94,0.5)"}
            ],

            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.08)"
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(15,23,42,0.7)",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "white"}
    )

    st.plotly_chart(fig, use_container_width=True)
# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙ ATS Controls")

skill_w = st.sidebar.slider("Skill Importance", 0.0, 1.0, 0.5)
sem_w = st.sidebar.slider("Semantic Importance", 0.0, 1.0, 0.5)

# -----------------------------
# INPUTS
# -----------------------------
job_desc = st.text_area("📌 Job Description", height=200)

files = st.file_uploader(
    "📂 Upload Resumes",
    type=["pdf", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# -----------------------------
# MAIN
# -----------------------------
if st.button("🚀 Run Smart ATS AI"):

    if files and job_desc:

        results = []

        for f in files:

            ext = f.name.split(".")[-1]
            path = f"temp_{uuid.uuid4().hex}.{ext}"

            with open(path, "wb") as file:
                file.write(f.read())

            text = extract_text(path, ext)

            score, matched, missing, skill_s, sem_s = calculate_score(
                text, job_desc, skill_w, sem_w
            )

            role = detect_role(matched)

            results.append({
                "name": f.name,
                "score": score,
                "matched": matched,
                "missing": missing,
                "role": role
            })

            os.remove(path)

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.subheader("🏆 Ranking Dashboard")

        # -----------------------------
        # CARDS UI
        # -----------------------------
        for i, r in enumerate(results):

            st.markdown(f"""
            <div class="card">
                <h3>#{i+1} {r['name']}</h3>
                <h2>{r['score']}%</h2>
                <p><b>Role:</b> {r['role']}</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.success("Matched Skills: " + ", ".join(r["matched"]))

            with col2:
                st.error("Missing Skills: " + ", ".join(r["missing"]))

            gauge(r["score"], r["name"])

    else:
        st.warning("Upload resumes + job description first")

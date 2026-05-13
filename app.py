import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os
import re
import uuid
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills_db import skills_db
# ✅ ADD FUNCTION HERE
def detect_role(job_description):
    ...
st.set_page_config(layout="wide")
def detect_role(job_description):
    jd = job_description.lower()

    if "python" in jd or "django" in jd or "flask" in jd:
        return "Python Developer"
    elif "data" in jd or "machine learning" in jd:
        return "Data Scientist"
    elif "hr" in jd or "recruitment" in jd:
        return "HR"
    elif "it" in jd or "support" in jd or "network" in jd or "system" in jd or "administrator" in jd:
        return "IT"
    else:
        return "General"

# =========================
# CSS LOAD
# =========================
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdf(file_path):
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except:
        text = ""

    if len(text.strip()) == 0:
        images = convert_from_path(file_path)
        for img in images:
            text += pytesseract.image_to_string(img)

    return clean_text(text)


# =========================
# UI
# =========================
st.title("🎫 Smart ATS AI DASHBOARD")
st.markdown(
    "<h3 style='color:#cbd5e1;'>AI-Based Resume Screening, Skill Matching & Candidate Ranking System</h3>",
    unsafe_allow_html=True
)

job_description = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

run = st.button("Run SMART ATS AI")

# =========================
# MAIN LOGIC
# =========================
if run:
    role = detect_role(job_description)
    if not uploaded_files or job_description.strip() == "":
        st.warning("Please upload file and enter job description")

    else:

        st.info("Processing...")

        for uploaded_file in uploaded_files:

            file_path = os.path.join(
                "temp_" + str(uuid.uuid4()) + ".pdf"
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            resume_text = extract_text_from_pdf(file_path)

            resume_lower = resume_text.lower()
            job_desc_lower = job_description.lower()

            # =========================
            # TF-IDF SEMANTIC SCORE
            # =========================
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(
                [resume_text, job_description]
            )

            cosine_score = cosine_similarity(
                tfidf_matrix[0],
                tfidf_matrix[1]
            )[0][0]

            tfidf_score = cosine_score * 100

            # =========================
            # SKILL MATCHING
            # =========================
            required_skills = [
                skill for skill in skills_db
                if skill.lower() in job_desc_lower
            ]

            matched_skills_real = [
                skill for skill in required_skills
                if skill.lower() in resume_lower
            ]

            matched_skills = [
                skill for skill in skills_db
                if skill.lower() in resume_lower
            ]

            missing_skills = [
                skill for skill in skills_db
                if skill.lower() not in resume_lower
            ]

            # =========================
            # SKILL SCORE
            # =========================
            if len(required_skills) > 0:
                skill_score = (
                    len(matched_skills_real) / len(required_skills)
                ) * 100
            else:
                skill_score = 0

            skill_score = min(skill_score, 100)

            # =========================
            # FINAL SCORE
            # =========================
            final_score = (
                0.7 * skill_score
            ) + (
                0.3 * tfidf_score
            )

            final_score = int(round(final_score))

            # =========================
            # GAUGE
            # =========================
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_score,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0, 40], "color": "#ef4444"},
                        {"range": [40, 70], "color": "#f59e0b"},
                        {"range": [70, 100], "color": "#22c55e"}
                    ]
                }
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )

            st.plotly_chart(fig, use_container_width=True)

            # =========================
            # DASHBOARD
            # =========================
            st.subheader("🏆 Ranking Dashboard")

            role = detect_role(job_description)
            st.markdown(
                f"""
                <div class="card">
                    <h2>ATS Score: {final_score}%</h2>
                    <h3>Role: {role}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # =========================
            # SKILLS UI
            # =========================
            st.markdown("## ✅ Matched Skills")

            for skill in matched_skills[:8]:
                st.markdown(f"<div style='color:#22c55e;font-weight:bold'>✔ {skill}</div>", unsafe_allow_html=True)

            st.markdown("## ❌ Missing Skills")

            for skill in missing_skills[:8]:
                st.markdown(f"<div style='color:#ef4444;font-weight:bold'>✘ {skill}</div>", unsafe_allow_html=True)

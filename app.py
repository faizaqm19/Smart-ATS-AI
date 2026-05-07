import streamlit as st
import PyPDF2
import pytesseract
import cv2
import re
import nltk
import numpy as np

from PIL import Image
from pdf2image import convert_from_path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -------------------------------
# SKILL DATABASE
# -------------------------------

SKILLS_DB = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "tensorflow",
    "pandas",
    "numpy",
    "excel",
    "communication",
    "leadership",
    "teamwork",
    "problem solving",
    "docker",
    "aws",
    "html",
    "css",
    "javascript",
    "react"
]

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------

def extract_pdf_text(pdf_file):

    text = ""

    reader = PyPDF2.PdfReader(pdf_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

# -------------------------------
# OCR EXTRACTION
# -------------------------------

def extract_ocr_from_pdf(uploaded_file):

    images = convert_from_path(uploaded_file)

    text = ""

    for img in images:

        img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        extracted = pytesseract.image_to_string(gray)

        text += extracted

    return text

# -------------------------------
# PREPROCESSING
# -------------------------------

stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    tokens = word_tokenize(text)

    filtered = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(filtered)

# -------------------------------
# SKILL EXTRACTION
# -------------------------------

def extract_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS_DB:

        if skill.lower() in text:

            found.append(skill)

    return list(set(found))

# -------------------------------
# ATS SCORE
# -------------------------------

def calculate_ats_score(resume_text, jd_text):

    resume_clean = preprocess_text(resume_text)

    jd_clean = preprocess_text(jd_text)

    tfidf = TfidfVectorizer()

    vectors = tfidf.fit_transform([
        resume_clean,
        jd_clean
    ])

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(similarity * 100, 2)

# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("Smart ATS - AI Resume Analyzer")

uploaded_resume = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if st.button("Analyze Resume"):

    if uploaded_resume is not None:

        # NORMAL PDF EXTRACTION
        resume_text = extract_pdf_text(uploaded_resume)

        # OCR BACKUP
        if len(resume_text.strip()) < 20:

            uploaded_resume.seek(0)

            resume_text = extract_ocr_from_pdf(uploaded_resume)

        # SKILLS
        resume_skills = extract_skills(resume_text)

        jd_skills = extract_skills(job_description)

        # SCORE
        ats_score = calculate_ats_score(
            resume_text,
            job_description
        )

        # MISSING SKILLS
        missing_skills = list(
            set(jd_skills) - set(resume_skills)
        )

        # OUTPUT
        st.subheader("ATS Score")
        st.success(f"{ats_score}%")

        st.subheader("Matched Skills")
        st.write(resume_skills)

        st.subheader("Missing Skills")
        st.write(missing_skills)

        # Suggestions
        st.subheader("Suggestions")

        if len(missing_skills) > 0:

            st.warning(
                "Add these skills: "
                + ", ".join(missing_skills)
            )

        else:

            st.success(
                "Excellent Resume Match!"
            )

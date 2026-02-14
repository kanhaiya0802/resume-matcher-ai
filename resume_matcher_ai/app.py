import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocess import clean_text, remove_stopwords
from utils.skill_extractor import extract_skills
from utils.pdf_reader import extract_text_from_pdf

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

lstm_model = load_model("models/lstm_match_model.h5")
tokenizer = joblib.load("models/tokenizer.pkl")

MAX_LEN = 300

st.set_page_config(page_title="Resume Matcher AI", layout="wide")

st.title("Resume and Job Description Matching System")
st.caption(
    "Upload a resume and provide a job description to evaluate alignment "
    "using rule-based NLP techniques and deep learning."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"])

with col2:
    st.subheader("Job Description")
    job_desc = st.text_area("Job Description", height=250, placeholder="Paste the job description here...")

mode = st.radio(
    "Matching Method",
    [
        "Rule-Based (TF-IDF + Skill Matching)",
        "Deep Learning (LSTM Classifier)"
    ]
)

analyze_btn = st.button("Run Analysis", use_container_width=True)

if analyze_btn:
    if resume_file is None or job_desc.strip() == "":
        st.error("Please upload resume and paste job description.")
        st.stop()

    with st.spinner("Analyzing resume and job description..."):
        resume_text = extract_text_from_pdf(resume_file)

        resume_clean = remove_stopwords(clean_text(resume_text))
        jd_clean = remove_stopwords(clean_text(job_desc))

        lstm_result = None
        if mode == "Deep Learning (LSTM)":
            combined_text = resume_clean + " " + jd_clean
            seq = tokenizer.texts_to_sequences([combined_text])
            padded = pad_sequences(seq, maxlen=MAX_LEN)

            pred = lstm_model.predict(padded, verbose=0)
            pred_class = np.argmax(pred)
            confidence = np.max(pred) * 100

            class_map = {
                0: " Low Match",
                1: " Medium Match",
                2: " High Match"
            }
            lstm_result = class_map[pred_class]


        # Similarity score
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])
        score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

        # Skills
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_desc)

        matched = sorted(set(resume_skills).intersection(set(jd_skills)))
        missing = sorted(set(jd_skills) - set(resume_skills))

        total_jd_skills = len(jd_skills)
        skill_match_percent = (len(matched) / total_jd_skills * 100) if total_jd_skills > 0 else 0
        final_score = 0.7 * skill_match_percent + 0.3 * score

    st.divider()

    if mode == "Deep Learning (LSTM)":
        st.divider()
        st.subheader("LSTM-Based Match Classification")
        st.success(lstm_result)
        st.caption(f"Prediction Confidence: {confidence:.2f}%")

    # Score Indicator
    st.subheader("Overall Match Score")
    st.progress(min(int(final_score), 100))

    if final_score >= 75:
        st.success(f" High Match: **{final_score:.2f}%**")
    elif final_score >= 50:
        st.warning(f" Medium Match: **{final_score:.2f}%**")
    else:
        st.error(f" Low Match: **{final_score:.2f}%**")

    st.divider()

    st.subheader("Skills Match Summary")
    c1, c2, c3 = st.columns(3)

    c1.metric("JD Skills", total_jd_skills)
    c2.metric("Matched", len(matched))
    c3.metric("Missing", len(missing))

    st.info(f"Skill Match Percentage: **{skill_match_percent:.2f}%** (based on skills mentioned in Job Description)")

    st.divider()
    st.subheader("Match Analytics (Charts)")

    chart_col1, chart_col2 = st.columns(2)

    #  Bar chart
    with chart_col1:
        st.caption("Skills Count Comparison")
        fig = plt.figure()
        plt.bar(
            ["Matched", "Missing", "Total JD Skills"],
            [len(matched), len(missing), total_jd_skills]
        )
        st.pyplot(fig)

    #  Pie chart
    with chart_col2:
        st.caption("Matched vs Missing (Skill-based)")
        fig2 = plt.figure()
        values = [len(matched), len(missing)]
        labels = ["Matched", "Missing"]

        # avoid pie chart error when both are 0
        if sum(values) == 0:
            st.info("Not enough skill data for pie chart.")
        else:
            plt.pie(values, labels=labels, autopct="%1.1f%%")
            st.pyplot(fig2)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Matched Skills")
        if matched:
            for skill in matched:
                st.markdown(f"- {skill}")
        else:
            st.info("No skills matched.")

    with colB:
        st.subheader("Missing Skills")
        if missing:
            for skill in missing:
                st.markdown(f"- {skill}")
        else:
            st.success("No missing skills 🎉")

    #  Suggestions
    if missing:
        st.divider()
        st.subheader("Recommendations")
        st.write("Add experience/projects mentioning these missing skills:")
        st.write(", ".join(missing))

    #  Resume Text Preview (optional)
    with st.expander("Preview Extracted Resume Text"):
        st.write(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

    #  Download Report
    st.divider()
    st.subheader("Download Report")

    report = f"""
AI Resume & Job Description Matcher Report
-----------------------------------------

TF-IDF Similarity Score: {score:.2f}%
Skill Match Score: {skill_match_percent:.2f}%
Final Match Score: {final_score:.2f}%

Matched Skills:
{', '.join(matched) if matched else 'None'}

Missing Skills:
{', '.join(missing) if missing else 'None'}

Suggestions:
{'Add projects/experience mentioning missing skills.' if missing else 'No suggestions needed.'}
"""

    st.download_button(
        label="Download Analysis Report (.txt)",
        data=report,
        file_name="resume_jd_report.txt",
        mime="text/plain"
    )
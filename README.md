# Resume and Job Description Matching System

This project is an AI-based system that analyzes the compatibility between a candidate’s resume and a given job description.  
It combines **rule-based NLP techniques** with a **deep learning (LSTM) model** to provide both quantitative scores and qualitative match classification.

The system is designed to simulate how modern ATS (Applicant Tracking Systems) evaluate resumes.

---

## Features

- Resume parsing from PDF files
- Text preprocessing and normalization
- Rule-based matching using:
  - TF-IDF vectorization
  - Cosine similarity
  - Skill overlap analysis
- Deep Learning-based classification using LSTM
- Confidence score for LSTM predictions
- Combined final match score (rule-based weighting)
- Skill match analytics with visualizations
- Downloadable analysis report
- Interactive web interface using Streamlit

---

## Tech Stack

- **Programming Language:** Python 3.11
- **Frontend / UI:** Streamlit
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow (Keras)
- **NLP:** TF-IDF, tokenization, sequence padding
- **Visualization:** Matplotlib
- **Data Handling:** Pandas, NumPy

---

## Project Structure

```bash
resume_matcher_ai/
│
├── app.py # Main Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore # Ignored files for Git
│
├── data/
│ └── resume_jd_dataset.csv # Dataset for LSTM training
│
├── models/
│ ├── lstm_match_model.h5 # Trained LSTM model
│ └── tokenizer.pkl # Saved tokenizer
│
├── notebooks/
│ ├── training.ipynb # Initial experiments
│ └── lstm_training.ipynb # LSTM training notebook
│
├── utils/
│ ├── preprocess.py # Text preprocessing utilities
│ ├── pdf_reader.py # PDF text extraction
│ └── skill_extractor.py # Skill extraction logic
│
└── venv/ # Virtual environment (ignored)

```
---

## How the System Works

### 1. Rule-Based Matching
- Resume and job description texts are cleaned and normalized.
- TF-IDF vectorization is applied.
- Cosine similarity is computed to measure textual similarity.
- Skills are extracted and compared.
- A final rule-based score is calculated using weighted logic.

### 2. Deep Learning (LSTM) Matching
- Resume and job description texts are combined.
- Text is tokenized and padded.
- An LSTM-based neural network predicts:
  - Low Match
  - Medium Match
  - High Match
- The model also outputs a confidence score.

### 3. Final Output
- Rule-based score and skill match percentage are shown.
- LSTM classification and confidence are displayed (if selected).
- Visual analytics and recommendations are generated.

---

## How to Run the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/kanhaiya0802/resume-matcher-ai.git
cd resume-matcher-ai
```
### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the application
```bash
streamlit run app.py
```

## Future Improvements

Increase dataset size for improved LSTM confidence
Add transformer-based models (e.g., BERT)
Support multiple resume formats
Improve skill extraction using NER
Add deployment pipeline using Docker or cloud services



Author

Developed by Kanhaiya Jee
B.Tech Information Technology Student


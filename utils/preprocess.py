import re
import nltk
from nltk.corpus import stopwords

# Download only once; NLTK will skip if already present
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)              # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # remove symbols/numbers
    text = re.sub(r"\s+", " ", text).strip()          # remove extra spaces
    return text

def remove_stopwords(text: str) -> str:
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

import re

# Skills + their possible aliases
SKILL_MAP = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "c": [" c "],  # handled carefully
    "sql": ["sql", "mysql", "postgresql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "fastapi": ["fastapi"],
    "streamlit": ["streamlit"],
    "tensorflow": ["tensorflow", "tf"],
    "keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "nlp": ["nlp", "natural language processing"],
    "deep learning": ["deep learning", "dl"],
    "machine learning": ["machine learning", "ml"],
    "huggingface": ["huggingface", "hugging face"],
    "transformers": ["transformers"],
    "langchain": ["langchain"],
    "faiss": ["faiss"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "git": ["git"],
    "github": ["github"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "nodejs": ["nodejs", "node.js", "node js"],
    "express": ["express", "express.js"],
    "javascript": ["javascript", "js"],
    "html": ["html"],
    "css": ["css"],
}

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

def extract_skills(text: str):
    text = normalize(text)

    found = set()

    for skill, aliases in SKILL_MAP.items():
        for alias in aliases:
            alias = alias.lower().strip()

            # word boundary match (best)
            pattern = r"\b" + re.escape(alias) + r"\b"

            # special case for "c" because \bc\b fails in many places
            if skill == "c":
                if " c " in f" {text} ":
                    found.add("c")
                continue

            if re.search(pattern, text):
                found.add(skill)
                break

    return sorted(found)

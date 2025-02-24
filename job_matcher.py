from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load parsed resumes and job listings
resumes = pd.read_csv("parsed_resumes.csv")
job_listings = pd.read_csv("job_listings.csv")

# Function to extract skills safely
def extract_skills(parsed_info):
    try:
        parsed_dict = ast.literal_eval(parsed_info)  # Convert string to dictionary
        skills = parsed_dict.get("skills", [])  # Extract skills list
        return " ".join(skills) if skills else "NoSkills"  # Convert list to text
    except:
        return "NoSkills"

# ✅ Create 'processed_text' for resumes (skills + resume text)
resumes["processed_text"] = resumes.apply(
    lambda x: f"{extract_skills(x['parsed_info'])} {x['Resume']}", axis=1
)

# ✅ Create 'processed_text' for jobs (title + description)
job_listings["processed_text"] = job_listings.apply(
    lambda x: f"{x['job_title']} {x['job_description']}", axis=1
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
resume_tfidf = vectorizer.fit_transform(resumes["processed_text"])
job_tfidf = vectorizer.transform(job_listings["processed_text"])

# Compute cosine similarity
similarity_matrix = cosine_similarity(resume_tfidf, job_tfidf)

# Find best job match for each resume
resumes["best_match"] = similarity_matrix.argmax(axis=1)
resumes["match_score"] = similarity_matrix.max(axis=1)

# Map matched job titles
resumes["matched_job"] = resumes["best_match"].apply(lambda idx: job_listings.iloc[idx]["job_title"])

# Save matched results
resumes.to_csv("matched_resumes.csv", index=False)
print("✅ Job matching completed! Results saved in matched_resumes.csv.")

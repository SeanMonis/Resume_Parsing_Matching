from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed resumes and job listings from CSV files
resumes = pd.read_csv("parsed_resumes.csv")
job_listings = pd.read_csv("job_listings.csv")

# Function to safely extract skills from parsed resume data
def extract_skills(parsed_info):
    try:
        parsed_dict = ast.literal_eval(parsed_info)  # Convert string representation to dictionary
        skills = parsed_dict.get("skills", [])  # Retrieve skills list if present
        return " ".join(skills) if skills else "NoSkills"  # Convert skills list to a string
    except:
        return "NoSkills"  # Default value if extraction fails

# Combine extracted skills with resume text for better matching
resumes["processed_text"] = resumes.apply(
    lambda x: f"{extract_skills(x['parsed_info'])} {x['Resume']}", axis=1
)

# Combine job title and description to create a comparable text field for jobs
job_listings["processed_text"] = job_listings.apply(
    lambda x: f"{x['job_title']} {x['job_description']}", axis=1
)

# Convert text data into numerical feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
resume_tfidf = vectorizer.fit_transform(resumes["processed_text"])
job_tfidf = vectorizer.transform(job_listings["processed_text"])

# Compute cosine similarity scores between resumes and job descriptions
similarity_matrix = cosine_similarity(resume_tfidf, job_tfidf)

# Identify the best job match for each resume based on highest similarity score
resumes["best_match"] = similarity_matrix.argmax(axis=1)
resumes["match_score"] = similarity_matrix.max(axis=1)

# Map matched job indices to actual job titles
resumes["matched_job"] = resumes["best_match"].apply(lambda idx: job_listings.iloc[idx]["job_title"])

# Save the results to a CSV file
resumes.to_csv("matched_resumes.csv", index=False)
print("✅ Job matching completed! Results saved in matched_resumes.csv.")

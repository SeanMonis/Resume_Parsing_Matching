from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resume text
resume = """
Python Developer with expertise in Machine Learning, Data Science, and SQL.
Worked for 3 years in XYZ Corp. 
Education: Bachelor's in Computer Science.
"""

# Sample job descriptions
job_descriptions = [
    "Looking for a Python Developer skilled in Machine Learning and SQL.",
    "Hiring a Software Engineer with experience in Java and Cloud Computing.",
    "Data Scientist needed with expertise in Python, Data Science, and AI."
]

# Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume] + job_descriptions)

# Compute similarity scores
similarities = cosine_similarity(vectors[0], vectors[1:])

# Rank jobs by similarity score
job_matches = sorted(enumerate(similarities[0]), key=lambda x: x[1], reverse=True)

print("Job Matching Scores:")
for idx, score in job_matches:
    print(f"Job {idx+1}: {job_descriptions[idx]} (Score: {score:.2f})")

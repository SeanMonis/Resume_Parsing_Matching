import pandas as pd
import spacy
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]
    return list(set(skills))

# Sample DataFrame with synthetic resumes
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack'],
    'Skills': [
        'Python Machine Learning NLP', 'Java Spring Boot Microservices', 'React JavaScript CSS',
        'Python Data Analysis SQL', 'JavaScript HTML CSS', 'C++ Embedded Systems IoT',
        'Java Kotlin Android Development', 'Python Django Flask', 'C# Unity Game Development', 'Go Microservices Cloud'
    ],
    'Experience': [3, 5, 2, 4, 3, 6, 5, 4, 3, 7],
    'Education': [
        "Bachelor's in CS", "Master's in CS", "Bachelor's in IT", "Master's in Data Science",
        "Bachelor's in Web Development", "Master's in Embedded Systems", "Bachelor's in Software Engineering",
        "Master's in CS", "Bachelor's in Game Development", "Master's in Cloud Computing"
    ],
    'Job Title': [
        'Data Scientist', 'Backend Developer', 'Frontend Developer', 'Data Analyst', 'Web Developer',
        'Embedded Systems Engineer', 'Mobile Developer', 'Backend Developer', 'Game Developer', 'Cloud Engineer'
    ]
}
df = pd.DataFrame(data)

# Extract skills using NLP
df['Extracted_Skills'] = df['Skills'].apply(extract_skills)

# Convert education to numeric levels
def map_education(edu):
    return 2 if "Master" in edu else 1

df['Education_Level'] = df['Education'].apply(map_education)

# TF-IDF Vectorization of Skills
tfidf = TfidfVectorizer()
skill_matrix = tfidf.fit_transform(df['Skills'])

# Convert to dense format for modeling
X = np.hstack((df[['Experience', 'Education_Level']].values, skill_matrix.toarray()))

y = df['Job Title']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train Random Forest Model
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
kf = KFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y_encoded, cv=kf, scoring='accuracy')

# Nearest Neighbor Matching for Resume-Job Similarity
nn = NearestNeighbors(n_neighbors=1, metric='cosine')
nn.fit(skill_matrix)
distances, indices = nn.kneighbors(skill_matrix)

# Evaluation
clf.fit(X, y_encoded)
y_pred = clf.predict(X)
precision = precision_score(y_encoded, y_pred, average='macro')
recall = recall_score(y_encoded, y_pred, average='macro')
f1 = f1_score(y_encoded, y_pred, average='macro')

print(f"Cross-Validation Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean()}")
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sentence_transformers import SentenceTransformer

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    return list(set(skills))

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

df['Extracted_Skills'] = df['Skills'].apply(extract_skills)

def map_education(edu):
    return 2 if "Master" in edu else 1

df['Education_Level'] = df['Education'].apply(map_education)

# Sentence Embeddings for Skills
skill_embeddings = np.array([model.encode(text) for text in df['Skills']])
X = np.hstack((df[['Experience', 'Education_Level']].values, skill_embeddings))

y = df['Job Title']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Hyperparameter Tuning
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X, y_encoded)
best_clf = grid_search.best_estimator_

# Nearest Neighbor Matching
nn = NearestNeighbors(n_neighbors=1, metric='cosine')
nn.fit(skill_embeddings)
distances, indices = nn.kneighbors(skill_embeddings)

# Evaluation
y_pred = best_clf.predict(X)
precision = precision_score(y_encoded, y_pred, average='macro')
recall = recall_score(y_encoded, y_pred, average='macro')
f1 = f1_score(y_encoded, y_pred, average='macro')
conf_matrix = confusion_matrix(y_encoded, y_pred)

print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

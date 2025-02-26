import pandas as pd
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the NLP model for processing text
nlp = spacy.load("en_core_web_sm")

# Load a pre-trained model to generate sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract skills from text using NLP
def extract_skills(text):
    doc = nlp(text)
    # Extract named entities related to organizations or products (can be technical skills)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    return list(set(skills))

# Sample dataset with candidate details
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

# Convert dataset to DataFrame
df = pd.DataFrame(data)

# Apply skill extraction to each candidate's skill set
df['Extracted_Skills'] = df['Skills'].apply(extract_skills)

# Function to map education level to numeric values
def map_education(edu):
    return 2 if "Master" in edu else 1  # Master's degree is given a higher value

df['Education_Level'] = df['Education'].apply(map_education)

# Generate sentence embeddings for skill sets
skill_embeddings = np.array([model.encode(text) for text in df['Skills']])

# Combine experience, education level, and skill embeddings for model input
X = np.hstack((df[['Experience', 'Education_Level']].values, skill_embeddings))

# Encode job titles as labels for classification
y = df['Job Title']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Set up a Random Forest classifier with hyperparameter tuning
clf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Define parameters to tune
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}

# Perform cross-validation with Leave-One-Out method
grid_search = GridSearchCV(clf, param_grid, cv=LeaveOneOut(), scoring='accuracy')
grid_search.fit(X, y_encoded)

# Select the best classifier
best_clf = grid_search.best_estimator_

# Use nearest neighbors to find closest matches between skill embeddings
nn = NearestNeighbors(n_neighbors=1, metric='cosine')
nn.fit(skill_embeddings)
distances, indices = nn.kneighbors(skill_embeddings)

# Make predictions using the trained model
y_pred = best_clf.predict(X)

# Evaluate model performance
precision = precision_score(y_encoded, y_pred, average='macro')
recall = recall_score(y_encoded, y_pred, average='macro')
f1 = f1_score(y_encoded, y_pred, average='macro')
conf_matrix = confusion_matrix(y_encoded, y_pred)

# Display results
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

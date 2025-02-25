import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load resumes dataset
try:
    resumes = pd.read_csv("parsed_resumes.csv")
except FileNotFoundError:
    print("Error: File 'parsed_resumes.csv' not found.")
    exit()

# Ensure required columns exist
required_columns = {"Resume", "Category"}
if not required_columns.issubset(resumes.columns):
    print("Error: Missing required columns in dataset.")
    exit()

# Convert text data into numerical representation using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes["Resume"])
y = resumes["Category"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save trained model and vectorizer
joblib.dump(classifier, "job_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Model evaluation
y_pred = classifier.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

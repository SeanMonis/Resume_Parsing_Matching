import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load resumes
resumes = pd.read_csv("parsed_resumes.csv")

# Ensure required columns exist
if "Resume" not in resumes.columns or "Category" not in resumes.columns:
    print("Error: Missing required columns in dataset.")
    exit()

# Convert text into numerical representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes["Resume"])
y = resumes["Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(classifier, "job_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Evaluate model
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

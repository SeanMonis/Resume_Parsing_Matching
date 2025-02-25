import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load resumes dataset with error handling
try:
    resumes = pd.read_csv("resumes_new.csv", usecols=["Category", "Resume"])
    resumes.rename(columns={"Category": "category", "Resume": "resume_text"}, inplace=True)
    print("✅ Resumes loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'resumes_new.csv' not found.")
    exit()
except pd.errors.EmptyDataError:
    print("❌ Error: 'resumes_new.csv' is empty.")
    exit()
except pd.errors.ParserError:
    print("❌ Error: CSV parsing issue. Check file format.")
    exit()

# Handle missing values
if resumes.isnull().sum().any():
    print("⚠️ Warning: Missing values detected. Filling with empty strings.")
    resumes.fillna("", inplace=True)

# Debugging: Check unique categories
print("\n🔍 Unique job categories:", resumes["category"].nunique())

# Convert text into numerical representation using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes["resume_text"])
y = resumes["category"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
classifier.fit(X_train, y_train)

# Save trained model and vectorizer
joblib.dump(classifier, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved successfully.")

# Model evaluation
y_pred = classifier.predict(X_test)
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📑 Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Debugging Information
print("\n📝 Sample Resumes:\n", resumes.head())
print("\n🔢 Feature Matrix Shape:", X.shape)
print("\n📊 Labels Distribution:\n", y.value_counts())

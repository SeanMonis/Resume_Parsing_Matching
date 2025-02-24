import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load resumes
try:
    resumes = pd.read_csv("resumes_new.csv", usecols=["Category", "Resume"])
    resumes.rename(columns={"Category": "category", "Resume": "resume_text"}, inplace=True)
    print("Resumes loaded successfully.")
except FileNotFoundError as e:
    print("Error: resumes_new.csv not found.", e)
    exit()
except pd.errors.EmptyDataError:
    print("Error: resumes_new.csv is empty.")
    exit()
except pd.errors.ParserError:
    print("Error: CSV parsing issue. Check file format.")
    exit()

# Debugging: Check for missing values and unique categories
print("Missing values:\n", resumes.isnull().sum())
print("Unique categories:\n", resumes["category"].unique())

# Convert text into numerical representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes["resume_text"])
y = resumes["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(classifier, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Evaluate model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Debugging prints
print("Sample Resumes:\n", resumes.head())
print("Feature Matrix Shape:", X.shape)
print("Labels Distribution:\n", y.value_counts())

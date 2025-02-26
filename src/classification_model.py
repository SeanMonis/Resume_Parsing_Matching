import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("matched_resumes.csv")

# Drop rows with missing values in essential columns
df = df.dropna(subset=["Resume", "matched_job"])

# Encode job labels as numerical values
df["matched_job"] = df["matched_job"].astype("category").cat.codes  

# Convert resume text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Resume"])  
y = df["matched_job"]

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model and vectorizer for future predictions
joblib.dump(clf, "job_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate model performance
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Define job category mapping (update if needed)
job_mapping = {0: "AI Engineer", 1: "Data Scientist", 2: "Database Administrator", 3: "Software Engineer"}

# Generate and visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=job_mapping.values(), yticklabels=job_mapping.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

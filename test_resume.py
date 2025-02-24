import joblib

# Load the trained model and vectorizer
classifier = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Example new resume
new_resume = ["Experienced in Python, data science, and machine learning."]
new_resume_vectorized = vectorizer.transform(new_resume)

# Predict category
predicted_category = classifier.predict(new_resume_vectorized)
print("Predicted Category:", predicted_category[0])

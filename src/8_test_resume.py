import joblib

# Function to load the trained model and vectorizer safely
def load_model_and_vectorizer(model_path="resume_classifier.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    try:
        classifier = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✅ Model and vectorizer loaded successfully.")
        return classifier, vectorizer
    except FileNotFoundError:
        print("❌ Error: Model or vectorizer file not found. Ensure the files exist.")
        exit()
    except Exception as e:
        print(f"❌ Unexpected error while loading model/vectorizer: {e}")
        exit()

# Function to predict the job category
def predict_resume_category(classifier, vectorizer, resume_text):
    if not resume_text or not isinstance(resume_text, str):
        print("❌ Error: Invalid input. Provide a non-empty resume text string.")
        return None

    try:
        resume_vectorized = vectorizer.transform([resume_text])
        predicted_category = classifier.predict(resume_vectorized)
        return predicted_category[0]
    except Exception as e:
        print(f"❌ Error during vectorization or prediction: {e}")
        return None

# Load the trained classifier and vectorizer
classifier, vectorizer = load_model_and_vectorizer()

# Example new resume
new_resume = "Experienced in Python, data science, and machine learning."

# Predict the job category
predicted_category = predict_resume_category(classifier, vectorizer, new_resume)

if predicted_category:
    print("🔍 Predicted Category:", predicted_category)

Resume Parsing and Matching

This project implements a Resume Parsing and Matching system using NLP and Machine Learning. The goal is to extract key details from resumes (skills, experience, education) and match candidates with suitable job roles.

Approach
1. Data Preprocessing
Cleaned and processed job descriptions and resumes.
Extracted relevant features such as skills, qualifications, and experience using NLP techniques.
2. NLP Techniques Used
Named Entity Recognition (NER): Extracted skills and relevant information from resumes.
TF-IDF & Embeddings: Converted text into numerical format for model training.
3. Machine Learning Models
Classification: Used a Random Forest classifier to categorize resumes into job roles.
Matching & Recommendation: Implemented a nearest neighbors approach to find the most relevant jobs for a candidate.
4. Evaluation Metrics
Precision, Recall, and F1-Score were used to measure model performance

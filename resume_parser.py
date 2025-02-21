import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample resume text
resume_text = """
John Doe
Software Engineer
Email: johndoe@email.com
Phone: +1 234 567 8901
Skills: Python, Machine Learning, Data Science, SQL
Experience: 3 years at XYZ Corp.
Education: Bachelor's in Computer Science
"""

# Split resume into lines and assume the first line is the name
lines = resume_text.strip().split("\n")
name = lines[0] if len(lines[0].split()) > 1 else "Not Found"  # Assume the first line is the name

# Process the text with spaCy NLP
doc = nlp(resume_text)

# Extract email using regex
email_match = re.findall(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", resume_text)
email = email_match[0] if email_match else "Not Found"

# Extract phone number using regex
phone_match = re.findall(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}", resume_text)
phone = phone_match[0] if phone_match else "Not Found"

# Extract skills using simple keyword matching
skills_list = ["Python", "Machine Learning", "Data Science", "SQL", "Java", "C++"]
extracted_skills = [skill for skill in skills_list if skill in resume_text]

# Extract experience using regex
experience_match = re.findall(r'(\d+) years?', resume_text)
experience_years = experience_match[0] if experience_match else "Not Found"

# Extract education with major using regex
education_match = re.findall(r"(Bachelor(?:'s)?|Master(?:'s)?|PhD)\s+in\s+([\w\s]+)", resume_text)
education = f"{education_match[0][0]} in {education_match[0][1]}" if education_match else "Not Found"



# Print extracted details
print("Extracted Information:")
print("Name:", name)
print("Email:", email)
print("Phone:", phone)
print("Skills:", extracted_skills)
print("Experience (years):", experience_years)
print("Education Level:", education)

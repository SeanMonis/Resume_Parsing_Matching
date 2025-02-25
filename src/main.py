import pandas as pd  
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("resume_data.csv")  # Update filename if needed

# Check if the dataset is empty or missing critical columns
if df.empty:
    print("Error: Data file is empty or missing required columns.")
    exit()

# Display initial data preview
print("Sample Data:")
print(df.head())

# Convert skills column from a semicolon-separated string to a list
df["Skills"] = df["Skills"].apply(lambda x: [skill.strip().lower() for skill in x.split(";")] if isinstance(x, str) else [])

# Display cleaned skills data
print("\nProcessed Skills Column:")
print(df[["Name", "Skills"]].head())

# Convert experience to numeric values (years of experience)
df["Experience"] = pd.to_numeric(df["Experience"], errors="coerce").fillna(0)

# Standardize education levels
def map_education(edu):
    if isinstance(edu, str):
        edu = edu.lower()
        if "bachelor" in edu:
            return "Bachelor"
        elif "master" in edu:
            return "Master"
        elif "phd" in edu:
            return "PhD"
    return "Other"

df["Education"] = df["Education"].apply(map_education)

# Display education and experience after processing
print("\nEducation & Experience:")
print(df[["Name", "Experience", "Education"]].head())

# Map education levels to numerical values
education_mapping = {"Bachelor": 1, "Master": 2, "PhD": 3, "Other": 0}
df["Education_Level"] = df["Education"].map(education_mapping)

# Store target labels separately
y = df["Job Title"].copy()

# Ensure data is sufficient for training
if len(df) < 2:
    print("Insufficient data samples. Exiting.")
    exit()

# One-hot encode job titles
encoder = OneHotEncoder(sparse_output=False)
job_encoded = encoder.fit_transform(df[["Job Title"]])
job_encoded_df = pd.DataFrame(job_encoded, columns=encoder.get_feature_names_out(["Job Title"]))

# Add encoded job titles to the dataset
df = pd.concat([df, job_encoded_df], axis=1)

# One-hot encode skills
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df["Skills"])
skills_encoded_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

# Merge encoded skills with the dataset
df = pd.concat([df, skills_encoded_df], axis=1)

# Drop unnecessary columns
df.drop(columns=["Skills", "Education", "Job Title", "Name"], inplace=True)

# Display processed DataFrame
print("\nFinal Processed Data:")
print(df.head())

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=None
)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model performance
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, zero_division=1))

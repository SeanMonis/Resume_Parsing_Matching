import pandas as pd  
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Data
df = pd.read_csv("resume_data.csv")  # Adjust the filename

if df.empty:
    print("Error: Data file is empty or missing required columns.")
    exit()

print("Initial Data:")
print(df.head())  # Check if data loaded correctly

# Convert skills column into lists
df["Skills"] = df["Skills"].apply(lambda x: [skill.strip().lower() for skill in x.split(";")] if isinstance(x, str) else [])

# Print cleaned skills
print("\nCleaned Skills Column:")
print(df[["Name", "Skills"]].head())

# Convert experience to numeric (years of experience)
df["Experience"] = pd.to_numeric(df["Experience"], errors="coerce").fillna(0)

# Standardize education levels
def simplify_education(edu):
    if isinstance(edu, str):
        edu = edu.lower()
        if "bachelor" in edu:
            return "Bachelor"
        elif "master" in edu:
            return "Master"
        elif "phd" in edu:
            return "PhD"
    return "Other"

df["Education"] = df["Education"].apply(simplify_education)

# Print cleaned education & experience
print("\nEducation & Experience:")
print(df[["Name", "Experience", "Education"]].head())

# Map education levels to numeric values
education_mapping = {"Bachelor": 1, "Master": 2, "PhD": 3, "Other": 0}
df["Education_Level"] = df["Education"].map(education_mapping)

# Store Job Title column separately
y = df["Job Title"].copy()

# Ensure there are enough samples to proceed
if df.empty:
    print("No sufficient data available for processing. Exiting.")
    exit()

# One-Hot Encode Job Titles
encoder = OneHotEncoder(sparse_output=False)
job_encoded = encoder.fit_transform(df[["Job Title"]])
job_encoded_df = pd.DataFrame(job_encoded, columns=encoder.get_feature_names_out(["Job Title"]))

# Add encoded job titles to the dataset
df = pd.concat([df, job_encoded_df], axis=1)

# One-Hot Encode Skills
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df["Skills"])
skills_encoded_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

# Merge with original DataFrame
df = pd.concat([df, skills_encoded_df], axis=1)

# Drop unnecessary columns
df.drop(columns=["Skills", "Education", "Job Title", "Name"], inplace=True)

# Print the processed DataFrame
print("\nProcessed DataFrame:")
print(df.head())

# Ensure that there are at least two samples for train_test_split
if len(df) < 2:
    print("Not enough data samples for training. Exiting.")
    exit()

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=None
)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate model with zero_division=1 to handle undefined metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

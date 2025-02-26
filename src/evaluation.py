import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("matched_resumes.csv")
print("✅ Columns in CSV:", df.columns.tolist())

# Ensure correct column names
if "actual_job" not in df.columns:
    print("⚠️ 'actual_job' not found. Using 'Category' instead.")
    df.rename(columns={"Category": "actual_job"}, inplace=True)

if "matched_job" not in df.columns:
    raise ValueError("❌ 'matched_job' column is missing. Ensure correct dataset.")

# Remove rows with missing values in actual_job and matched_job
df.dropna(subset=["actual_job", "matched_job"], inplace=True)

# Assign numerical labels
job_mapping = {job: idx for idx, job in enumerate(df["actual_job"].unique())}
df["actual_job_label"] = df["actual_job"].map(job_mapping)
df["matched_job_label"] = df["matched_job"].map(job_mapping)

# Handle NaN values by replacing them with -1 (unknown category)
df["actual_job_label"].fillna(-1, inplace=True)
df["matched_job_label"].fillna(-1, inplace=True)

# Convert labels to integers
df = df.astype({"actual_job_label": "int", "matched_job_label": "int"})

# Classification report
report = classification_report(df["actual_job_label"], df["matched_job_label"], zero_division=0)
print("📊 Classification Report:\n", report)

# Confusion matrix visualization
cm = confusion_matrix(df["actual_job_label"], df["matched_job_label"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=job_mapping.keys(), yticklabels=job_mapping.keys())
plt.xlabel("Predicted Job")
plt.ylabel("Actual Job")
plt.title("Confusion Matrix")
plt.show()

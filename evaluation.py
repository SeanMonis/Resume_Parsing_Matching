import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("matched_resumes.csv")

# Ensure column names exist
if "actual_job" not in df.columns or "matched_job" not in df.columns:
    raise ValueError("Columns 'actual_job' and 'matched_job' are required in matched_resumes.csv")

# Convert job categories to numerical labels
job_mapping = {job: idx for idx, job in enumerate(df["actual_job"].unique())}
df["actual_job_label"] = df["actual_job"].map(job_mapping)
df["matched_job_label"] = df["matched_job"].map(job_mapping)

# Remove NaN values
df = df.dropna(subset=["actual_job_label", "matched_job_label"])

# Compute classification report
report = classification_report(df["actual_job_label"], df["matched_job_label"], zero_division=0)
print("Classification Report:\n", report)

# Generate confusion matrix
cm = confusion_matrix(df["actual_job_label"], df["matched_job_label"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=job_mapping.keys(), yticklabels=job_mapping.keys())
plt.xlabel("Predicted Job")
plt.ylabel("Actual Job")
plt.title("Confusion Matrix")
plt.show()

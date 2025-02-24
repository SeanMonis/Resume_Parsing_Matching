import pandas as pd

# Load matched resumes dataset
df = pd.read_csv("matched_resumes.csv")

# Print unique job titles and their assigned numerical labels
df["matched_job"] = df["matched_job"].astype("category")
job_mapping = dict(enumerate(df["matched_job"].cat.categories))

print("Job Category Mapping:")
for num, job in job_mapping.items():
    print(f"{num}: {job}")

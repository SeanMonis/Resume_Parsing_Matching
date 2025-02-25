import pandas as pd

# Load matched resumes dataset
df = pd.read_csv("matched_resumes.csv")

# Convert job titles to categorical type and create a mapping
df["matched_job"] = df["matched_job"].astype("category")
job_mapping = {idx: job for idx, job in enumerate(df["matched_job"].cat.categories)}

# Print job category mapping
print("\nJob Category Mapping:")
for num, job in job_mapping.items():
    print(f"{num}: {job}")

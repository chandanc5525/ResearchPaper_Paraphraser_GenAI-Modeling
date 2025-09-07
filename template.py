import os

# Project name
project_name = "ResearchPaper-Paraphraser"

# Files to create
files = [
    "app.py",
    "requirements.txt",
    "sample.txt",
    "ResearchPaper-Paraphraser/config.toml"
]

# Create folder
os.makedirs(project_name, exist_ok=True)

# Create Folder and Files:

for file in files:
    file_path = os.path.join(project_name, file)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")  
        print(f"Created: {file_path}")
    else:
        print(f"Already exists: {file_path}")

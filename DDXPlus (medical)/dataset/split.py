import pandas as pd
import json
from pathlib import Path
import os

RANDOM_SEED = 42

# Load all datasets.
# As split in paper is 80/10/10, we combine all and do an 80/20 split.
train_df = pd.read_csv('data/release_train_patients.csv')
test_df = pd.read_csv('data/release_test_patients.csv')
validate_df = pd.read_csv('data/release_validate_patients.csv')

# Combine all datasets
combined_df = pd.concat([train_df, test_df, validate_df], ignore_index=True)

# Target pathologies
target_pathologies = [
    'URTI',
    'Viral pharyngitis',
    'Anemia',
    'HIV (initial infection)',
    'Anaphylaxis',
    'Localized edema',
    'Pulmonary embolism',
    'Influenza',
    'Bronchitis',
    'GERD'
]

# Filter for target pathologies
filtered_df = combined_df[combined_df['PATHOLOGY'].isin(target_pathologies)]

# Load evidence and condition metadata
with open('release_evidences.json', 'r') as f:
    evidences = json.load(f)

with open('release_conditions.json', 'r') as f:
    conditions = json.load(f)

# Helper function to decode an evidence entry
def decode_evidence(code):
    if '_@_' in code:
        base, val = code.split('_@_')
        base = base.strip()
        val = val.strip()
        entry = evidences.get(base)

        if entry:
            question = entry.get('question_en', base)
            is_antecedent = entry.get('is_antecedent', False)

            # Check if value is mapped to a string meaning
            if 'value_meaning' in entry and val in entry['value_meaning']:
                answer = entry['value_meaning'][val]['en']
            else:
                # Assume numeric (e.g. 0–10 scale)
                answer = f"{val}/10"

            return base, question, answer, is_antecedent
        else:
            return base, base, val, False
    else:
        entry = evidences.get(code)
        if entry:
            question = entry.get('question_en', code)
            return code, question, 'Yes', entry.get('is_antecedent', False)
        else:
            return code, code, 'Yes', False

# Function to generate query text for a row
def generate_query(row):
    age = row['AGE']
    sex = row['SEX']
    evids = eval(row['EVIDENCES'])
    initial = row['INITIAL_EVIDENCE']
    pathology = row['PATHOLOGY']

    symptoms_raw = {}
    antecedents_raw = {}

    for e in evids:
        code, question, answer, is_antecedent = decode_evidence(e)

        bucket = antecedents_raw if is_antecedent else symptoms_raw

        if question not in bucket:
            bucket[question] = set()

        bucket[question].add(answer)

    # Format output
    query = f"Sex: {sex}, Age: {age}\n\nSymptoms:\n---------\n"
    if symptoms_raw:
        for q, answers in symptoms_raw.items():
            if len(answers) == 1:
                query += f"- {q} {next(iter(answers))}\n"
            else:
                joined = ", ".join(sorted(answers))
                query += f"- {q} {joined}\n"
    else:
        query += "None\n"

    query += "\nAntecedents:\n------------\n"
    if antecedents_raw:
        for q, answers in antecedents_raw.items():
            if len(answers) == 1:
                query += f"- {q} {next(iter(answers))}\n"
            else:
                joined = ", ".join(sorted(answers))
                query += f"- {q} {joined}\n"
    else:
        query += "None\n"
    
    return query

# Create output directory if it doesn't exist
Path("output").mkdir(exist_ok=True)

# Initialize empty DataFrames for each set
train_data = []
test_data = []
full_data = []

# Process each pathology
for pathology in target_pathologies:
    # Get all cases for this pathology
    pathology_df = filtered_df[filtered_df['PATHOLOGY'] == pathology]
    
    # Sample 1500 cases
    sample_size = min(1500, len(pathology_df))
    sampled_df = pathology_df.sample(n=sample_size, random_state=RANDOM_SEED)
    
    # Generate queries for each case
    for idx, row in sampled_df.iterrows():
        query = generate_query(row)
        full_data.append({'query': query, 'pathology': pathology})
        
        # Split into train (1200) and test (300)
        if len(train_data) < 1200 * (target_pathologies.index(pathology) + 1):
            train_data.append({'query': query, 'pathology': pathology})
        else:
            test_data.append({'query': query, 'pathology': pathology})

# Convert to DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
full_df = pd.DataFrame(full_data)


os.makedirs('split', exist_ok=True)
train_df.to_csv('split/train.csv', index=False)
test_df.to_csv('split/test.csv', index=False)
full_df.to_csv('split/full.csv', index=False)

print("Processing complete. Files have been saved to the 'output' directory.")
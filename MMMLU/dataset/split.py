import pandas as pd
import glob
import os
import random

# --- Configuration ---
DATA_DIR = 'data/'
OUTPUT_DIR = 'split/'
TRAIN_SAMPLES_PER_LANG = 1200
TEST_SAMPLES_PER_LANG = 300
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# --- Helper Function ---
def format_prompt(row):
    """Formats the prompt string from a DataFrame row."""
    question = row['Question']
    option_a = row['A']
    option_b = row['B']
    option_c = row['C']
    option_d = row['D']
    return f"{question}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}"

def extract_language_code(filename):
    """Extracts language code like 'FR-FR' from 'mmlu_FR-FR.csv'."""
    base = os.path.basename(filename)
    # Remove prefix 'mmlu_' and suffix '.csv'
    code = base[len('mmlu_'):-len('.csv')]
    return code

# --- Main Script ---

print("Starting dataset preparation...")

# 1. Find all MMLU language files
csv_files = glob.glob(os.path.join(DATA_DIR, 'mmlu_*.csv'))
if not csv_files:
    raise FileNotFoundError(f"No 'mmlu_*.csv' files found in directory: {DATA_DIR}")

print(f"Found {len(csv_files)} language files:")
for f in csv_files:
    print(f" - {os.path.basename(f)}")

print(f"\nSelecting {TRAIN_SAMPLES_PER_LANG} unique questions for the training set...")

# Use the first file found to determine the training question IDs
reference_file = csv_files[0]
print(f"Using '{os.path.basename(reference_file)}' as reference for question selection.")
try:
    ref_df = pd.read_csv(reference_file)
    # Rename the unnamed first column to 'id'
    ref_df = ref_df.rename(columns={ref_df.columns[0]: 'id'})
except Exception as e:
    print(f"Error reading reference file {reference_file}: {e}")
    exit()

# Create a unique identifier for each question
ref_df['unique_id'] = ref_df['Subject'] + '_' + ref_df['id'].astype(str)

if len(ref_df) < TRAIN_SAMPLES_PER_LANG:
     raise ValueError(f"Reference file '{os.path.basename(reference_file)}' only has {len(ref_df)} questions, "
                      f"but {TRAIN_SAMPLES_PER_LANG} are needed for training.")

# Sample unique IDs for training
train_question_unique_ids = set(ref_df['unique_id'].sample(n=TRAIN_SAMPLES_PER_LANG, random_state=RANDOM_SEED))
print(f"Selected {len(train_question_unique_ids)} unique question IDs for training.")

# 3. Prepare Training Data (Same questions across languages)
all_train_data = []
print("\nProcessing files for TRAINING data...")

for file_path in csv_files:
    lang_code = extract_language_code(file_path)
    print(f" - Processing {lang_code} ({os.path.basename(file_path)})...")

    try:
        df = pd.read_csv(file_path)
        # Rename the unnamed first column to 'id'
        df = df.rename(columns={df.columns[0]: 'id'})
        df['unique_id'] = df['Subject'] + '_' + df['id'].astype(str)

        # Filter for the selected training questions
        train_df_lang = df[df['unique_id'].isin(train_question_unique_ids)].copy() # Use .copy() to avoid SettingWithCopyWarning

        if len(train_df_lang) != TRAIN_SAMPLES_PER_LANG:
             print(f"   WARNING: Found {len(train_df_lang)} matching training questions for {lang_code}, expected {TRAIN_SAMPLES_PER_LANG}. "
                   f"This might indicate missing questions in this file.")
             # Optional: Decide whether to raise an error or continue with fewer samples for this lang if this happens

        # Format the prompt
        train_df_lang['query'] = train_df_lang.apply(format_prompt, axis=1)

        # Add language column
        train_df_lang['language'] = lang_code

        # Select and order final columns
        final_train_cols = train_df_lang[['query', 'Subject', 'language', 'id']]
        all_train_data.append(final_train_cols)

    except Exception as e:
        print(f"   ERROR processing {file_path}: {e}")
        continue # Skip this file if error occurs

# Combine training data from all languages
train_data_final = pd.concat(all_train_data, ignore_index=True)
# Shuffle the combined training data
train_data_final = train_data_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nTotal training samples generated: {len(train_data_final)}")

# 4. Prepare Testing Data (Different questions per language, not used in training)
all_test_data = []
print("\nProcessing files for TESTING data...")

for file_path in csv_files:
    lang_code = extract_language_code(file_path)
    print(f" - Processing {lang_code} ({os.path.basename(file_path)})...")

    try:
        df = pd.read_csv(file_path)
        # Rename the unnamed first column to 'id'
        df = df.rename(columns={df.columns[0]: 'id'})
        df['unique_id'] = df['Subject'] + '_' + df['id'].astype(str)

        # Filter out questions used in the training set
        potential_test_df = df[~df['unique_id'].isin(train_question_unique_ids)].copy() # Use .copy()

        if len(potential_test_df) < TEST_SAMPLES_PER_LANG:
            print(f"   WARNING: Only {len(potential_test_df)} questions available for testing in {lang_code} "
                  f"(after removing training questions), requested {TEST_SAMPLES_PER_LANG}. Using available samples.")
            actual_test_samples = len(potential_test_df)
            if actual_test_samples == 0:
                print(f"   ERROR: No samples left for testing in {lang_code}. Skipping.")
                continue
        else:
            actual_test_samples = TEST_SAMPLES_PER_LANG

        # Sample *different* questions for this language's test set
        test_df_lang = potential_test_df.sample(n=actual_test_samples, random_state=RANDOM_SEED)

        # Format the prompt
        test_df_lang['query'] = test_df_lang.apply(format_prompt, axis=1)

        # Add language column
        test_df_lang['language'] = lang_code

        # Select and order final columns
        final_test_cols = test_df_lang[['query', 'Subject', 'language', 'id']]
        all_test_data.append(final_test_cols)
        print(f"   Selected {len(final_test_cols)} test samples for {lang_code}.")


    except Exception as e:
        print(f"   ERROR processing {file_path}: {e}")
        continue # Skip this file if error occurs

# Combine testing data from all languages
test_data_final = pd.concat(all_test_data, ignore_index=True)
# Shuffle the combined test data
test_data_final = test_data_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nTotal testing samples generated: {len(test_data_final)}")

# 5. Save the datasets
print(f"\nSaving training data to 'train.csv'...")
train_data_final.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)

print(f"Saving testing data to 'test.csv'...")
test_data_final.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

print("\nDataset preparation complete!")
print(f"Training set shape: {train_data_final.shape}")
print(f"Testing set shape: {test_data_final.shape}")

# Display sample rows
print("\nSample Training Data:")
print(train_data_final.head())
print("\nSample Testing Data:")
print(test_data_final.head())
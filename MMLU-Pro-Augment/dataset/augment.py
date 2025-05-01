import pandas as pd
import os
import sys
import numpy as np
import random
import json
import ast  # For safely evaluating the string representation of lists
import time
import logging
import asyncio
from openai import AsyncOpenAI

# --- Configuration ---
MODEL_NAME = "openai/gpt-4o"
API_KEY = "SET_API_KEY"
BASE_URL = "https://openrouter.ai/api/v1"

INPUT_CSV = "data/train-orig.csv"
OUTPUT_CSV = "data/train.csv"
BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 8

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Target Counts per Discipline ---
TARGET_COUNT = 1000
DISCIPLINE_NEEDS = {
    'math': 0, 'physics': 0, 'chemistry': 68, 'law': 99, 'engineering': 231,
    'economics': 356, 'health': 382, 'psychology': 402, 'business': 411, 'biology': 483,
}


async_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Structured Output Schema --- (Keep the same as before)
single_question_schema = {
    "type": "object",
    "properties": {
        "original_question_id": {
            "type": "string",
            "description": "The unique identifier of the original question provided in the input."
        },
        "augmented_question": {
            "type": "string",
            "description": "The rephrased version of the original question, maintaining the core meaning and factual accuracy."
        },
        "augmented_options": {
            "type": "array",
            "description": "The rephrased versions of the multiple-choice options, presented in the *same order* as the input options. The correct answer should remain correct, just potentially rephrased.",
            "items": {"type": "string"}
        }
    },
    "required": ["original_question_id", "augmented_question", "augmented_options"],
    "additionalProperties": False
}

# Schema for the batch response (list of augmented questions)
batch_response_schema = {
    "type": "object",
    "properties": {
        "augmented_questions": {
            "type": "array",
            "description": "A list containing the augmented versions of the questions provided in the batch.",
            "items": single_question_schema
        }
    },
    "required": ["augmented_questions"],
    "additionalProperties": False
}

# --- Helper Functions ---
def parse_options(options_str):
    """Safely parse the string representation of a list."""
    try:
        options = ast.literal_eval(options_str)
        if isinstance(options, list):
            return options
        else:
            logging.warning(f"Parsed options are not a list: {options_str}")
            return None
    except (ValueError, SyntaxError, TypeError) as e:
        logging.error(f"Error parsing options string: {options_str} | Error: {e}")
        return None

async def augment_batch_async(batch_data):
    """
    Sends a batch of questions to the LLM for augmentation asynchronously.

    Args:
        batch_data (list): A list of dictionaries, each containing
                           'question_id', 'question', 'options' (list), 'answer_index'.

    Returns:
        tuple: (list of augmented data or None if error, batch_data for reference)
    """
    input_for_api = []
    for item in batch_data:
        input_for_api.append({
            "question_id": str(item['question_id']), # Ensure string
            "question": item['question'],
            "options": item['options'],
            "correct_option_index": item['answer_index']
        })

    system_prompt = (
        "You are an expert data augmenter specializing in multiple-choice questions. "
        "Your task is to rephrase the provided questions and their corresponding answer choices. "
        "Follow these instructions carefully for each question in the batch:\n"
        "1.  **Rephrase Question:** Slightly modify the wording of the 'question' without changing its core meaning, factual content, or the underlying concept being tested. Do NOT change numbers or specific entities if it alters the problem.\n"
        "2.  **Rephrase Options:** Slightly modify the wording of each 'option' in the 'options' list. Maintain the original meaning and ensure the option corresponding to the 'correct_option_index' remains the correct answer after rephrasing.\n"
        "3.  **Maintain Order:** Return the 'augmented_options' in the *exact same order* as the input 'options'. Do NOT shuffle them.\n"
        "4.  **Format:** Respond strictly according to the provided JSON schema."
    )
    user_prompt = f"Please augment the following batch of questions:\n{json.dumps(input_for_api, indent=2)}"

    try:
        logging.info(f"Sending batch of {len(batch_data)} questions to {MODEL_NAME}...")
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "strict": True,
                    "schema": batch_response_schema,
                }
            },
            temperature=0.5,
        )

        if isinstance(response.choices[0].message.content, str):
             try:
                 output_data = json.loads(response.choices[0].message.content)
             except json.JSONDecodeError as e:
                 logging.error(f"Failed to decode JSON response: {e}")
                 logging.error(f"Raw response content: {response.choices[0].message.content}")
                 return None, batch_data
        elif hasattr(response.choices[0].message, 'content') and isinstance(response.choices[0].message.content, dict):
             output_data = response.choices[0].message.content
        else:
             logging.error(f"Unexpected response format. Content type: {type(response.choices[0].message.content)}")
             logging.error(f"Raw response choice: {response.choices[0]}")
             try:
                 output_data = json.loads(str(response.choices[0].message.content))
             except:
                 return None, batch_data

        if isinstance(output_data, dict) and 'augmented_questions' in output_data and isinstance(output_data['augmented_questions'], list):
            # Case 1: Correct structure received
            list_to_process = output_data['augmented_questions']
            logging.info(f"Received {len(list_to_process)} augmented items from API.")
        elif isinstance(output_data, list):
            # Case 2: List received directly (deviation from schema)
            list_to_process = output_data
            logging.info(f"Received {len(list_to_process)} augmented items from API.")
        else:
            logging.error(f"Invalid or unexpected structure in LLM response: {output_data}")
            return None, batch_data
        
        return list_to_process, batch_data

    except Exception as e:
        logging.error(f"An error occurred during API call: {e}")
        logging.error(response)
        import traceback
        logging.error(traceback.format_exc())
        return None, batch_data

def process_augmented_results(augmented_items, batch_df, augmented_rows, augmented_ids):
    """Process the results from an augmented batch."""
    processed_count = 0
    for augmented_item in augmented_items:
        # Get fields from response
        original_id = augmented_item.get('original_question_id', augmented_item.get('question_id'))
        augmented_q = augmented_item.get('augmented_question', augmented_item.get('question'))
        augmented_opts = augmented_item.get('augmented_options', augmented_item.get('options'))

        # Find original row
        original_row_matches = batch_df[batch_df['question_id'] == str(original_id)]
        if original_row_matches.empty or not augmented_q or not augmented_opts:
            logging.warning(f"Missing data for augmented ID {original_id}. Skipping.")
            continue

        original_row = original_row_matches.iloc[0]
        original_options = parse_options(original_row['options'])
        
        try:
            original_answer_index = int(original_row['answer_index'])
        except (ValueError, TypeError):
            logging.warning(f"Invalid original answer index for {original_id}. Skipping.")
            continue

        # Validate options and answer index
        if (not original_options or len(original_options) != len(augmented_opts) or
            not (0 <= original_answer_index < len(original_options)) or
            not (0 <= original_answer_index < len(augmented_opts))):
            logging.warning(f"Option length mismatch or invalid index for ID {original_id}. Skipping.")
            continue

        # Get correct answer and shuffle options
        correct_answer_text_augmented = augmented_opts[original_answer_index]
        shuffled_options = augmented_opts[:]
        random.shuffle(shuffled_options)

        try:
            new_answer_index = shuffled_options.index(correct_answer_text_augmented)
        except ValueError:
            logging.error(f"Correct answer not found in shuffled options for ID {original_id}. Skipping.")
            continue

        # Create new row
        new_row = {
            'question_id': f"{original_row['question_id']}_2",
            'question': augmented_q,
            'options': json.dumps(shuffled_options),
            'answer': chr(ord('A') + new_answer_index),
            'answer_index': new_answer_index,
            'category': original_row['category'],
            'src': f"{original_row['src']}_augment"
        }
        
        augmented_rows.append(new_row)
        augmented_ids.add(original_row['question_id'])
        processed_count += 1
        
    return processed_count

async def main_async():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}")
        logging.error("Please run 'conv-and-split.py' first or ensure the file exists.")
        sys.exit(1)

    logging.info(f"Loading original training data from {INPUT_CSV}...")
    try:
        df_orig = pd.read_csv(INPUT_CSV)
        logging.info(f"Loaded {len(df_orig)} original training questions.")
    except Exception as e:
        logging.error(f"Failed to load {INPUT_CSV}: {e}")
        sys.exit(1)

    required_cols = ['question_id', 'question', 'options', 'answer', 'answer_index', 'category', 'src']
    if not all(col in df_orig.columns for col in required_cols):
        logging.error(f"Input CSV missing required columns. Found: {df_orig.columns}. Required: {required_cols}")
        sys.exit(1)

    augmented_rows = []
    augmented_ids = set() # Keep track of original IDs that have been augmented
    total_augmented_count = 0
    api_calls_made = 0

    disciplines_to_augment = [d for d, n in DISCIPLINE_NEEDS.items() if n > 0]

    # Ensure 'question_id' is string type for reliable checking in augmented_ids
    df_orig['question_id'] = df_orig['question_id'].astype(str)

    for discipline in disciplines_to_augment:
        needed = DISCIPLINE_NEEDS[discipline]
        if needed <= 0:
            logging.info(f"Skipping {discipline}, no augmentation needed.")
            continue

        logging.info(f"--- Augmenting {discipline} ---")
        logging.info(f"Target: {TARGET_COUNT}, Need to augment: {needed}")

        augmented_count_for_discipline = 0

        # Loop until target for discipline is met or no more options
        while augmented_count_for_discipline < needed:
            # Find available original questions for this discipline
            discipline_df = df_orig[df_orig['category'] == discipline]
            available_to_augment = discipline_df[~discipline_df['question_id'].isin(augmented_ids)]

            if available_to_augment.empty:
                logging.warning(f"No more original questions available to augment for {discipline}. Achieved {augmented_count_for_discipline}/{needed}.")
                break

            # Determine batch size for this iteration
            num_still_needed = needed - augmented_count_for_discipline
            target_batch_count = min(MAX_CONCURRENT_REQUESTS, (num_still_needed + BATCH_SIZE - 1) // BATCH_SIZE)
            sample_size = min(num_still_needed, len(available_to_augment), BATCH_SIZE * target_batch_count)
            
            logging.info(f"Need {num_still_needed} more for {discipline}. Sampling {sample_size} from {len(available_to_augment)} available questions.")
            logging.info(f"Will process in {target_batch_count} concurrent batches")
            
            # Sample questions to augment
            questions_to_augment_df = available_to_augment.sample(n=sample_size, random_state=int(time.time()))
            
            # Prepare all batches for concurrent processing
            all_batches = []
            all_batch_dfs = []
            
            for i in range(0, len(questions_to_augment_df), BATCH_SIZE):
                if augmented_count_for_discipline >= needed:
                    break
                    
                batch_df = questions_to_augment_df.iloc[i:i + BATCH_SIZE]
                batch_input_data = []

                # Prepare batch data
                for _, row in batch_df.iterrows():
                    options_list = parse_options(row['options'])
                    try:
                        answer_idx_int = int(row['answer_index'])
                        if options_list and 0 <= answer_idx_int < len(options_list):
                            batch_input_data.append({
                                'question_id': row['question_id'],
                                'question': row['question'],
                                'options': options_list,
                                'answer_index': answer_idx_int
                            })
                        else:
                            logging.warning(f"Skipping question {row['question_id']}: invalid options or answer_index out of bounds.")
                    except (ValueError, TypeError):
                        logging.warning(f"Skipping question {row['question_id']} due to invalid answer_index: {row['answer_index']}")

                if not batch_input_data:
                    logging.warning("Skipping empty or invalid batch.")
                    continue
                    
                all_batches.append(batch_input_data)
                all_batch_dfs.append(batch_df)
            
            # Process batches concurrently
            if all_batches:
                # Create tasks for each batch
                tasks = [augment_batch_async(batch) for batch in all_batches]
                
                # Execute all tasks concurrently and gather results
                api_calls_made += len(tasks)
                results = await asyncio.gather(*tasks)
                
                # Process all results
                for augmented_batch_results, batch_data in results:
                    if not augmented_batch_results:
                        logging.error(f"API call failed for a batch in {discipline}.")
                        continue
                        
                    # Find the matching batch_df
                    batch_index = all_batches.index(batch_data)
                    batch_df = all_batch_dfs[batch_index]
                    
                    # Process the results
                    processed = process_augmented_results(
                        augmented_batch_results, 
                        batch_df, 
                        augmented_rows, 
                        augmented_ids
                    )
                    
                    augmented_count_for_discipline += processed
                    total_augmented_count += processed
                    
                    if augmented_count_for_discipline >= needed:
                        logging.info(f"Reached target for {discipline}: {augmented_count_for_discipline}/{needed}")
                        break

    # --- Post-Augmentation Summary and Saving ---
    logging.info(f"--- Augmentation Summary ---")
    logging.info(f"Total questions successfully augmented: {total_augmented_count}")
    logging.info(f"Total API calls made: {api_calls_made}")
    logging.info(f"Number of unique original questions augmented: {len(augmented_ids)}")

    df_augmented = pd.DataFrame(augmented_rows)
    df_final = pd.concat([df_orig, df_augmented], ignore_index=True)
    logging.info(f"Combined dataset size: {len(df_final)} rows.")

    final_counts = df_final['category'].value_counts()
    logging.info("Final counts per category:\n" + str(final_counts))
    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

def main():
    random.seed(42)
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
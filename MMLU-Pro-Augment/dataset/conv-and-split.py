import pandas as pd
import os
import sys
import numpy as np

def convert_parquet_to_csv(parquet_path, separator=',', encoding='utf-8'):
    """
    Converts the MMLU Test Parquet (from cais/MMLU) to the format found in MMMLU
    """
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at '{parquet_path}'")
        sys.exit(1)

    try:
        print(f"Reading Parquet file: {parquet_path} ...")
        df = pd.read_parquet(parquet_path)
        print(f"Read {len(df)} rows.")
        
        # Filter to specified categories
        categories = [
            'math', 'health', 'physics', 'business', 'biology', 
            'chemistry', 'economics', 'engineering', 'psychology', 'law'
        ]
        filtered_df = df[df['category'].isin(categories)]
        print(f"Filtered to {len(filtered_df)} rows after category filtering.")
        
        # Remove the 'cot_content' column
        if 'cot_content' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['cot_content'])
            print("Removed 'cot_content' column.")
        
        # Initialize test and train dataframes
        test_df = pd.DataFrame()
        train_df = pd.DataFrame()
        
        # Process each category
        for category in categories:
            category_df = filtered_df[filtered_df['category'] == category]
            print(f"Category '{category}' has {len(category_df)} rows")
            
            # Shuffle the category dataframe
            category_df = category_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Take up to 200 rows for test set
            test_size = min(200, len(category_df))
            category_test = category_df[:test_size]
            category_train = category_df[test_size:]
            
            # Limit train set to 1000 items if more are available
            max_train_size = 1000
            if len(category_train) > max_train_size:
                category_train = category_train[:max_train_size]
                print(f"Limiting '{category}' train set to {max_train_size} items")
            
            # Append to test and train dataframes
            test_df = pd.concat([test_df, category_test])
            train_df = pd.concat([train_df, category_train])
        
        print(f"Split into {len(train_df)} train rows and {len(test_df)} test rows.")
        
        # Write train and test CSV files
        train_csv_path = "data/train-orig.csv"
        test_csv_path = "data/test.csv"
        
        print(f"Writing train CSV file: {train_csv_path} ...")
        train_df.to_csv(
            train_csv_path,
            sep=separator,
            encoding=encoding,
            index=False
        )
        
        print(f"Writing test CSV file: {test_csv_path} ...")
        test_df.to_csv(
            test_csv_path,
            sep=separator,
            encoding=encoding,
            index=False
        )
        
        print("Conversion and splitting successful!")

    except ImportError:
         print("Error: Missing necessary libraries. Please install pandas and pyarrow (or fastparquet):")
         print("pip install pandas pyarrow")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during conversion: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    convert_parquet_to_csv("data/test-00000-of-00001.parquet")
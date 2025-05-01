import pandas as pd
import os
import sys

def convert_parquet_to_csv(parquet_path, csv_path, reference_csv_path, separator=',', encoding='utf-8'):
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

        # --- Data Transformation ---
        # 1. Expand 'choices' into A, B, C, D columns
        try:
            choices_df = pd.DataFrame(
                df['choices'].tolist(),
                index=df.index,
                columns=['A', 'B', 'C', 'D']
            )
            df = pd.concat([df, choices_df], axis=1)
        except Exception as e:
            print(f"\nError processing 'choices' column: {e}")
            sys.exit(1)

        # 2. Map numeric 'answer' (0-3) to letters (A-D)
        try:
            answer_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            df['Answer'] = df['answer'].astype(int).map(answer_mapping)
            if df['Answer'].isnull().any():
                 print("\nWarning: Some 'answer' values could not be mapped to A/B/C/D.")
                 sys.exit(1)
        except KeyError as e:
             print(f"\nError mapping 'answer': Invalid value {e} found in 'answer' column.")
             sys.exit(1)
        except Exception as e:
            print(f"\nError processing 'answer' column: {e}")
            sys.exit(1)

        # 3. Rename 'question' and 'subject' columns
        df.rename(columns={'question': 'Question', 'subject': 'Subject'}, inplace=True)

        # 4. Select and reorder columns
        final_columns = ['Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject']
        missing_cols = [col for col in final_columns if col not in df.columns]
        if missing_cols:
            print(f"\nError: The following expected columns are missing after processing: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)

        df_final = df[final_columns]
        
        # Group by Subject and reset index within each group
        df_final = df_final.groupby('Subject', group_keys=False).apply(lambda x: x.reset_index(drop=True))

        if reference_csv_path:
            print(f"\nComparing generated data with reference CSV: {reference_csv_path}")
            if not os.path.exists(reference_csv_path):
                print(f"Error: Reference CSV file not found at '{reference_csv_path}'")
                sys.exit(1)

            try:
                print("Reading reference CSV...")
                df_ref = pd.read_csv(
                    reference_csv_path,
                    sep=separator,
                    encoding=encoding,
                    index_col=0
                )
                print(f"Read {len(df_ref)} rows from reference CSV.")
                comparison_columns = ['Answer', 'Subject']

                missing_ref_cols = [col for col in comparison_columns if col not in df_ref.columns]
                if missing_ref_cols:
                    print(f"\nError: Reference CSV is missing required columns for comparison: {missing_ref_cols}")
                    print(f"Reference CSV columns: {df_ref.columns.tolist()}")
                    sys.exit(1)

                df_new_subset = df_final[comparison_columns]
                df_ref_subset = df_ref[comparison_columns]

                print("Performing comparison...")
                are_equal = df_new_subset.equals(df_ref_subset)

                if are_equal:
                    print("Comparison successful")
                else:
                    print("\nError: Comparison failed!")
                    print("Columns A, B, C, D, Answer, Subject DO NOT match the reference CSV.")
                    sys.exit(1)

            except Exception as e:
                print(f"\nAn error occurred during comparison: {e}")
                import traceback
                print(traceback.format_exc())
                sys.exit(1)


        print(f"Writing CSV file: {csv_path} ...")
        df_final.to_csv(
            csv_path,
            sep=separator,
            encoding=encoding,
            index=True
        )
        print("Conversion successful!")

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
    convert_parquet_to_csv("test-00000-of-00001.parquet", "mmlu_EN-EN.csv", "mmlu_ID-ID.csv")
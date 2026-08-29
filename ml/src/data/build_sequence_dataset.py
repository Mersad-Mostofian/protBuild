import sys, os
import pandas as pd
from pathlib import Path


def get_csv_files(directory: str) -> list[Path]:
    return [
         file for file in Path(directory).iterdir() 
            if file.is_file() and file.suffix == ".csv"
        ]

def extract_seqs(input_dir, output_file):
    dataframes = []
    for file in get_csv_files(input_dir):
        print(f"Reading: {file}")

        df = pd.read_csv(file, usecols=['sequence'])
        dataframes.append(df)
    if not dataframes:
        print("Error: No CSV files found!")
        sys.exit(1)

    df = pd.concat(dataframes, ignore_index=True)
    df = df.dropna(subset=["sequence"])
    df = df.drop_duplicates(subset=["sequence"])
    df = df.reset_index(drop=True)

    df.to_csv(output_file, index=False)

    print(f"\nTotal unique sequences: {len(df)}")
    print(f"Output: {output_file}")

if __name__ == "__main__":

    if len(sys.argv) <= 1:
            print("\nPlease provide input directory.")
            print('\nUsage:')
            print('build_sequence_dataset.py <input_dir> [output_file]')
            sys.exit(1)
    
    if sys.argv[1] in ('-h', '--help'):
        print('\nUsage:')
        print('build_sequence_dataset.py <input_dir> [output_file]')
        sys.exit(0)

    input_dir = sys.argv[1]

    if not os.path.isdir(input_dir):
        print(f'\nError: Directory {input_dir} does not exist!')
        sys.exit(1)

    output_file = sys.argv[2] if len(sys.argv) > 2 else "protein_sequences.csv"

    extract_seqs(input_dir, output_file)


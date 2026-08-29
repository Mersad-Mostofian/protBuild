import pandas as pd
import sys, os

def dataset_to_text(dataset_file: str, special_token: str) -> str:
    df = pd.read_csv(dataset_file)
    return special_token.join(df['sequence'].tolist())

def write_file(output_file: str, text: str):
      with open(output_file, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
                print("\nPlease provide input file.")
                print('\nUsage:')
                print('build_protein_corpus.py <input_file> [output_file]')
                sys.exit(1)
        
    if sys.argv[1] in ('-h', '--help'):
        print('\nUsage:')
        print('build_protein_corpus.py <input_file> [output_file]')
        sys.exit(0)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f'\nError: File {input_file} does not exist!')
        sys.exit(1)

    output_file = sys.argv[2] if len(sys.argv) > 2 else "protein_sequences.txt"

    write_file(output_file,
                dataset_to_text(input_file, special_token='<|endofprotein|>'))

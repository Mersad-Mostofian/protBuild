import pandas as pd
import sys, os

def dataset_to_corpus(dataset_file: str, output_file: str,special_token: str) -> str:
    for chunk in pd.read_csv(dataset_file, usecols=['sequence'], chunksize=10_000):
          with open(output_file, 'a') as f:
                for seq in chunk['sequence']:
                    f.write(seq)
                    f.write(special_token)

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

    dataset_to_corpus(dataset_file=input_file,
                       output_file=output_file, special_token='<|endofprotein|>')
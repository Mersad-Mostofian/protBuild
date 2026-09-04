import sys, os
from .read_file import _read_chunk_by_chunk
from tqdm import tqdm


def _count_of_proteins(corpus_path, end_token):
    cnt = 0
    for _ in _read_chunk_by_chunk(corpus_path, end_token):
        cnt += 1
    return cnt

def split(corpus_path, end_token, split_rate=0.9):
    number_of_proteins = _count_of_proteins(
        corpus_path,
        end_token
    )

    split_idx = int(number_of_proteins * split_rate)
    train_path = corpus_path + '_train.txt'
    val_path = corpus_path + '_val.txt'

    with open(train_path, 'w') as train_file, \
         open(val_path, 'w') as val_file:

        for i, protein in tqdm(
            enumerate(_read_chunk_by_chunk(corpus_path, end_token)),
            total=number_of_proteins,
            desc="Splitting corpus",
            unit="protein"
        ):

            if i < split_idx:
                train_file.write(protein)
                train_file.write(end_token)
            else:
                val_file.write(protein)
                val_file.write(end_token)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("\nPlease provide corpus file.")
        print("\nUsage:")
        print("split_corpus.py <corpus_path> [split_rate]")
        sys.exit(1)

    if sys.argv[1] in ('-h', '--help'):
        print("\nUsage:")
        print("split_corpus.py <corpus_path> [split_rate]")
        print("\nArguments:")
        print("  corpus_path    Path to protein corpus")
        print("  split_rate     Train split ratio (default: 0.9)")
        sys.exit(0)

    corpus_path = sys.argv[1]

    if not os.path.exists(corpus_path):
        print(f"\nError: File '{corpus_path}' does not exist!")
        sys.exit(1)

    split_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9

    if not 0 < split_rate < 1:
        print("\nError: split_rate must be between 0 and 1.")
        sys.exit(1)

    split(
        corpus_path=corpus_path,
        end_token='<|endofprotein|>',
        split_rate=split_rate
    )

    print("\nCorpus split successfully.")
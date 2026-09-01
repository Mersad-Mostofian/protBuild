import argparse
from pathlib import Path

from .bpe import BPETokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on a protein corpus."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the protein corpus."
    )
    parser.add_argument(
        "output_name",
        type=Path,
        help="Output tokenizer name."
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Final vocabulary size. Default: 5000"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Number of protein occurrences used for BPE training."
            "Default: use the full corpus."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling. Default: 0"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokenizer"),
        help="Directory where vocab and merges are saved."
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        parser.error(
            f"Input file does not exist: {args.input_file}"
        )

    if args.vocab_size <= 0:
        parser.error(
            "vocab-size must be greater than 0."
        )

    if args.sample_size is not None and args.sample_size <= 0:
        parser.error(
            "sample-size must be greater than 0."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    vocab_path = (
        args.output_dir /
        f"{args.output_name}_vocab.json"
    )

    merges_path = (
        args.output_dir /
        f"{args.output_name}_merges.json"
    )

    print("Training BPE tokenizer...")
    print(f"Input: {args.input_file}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Sample size: {args.sample_size}")
    print(f"Seed: {args.seed}")
    print()

    tokenizer = BPETokenizer()

    tokenizer.train(
        file_path=args.input_file,
        vocab_size=args.vocab_size,
        allowed_special={"<|endofprotein|>"},
        sample_size=args.sample_size,
        seed=args.seed
    )

    tokenizer.save_vocab_and_merges(
        vocab_path=vocab_path,
        merges_path=merges_path
    )

    print()
    print("Tokenizer trained successfully.")
    print(f"Vocabulary : {vocab_path}")
    print(f"Merges     : {merges_path}")
    print(f"Tokens     : {len(tokenizer.vocab)}")
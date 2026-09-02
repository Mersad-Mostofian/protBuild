from .read_file import _read_chunk_by_chunk
from torch.utils.data import IterableDataset
import torch

class ProteinDataset(IterableDataset):
    def __init__(self, corpus_path, tokenizer,
                  context_length, end_token="<|endofprotein|>"):
        super().__init__()
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.end_token = end_token

    def __iter__(self):
        for protein in _read_chunk_by_chunk(self.corpus_path, self.end_token):
            tokens = self.tokenizer.encode(protein, allowed_special=self.end_token)

            if len(tokens) < 2:
                continue

            for start in range(0, len(tokens) - 1,
                                self.context_length):
                chunk = tokens[
                    start:start + self.context_length + 1
                ]

                if len(chunk) < 2:
                    continue

                input_ids = chunk[:-1]
                target_ids = chunk[1:]

                yield(
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long)
                )

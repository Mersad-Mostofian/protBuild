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
        buffer = []
        for protein in _read_chunk_by_chunk(self.corpus_path, self.end_token):
            tokens = self.tokenizer.encode(protein, allowed_special={self.end_token})
            buffer.extend(tokens)

            while len(buffer) >= self.context_length + 1:
                chunk = buffer[:self.context_length + 1]
                input_ids = chunk[:-1]
                target_ids = chunk[1:]

                yield (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long)
                )

                buffer = buffer[self.context_length:]


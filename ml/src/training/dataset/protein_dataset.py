from .read_file import _read_chunk_by_chunk_from_offset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
import random
import torch

class ProteinDataset(IterableDataset):
    def __init__(self, corpus_path, tokenizer,
                  context_length,end_token="<|endofprotein|>", buffer_size=500,
                  resume_state=None, position_array=None, idx_array=None):
        super().__init__()
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.end_token = end_token
        self.buffer_size = buffer_size
        self.resume_state = resume_state or {}
        self.position_array = position_array
        self.idx_array = idx_array

    def __iter__(self):
        token_buffer = []
        shuffle_buffer = []
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        byte_offset, start_idx = self.resume_state.get(worker_id, (0, 0))
        idx = start_idx
        for protein, byte_pos in _read_chunk_by_chunk_from_offset(self.corpus_path, self.end_token, byte_offset):
            current_idx = idx
            idx += 1
            
            if current_idx % num_workers != worker_id:
                continue
            
            tokens = self.tokenizer.encode(protein, allowed_special={self.end_token})
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.context_length + 1:
                chunk = token_buffer[:self.context_length + 1]
                input_ids = chunk[:-1]
                target_ids = chunk[1:]

                sample = (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                    byte_pos,
                    idx,
                )

                shuffle_buffer.append(sample)
                if len(shuffle_buffer) >= self.buffer_size:
                    random_idx = random.randrange(len(shuffle_buffer))
                    yielded = shuffle_buffer.pop(random_idx)
                    if self.position_array is not None:
                        self.position_array[worker_id] = yielded[2]
                        self.idx_array[worker_id] = yielded[3]
                    yield yielded[0], yielded[1]

                token_buffer = token_buffer[self.context_length:]

        while shuffle_buffer:
            random_idx = random.randrange(len(shuffle_buffer))
            yielded = shuffle_buffer.pop(random_idx)
            if self.position_array is not None:
                self.position_array[worker_id] = yielded[2]
                self.idx_array[worker_id] = yielded[3]
            yield yielded[0], yielded[1]
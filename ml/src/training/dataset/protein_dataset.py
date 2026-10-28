from .read_file import _read_chunk_by_chunk_from_offset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import random
import json
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

class ProteinTextDataset(IterableDataset):
    def __init__(self, path, protein_tokenizer,
                  text_tokenizer, max_protein_len=1024,
                    max_text_len=256, resume_state=None,
                    position_array=None, idx_array=None):
        super().__init__()
        
        self.path = path
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_protein_len = max_protein_len
        self.max_text_len = max_text_len
        self.resume_state = resume_state or {}

        self.position_array = position_array
        self.idx_array = idx_array

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        start_offset, start_idx = self.resume_state.get(worker_id, (0, 0))

        with open(self.path, 'r') as f:
            f.seek(start_offset)
            idx = start_idx
            while True:
                line = f.readline()
                if not line:
                    break
                current_position = f.tell()

                if idx % num_workers != worker_id:
                    continue
                
                item = json.loads(line)
                protein_enc = self.protein_tokenizer(
                    item['sequence'],
                    truncation=True,
                    max_length=self.max_protein_len,
                    padding=False,
                    return_tensors=None,
                )

                text_enc = self.text_tokenizer(
                    item['description'],
                    truncation=True,
                    max_length=self.max_text_len,
                    padding=False,
                    return_tensors=None,
                )

                protein_ids = torch.tensor(protein_enc['input_ids'], dtype=torch.long)
                text_ids = torch.tensor(text_enc['input_ids'], dtype=torch.long)

                if self.position_array is not None:
                    self.position_array[worker_id] = current_position
                if self.idx_array is not None:
                    self.idx_array[worker_id] = idx + 1

                yield {
                    "protein_ids": protein_ids,
                    "text_ids": text_ids,
                    "protein_len": protein_ids.size(0),
                    "text_len": text_ids.size(0),
                }
                idx += 1


def make_collate_fn(protein_pad_id, text_pad_id, ignore_index=-100, model_type='p2t'):
    def collate_fn(batch):
        protein_ids = [b['protein_ids']for b in batch]
        text_ids = [b['text_ids'] for b in batch]
        protein_lens = torch.tensor([b['protein_len'] for b in batch])
        text_lens = torch.tensor([b['text_len'] for b in batch])

        protein_ids = pad_sequence(protein_ids, batch_first=True, padding_value=protein_pad_id)
        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=text_pad_id)

        B, P = protein_ids.shape
        _, T = text_ids.shape

        protein_arange = torch.arange(P).unsqueeze(0)
        protein_valid_mask = protein_arange < protein_lens.unsqueeze(1)
        protein_attention_mask = protein_valid_mask.long()

        text_arange = torch.arange(T).unsqueeze(0)
        text_valid_mask = text_arange < text_lens.unsqueeze(1)
        text_attn_mask = text_valid_mask.long()

        if model_type == 'p2t':
            labels = text_ids.clone()
            labels[~text_valid_mask] = ignore_index
        elif model_type == 't2p':
            labels = protein_ids.clone()
            labels[~protein_valid_mask] = ignore_index
        else:
            labels = None


        return {
            "protein_ids": protein_ids,
            "text_idxs": text_ids,
            "protein_attention_mask": protein_attention_mask,
            "attention_mask": text_attn_mask,
            "labels": labels
        }
    return collate_fn
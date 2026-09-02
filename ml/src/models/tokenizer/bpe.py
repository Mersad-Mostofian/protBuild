from collections import Counter, defaultdict
from .word_freq_streaming import _build_word_freq_streaming, _iter_words_from_file
from .packed_tuple import _pack, _unpack, PAIR_BASE
from array import array
from tqdm import tqdm
import random
import json

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.merge_ranks = {}
        self.unk_token = '<|unk|>'
        self.end_token = '<|endofprotein|>'

    def _initialize_vocab(self, unique_chars, special_tokens):
        self.vocab = {
            token_id: token
            for token_id, token in enumerate(unique_chars)
        }

        self.inverse_vocab = {
            token: token_id
            for token_id, token in self.vocab.items()
        }

        for token in sorted(special_tokens):
            if token not in self.inverse_vocab:
                token_id = len(self.vocab)

                self.vocab[token_id] = token
                self.inverse_vocab[token] = token_id

    @staticmethod
    def _reservoir_sample(file_path, end_token, sample_size, seed=0):
        rng = random.Random(seed)

        reservoir = []
        seen = 0

        for protein in _iter_words_from_file(file_path, end_token):
            if not protein:
                continue
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(protein)
                continue
            j = rng.randrange(seen)
            if j < sample_size:
                reservoir[j] = protein
        return reservoir

    @staticmethod
    def _collect_unique_chars(file_path, end_token):
        unique_chars = set()
        for protein in _iter_words_from_file(file_path, end_token):
            unique_chars.update(protein)
        return unique_chars

    def _encode_word_to_array(self, word):
        unk_id = self.inverse_vocab[self.unk_token]
        return array(
            "I",
            (
                self.inverse_vocab.get(char, unk_id)
                for char in word
            )
        )

    def _build_training_words(self, file_path, 
                              end_token, sample_size, seed):
        if sample_size is not None:
            proteins = self._reservoir_sample(
                file_path,
                end_token,
                sample_size,
                seed
            )
            return [
                self._encode_word_to_array(protein)
                for protein in proteins
            ]
        words = []
        for protein in _iter_words_from_file(file_path, end_token):
            if protein:
                words.append(
                    self._encode_word_to_array(protein)
                )
        return words

    @staticmethod
    def _build_pair_statistics(words):
        pair_counts = Counter()
        pair_to_words = defaultdict(set)
        for word_index, word in enumerate(words):
            for i in range(len(word)-1):
                pair = _pack(word[i], word[i+1])
                pair_counts[pair] += 1
                pair_to_words[pair].add(word_index)
        return pair_counts, pair_to_words

    @staticmethod
    def _merge_word(word, best_a, best_b, new_id):
        merged = array("I")
        i = 0
        while i < len(word):
            if(i < len(word) - 1 and word[i] == best_a and word[i+1] == best_b):
                merged.append(new_id)
                i+=2
            else:
                merged.append(word[i])
                i+=1
        return merged

    def train(self, file_path, vocab_size,
               allowed_special=None, sample_size=None, seed=0):
        if allowed_special is None:
            allowed_special = { self.end_token }

        special_tokens = set(allowed_special)
        special_tokens.add(self.unk_token)
        end_token = self.end_token

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than 0"
            )

        if vocab_size >= PAIR_BASE:
            raise ValueError(
                f"vocab_size must be less than PAIR_BASE ({PAIR_BASE})"
            )
        
        self.vocab.clear()
        self.inverse_vocab.clear()
        self.bpe_merges.clear()
        self.merge_ranks.clear()

        unique_chars = self._collect_unique_chars(file_path, end_token)
        self._initialize_vocab(unique_chars, special_tokens)
        del unique_chars

        words = self._build_training_words(
            file_path, end_token, sample_size, seed
        )
        if not words:
            raise ValueError(
                "No protein sequences were found in the corpus."
            )

        pair_counts, pair_to_words = self._build_pair_statistics(words)

        next_id = len(self.vocab)

        with tqdm(total=vocab_size-next_id, desc="Training BPE", unit="merge") as pbar:
            while next_id < vocab_size:
                if not pair_counts:
                    break
                best_packed, best_count = pair_counts.most_common(1)[0]

                if best_count <= 0:
                    break

                best_a, best_b = _unpack(best_packed)
                new_id = next_id
                self.bpe_merges[(best_a, best_b)] = new_id
                self.merge_ranks[(best_a, best_b)] = len(self.merge_ranks)

                affected = pair_to_words.pop(best_packed, set())
                pair_counts.pop(best_packed, None)

                for idx in affected:
                    w = words[idx]

                    for i in range(len(w) - 1):
                        pair = _pack(w[i], w[i+1])
                        pair_counts[pair] -= 1
                        if pair_counts[pair] <= 0:
                            pair_counts.pop(pair, None)
                        pair_to_words[pair].discard(idx)
                    

                    merged = self._merge_word(w, best_a, best_b, new_id)
                    words[idx] = merged

                    for i in range(len(merged)-1):
                        pair = _pack(merged[i], merged[i+1])
                        pair_counts[pair] += 1
                        pair_to_words[pair].add(idx)
                next_id += 1
                pbar.update(1)

        for (a, b), new_id in self.bpe_merges.items():
            token_a = self.vocab[a]
            token_b = self.vocab[b]

            merged_token = (
                token_a, token_b
            )
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

        del words
        del pair_counts
        del pair_to_words
    
    def encode(self, text, allowed_special=None):
        if allowed_special is None:
            allowed_special = set()

        special_tokens = sorted(
            allowed_special,
            key=len,
            reverse=True
        )

        token_ids = []
        i = 0
        while i < len(text):
            matched = False
            for special_token in special_tokens:
                if text.startswith(special_token, i):
                    token_id = self.inverse_vocab.get(special_token)
                    if token_id is None:
                        raise ValueError(f"Special token '{special_tokens}' not found in vocabulary.")

                    token_ids.append(token_id)
                    i+=len(special_token)
                    matched = True
                    break
            if matched:
                continue
            j = i

            while j < len(text):
                if any(text.startswith(special_token, j)
                       for special_token in special_tokens):
                    break
                j+=1
            chunk = text[i:j]
            chunk_ids = self.tokenize(chunk)
            token_ids.extend(chunk_ids)

            i = j
        return token_ids


    def decode(self, token_ids):
        decoded = []
        for token_id in token_ids:
            token = self.vocab.get(token_id)
            if token_id is None:
                raise ValueError(
                    f"Token ID {token_id} not found in vocabulary."
                )
            decoded.append(token)
        return "".join(decoded)

    def tokenize(self, text):
        unk_id = self.inverse_vocab[self.unk_token]

        token_ids = [self.inverse_vocab.get(char, unk_id) 
                     for char in text]
        if len(token_ids) < 2:
            return token_ids

        while len(token_ids) >= 2:
            best_index = None
            best_rank = None

            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i+1])
                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i
            if best_index is None:
                break
            pair = (
                token_ids[best_index],
                token_ids[best_index+1]
            )
            new_id = self.bpe_merges[pair]
            token_ids[best_index: best_index+2] = [new_id]
        return token_ids
    
    def save_vocab_and_merges(self, vocab_path, merges_path):
        with open(vocab_path, 'w', encoding='utf-8') as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        merges_list = []
        for rank, ((a,b), new_id) in enumerate(self.bpe_merges.items()):
            merges_list.append(
                {
                    'pair': [a, b],
                    'new_id': new_id,
                    'rank': rank
                }
            )

        with open(merges_path, 'w', encoding='utf-8') as file:
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, merges_path):
        self.vocab.clear()
        self.inverse_vocab.clear()
        self.bpe_merges.clear()
        self.merge_ranks.clear()

        with open(vocab_path, 'r', encoding='utf-8') as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        with open(merges_path, 'r', encoding='utf-8') as file:
            merges_list = json.load(file)
            for default_rank, merge in enumerate(merges_list):
                pair = tuple(merge['pair'])
                new_id = merge['new_id']
                rank = merge.get("rank", default_rank)
                self.bpe_merges[pair] = new_id
                self.merge_ranks[pair] = rank


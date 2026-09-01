from collections import Counter

def _iter_words_from_file(file_path, end_token, chunk_size=8 * 1024 * 1024):
    buffer=''
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(end_token)
            buffer = parts.pop()
            for p in parts:
                if p:
                    yield p
        if buffer:
            yield buffer

def _build_word_freq_streaming(file_path, end_token):
    word_freq = Counter()
    unique_chars = set()
    for word in _iter_words_from_file(file_path, end_token):
        word_freq[word] += 1
        unique_chars.update(word)
    return word_freq, sorted(unique_chars)

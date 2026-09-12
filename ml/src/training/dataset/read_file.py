def _read_chunk_by_chunk(file_path, end_token, chunk_size=8 * 1024 * 1024):
    buffer = ''
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
                    yield p + end_token
        if buffer:
            yield buffer

def _read_chunk_by_chunk_from_offset(file_path, end_token, offset=0, chunk_size=8 * 1024 * 1024):
    end_token_bytes = end_token.encode('utf-8')
    pos = offset
    leftover = b''

    with open(file_path, 'rb') as f:
        f.seek(offset)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = leftover + chunk
            parts = data.split(end_token_bytes)
            leftover = parts.pop()
            for p in parts:
                if p:
                    pos += len(p) + len(end_token_bytes)
                    yield p.decode('utf-8') + end_token, pos
        if leftover:
            pos += len(leftover)
            yield leftover.decode('utf-8'), pos
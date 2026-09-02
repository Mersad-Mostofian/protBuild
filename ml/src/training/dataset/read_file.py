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
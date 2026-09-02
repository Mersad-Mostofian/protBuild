import torch

def generate_sequnce(
        model,
        idx,
        max_new_tokens,
        context_size,
        end_token_id=None,
        temperature=0,
        topk=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if topk is not None:
            top_logits, _ = torch,topk(logits, topk)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
        if end_token_id is not None:
            if torch.all(idx_next == end_token_id):
                break
    return idx

def generate_protein(model, prompt, tokenizer,
                      max_new_tokens, context_size,
                        device, end_token="<|endofprotein|>",
                        temperature=0, topk=None):
    token_ids = tokenizer.encode(prompt, allowed_special={end_token})
    idx = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    end_token_id = tokenizer.inverse_vocab[end_token]
    idx = generate_sequnce(model, idx, max_new_tokens,
                            context_size, end_token_id=end_token_id,
                            temperature=temperature, topk=topk)

    return tokenizer.decode(idx[0].tolist())
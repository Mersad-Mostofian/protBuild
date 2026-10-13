import torch 

@torch.no_grad()
def predict_protein_description(
        model, protein_tokenizer, text_tokenizer,
        protein_sequence: str, max_new_token: int = 256,
        max_length_protein: int = 1024, device=None 
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    protein_inputs = protein_tokenizer(
        protein_sequence,
        return_tensors='pt',
        truncation=True,
        max_length=max_length_protein,
    )

    protein_ids = protein_inputs['input_ids'].to(device)
    protein_attention_mask = protein_inputs['attention_mask'].to(device)

    generated_ids = model.generate(
        protein_ids=protein_ids,
        protein_attention_mask=protein_attention_mask,
        max_new_tokens=max_new_token
    )

    description = text_tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    return description.strip()
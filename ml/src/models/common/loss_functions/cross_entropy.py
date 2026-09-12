from torch.nn import functional as F

def calculate_loss_batch(input_batch, target_batch, model, device, unk_id=None):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten(),
                            ignore_index= unk_id if unk_id is not None else -100)
    return loss

def calculate_loss_loader(data_loader, model, device, unk_id=None, num_batches=None):
    total_loss = 0.0
    batch_count = 0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        loss = calculate_loss_batch(input_batch, target_batch,
                                     model, device, unk_id)
        total_loss += loss.item()
        batch_count += 1

    if batch_count == 0:
        return float('nan')

    return total_loss / batch_count

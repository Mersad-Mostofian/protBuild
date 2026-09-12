import torch
from ml.src.models.common.loss_functions.cross_entropy import calculate_loss_loader

def evaluate_model(model, train_loader, val_loader,
                    device, eval_iter, unk_id=None):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device, unk_id, eval_iter)
        val_loss = calculate_loss_loader(val_loader, model, device, unk_id, eval_iter)
    model.train()
    return train_loss, val_loss
import torch
from ml.src.models.common.loss_functions.cross_entropy import calculate_loss_loader

def evaluate_model(model, train_loader, val_loader,
                    device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device)
        val_loss = calculate_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss
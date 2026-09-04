import torch
from ml.src.models.common.loss_functions.cross_entropy import calculate_loss_loader
from ml.src.training.evaluator.evaluate import evaluate_model
from ml.src.inference.sequence_generator import generate_protein
from .checkpoint import save_model

def train_model(model, train_loader, val_loader, optimizer,
                 device, num_epochs, eval_freq, eval_iter,
                   checkpoint_path='ml/src/training/checkpoint', save_file_name=None):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, steps = 0, -1

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_loss_loader(train_loader, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            steps += 1

            if steps % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader,
                                                      val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {steps}):")
                print(f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                if save_file_name is not None:
                    save_model(model, checkpoint_path, save_file_name)

    return train_losses, val_losses, track_tokens_seen
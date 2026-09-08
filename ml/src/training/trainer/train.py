import os
import torch
from tqdm import tqdm
from ml.src.models.common.loss_functions.cross_entropy import calculate_loss_batch
from ml.src.training.evaluator.evaluate import evaluate_model
from ml.src.inference.sequence_generator import generate_protein
from .checkpoint import save_model

def train_model(model, train_loader, val_loader, num_train, num_val, optimizer,
                 device, num_epochs, eval_freq, eval_iter, unk_id,
                   checkpoint_path='ml/src/training/checkpoint', save_file_name=None):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, steps = 0, -1
    start_epoch = 0

    checkpoint_file = os.path.join(
        checkpoint_path,
        save_file_name
    )
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        steps = checkpoint["steps"]
        tokens_seen = checkpoint["tokens_seen"]
    model.to(device)

    train_batch_size = train_loader.batch_size
    val_batch_size = val_loader.batch_size

    train_batches = (num_train + train_batch_size - 1) // train_batch_size
    val_batches = (num_val + val_batch_size - 1) // val_batch_size

    for epoch in range(start_epoch, num_epochs, 1):
        model.train()
        progress_bar = tqdm(
            train_loader,
            total=train_batches,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch"
        )
        for input_batch, target_batch in progress_bar:
            optimizer.zero_grad()
            loss = calculate_loss_batch(input_batch, target_batch, model, device, unk_id)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            steps += 1

            progress_bar.set_postfix(loss=f"{loss.item():.3f}")

            if steps % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader,
                                                      val_loader, device, unk_id, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {steps}):")
                print(f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                if save_file_name is not None:
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "steps": steps,
                        "tokens_seen": tokens_seen,
                    }
                    save_model(checkpoint, checkpoint_path, save_file_name)

    return train_losses, val_losses, track_tokens_seen
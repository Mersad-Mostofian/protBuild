import os
import math
import torch
from tqdm import tqdm
from ml.src.models.common.loss_functions.cross_entropy import calculate_loss_batch
from ml.src.training.evaluator.evaluate import evaluate_model
from ml.src.inference.sequence_generator import generate_protein
from .checkpoint import save_model

def train_model(model, train_loader, val_loader, train_val_loader, optimizer, scheduler,
                 device, num_epochs, eval_freq, eval_iter, train_position_array, train_idx_array,
                   num_workers, checkpoint_path='ml/src/training/checkpoint', save_file_name=None, unk_id=None):
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
            map_location='cpu',
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
                
        start_epoch = checkpoint["epoch"]
        steps = checkpoint["steps"]
        tokens_seen = checkpoint["tokens_seen"]

        del checkpoint
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    for epoch in range(start_epoch, num_epochs, 1):
        model.train()
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch"
        )
        for input_batch, target_batch in progress_bar:
            optimizer.zero_grad()
            loss = calculate_loss_batch(input_batch, target_batch, model, device, unk_id)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            steps += 1

            progress_bar.set_postfix(loss=f"{loss.item():.3f}")

            if steps % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_val_loader,
                                                      val_loader, device, eval_iter, unk_id)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {steps}):")
                print(f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                print(f"Train perplexity {math.exp(train_loss):.3f}, Val perplexity {math.exp(val_loss):.3f}")

                if save_file_name is not None:
                    worker_byte_offsets = {i: train_position_array[i] for i in range(len(train_position_array))}
                    worker_next_idx     = {i: train_idx_array[i] for i in range(len(train_idx_array))}
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "steps": steps,
                        "tokens_seen": tokens_seen,
                        "worker_byte_offsets": worker_byte_offsets,
                        "worker_next_idx": worker_next_idx,
                        "num_workers": num_workers,
                    }
                    save_model(checkpoint, checkpoint_path, save_file_name)
    
    return train_losses, val_losses, track_tokens_seen
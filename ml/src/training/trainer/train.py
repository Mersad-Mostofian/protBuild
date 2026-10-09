import os
import math
import torch
from torchmetrics.text import BLEUScore, ROUGEScore, BERTScore
from tqdm import tqdm
from ml.src.models.common.loss_functions.cross_entropy import calculate_loss_batch
from ml.src.training.evaluator.evaluate import evaluate_model
from ml.src.inference.sequence_generator import generate_protein
from .checkpoint import save_model
from multiprocessing import Array
from torch.utils.data import DataLoader

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


def train_model_protein2text(model, tokenizer, train_dataset, val_loader, optimizer, scheduler,
                 device, num_epochs, eval_freq, eval_iter, train_position_array, train_idx_array, 
                 checkpoint_iter, collate_fn, checkpoint_path=None, save_file_name=None, num_workers=1, batch_size=1, max_new_tokens=256):
    metrics_history = dict()
    steps = 0

    checkpoint_file = None
    if checkpoint_path is not None and save_file_name is not None:
        checkpoint_file = f"{checkpoint_path}/{save_file_name}"

    start_epoch = 0
    resume_state = {}

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint = None

    if checkpoint_file:
        try:
            checkpoint = torch.load(
                checkpoint_file,
                map_location='cpu'
            )
            print(f"\nLoading checkpoint from: {checkpoint_file}")

            model_state_dict = checkpoint['model_state_dict']
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)

            optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

            if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict']
                )
            steps = checkpoint.get('steps', 0)
            saved_epoch = checkpoint.get('epoch', 0)
            metrics_history = checkpoint.get('metrics_history', {})

            saved_offsets = checkpoint.get('worker_byte_offsets', {})
            saved_indices = checkpoint.get('worker_next_idx', {})
            saved_num_workers = checkpoint.get('num_workers', num_workers)

            if saved_num_workers != num_workers:
                raise RuntimeError(f"\nNumber of workers in checkpoint does not match current number of workers. ({saved_num_workers} | {num_workers})")

            for worker_id in range(num_workers):
                offset = int(saved_offsets.get(worker_id, 0))
                idx = int(saved_indices.get(worker_id, 0))
                resume_state[worker_id] = (offset, idx)
                train_position_array[worker_id] = offset
                train_idx_array[worker_id] = idx

            start_epoch = saved_epoch
            print("Checkpoint loaded successfully.")
            print(f'Resuming from: \nEpoch: {start_epoch+1}\nStep: {steps}')
            print(f"Worker resume state: \n{resume_state}")
        except FileNotFoundError:
            print("No checkpoint found.")

    train_dataset.resume_state = resume_state
    train_dataset.position_array = train_position_array
    train_dataset.idx_array = (train_idx_array)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    del checkpoint

    bleu_metric = BLEUScore(n_gram=4)
    rouge_metric = ROUGEScore()
    bert_metric = BERTScore(model_name_or_path="bert-base-uncased")


    for epoch in range(start_epoch, num_epochs):
        model.train()

        running_train_loss = 0.0 
        running_train_steps = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch"
        )
        for batch in progress_bar:

            protein_ids = batch['protein_ids'].to(device, non_blocking=True)
            protein_attention_mask = batch['protein_attention_mask'].to(device, non_blocking=True)
            text_idxs = batch['text_idxs'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            optimizer.zero_grad()

            output = model(
                protein_ids=protein_ids,
                protein_attention_mask=protein_attention_mask,
                text_idxs=text_idxs,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = loss.item()
            running_train_loss += loss_value
            running_train_steps += 1
            steps += 1

            progress_bar.set_postfix(loss=f"{loss_value:.3f}")

            if steps % eval_freq == 0:
                model.eval()
                total_val_loss = 0.0
                num_val_batches = 0

                predictions = []
                references = []

                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        if i >= eval_iter:
                            break
                        protein_ids = batch[
                            "protein_ids"
                        ].to(device, non_blocking=True)

                        protein_attention_mask = batch[
                            "protein_attention_mask"
                        ].to(device, non_blocking=True)

                        text_idxs = batch[
                            "text_idxs"
                        ].to(device, non_blocking=True)

                        attention_mask = batch[
                            "attention_mask"
                        ].to(device, non_blocking=True)

                        labels = batch[
                            "labels"
                        ].to(device, non_blocking=True)

                        output = model(
                            protein_ids=protein_ids,
                            protein_attention_mask=protein_attention_mask,
                            text_idxs=text_idxs,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        total_val_loss += output.loss.item()
                        num_val_batches += 1  

                        generated_ids = model.generate(
                            protein_ids=protein_ids,
                            protein_attention_mask=protein_attention_mask,
                            max_new_tokens=max_new_tokens,
                        )

                        preds = tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                        decode_labels = labels.clone()
                        decode_labels[decode_labels == -100] = tokenizer.pad_token_id

                        refs = tokenizer.batch_decode(
                            decode_labels,
                            skip_special_tokens=True
                        )

                        predictions.extend(preds)
                        references.extend(refs)
                
                if num_val_batches > 0:
                    val_loss = (
                        total_val_loss / num_val_batches
                    )
                else:
                    val_loss = float('nan')

                if running_train_steps > 0:
                    train_loss = (
                        running_train_loss / running_train_steps
                    )
                else:
                    train_loss = float('nan')

                bleu_score = bleu_metric(
                    predictions,
                    [[ref] for ref in references]
                )

                rouge_scores = rouge_metric(
                    predictions,
                    references
                )

                bert_scores = bert_metric(
                    predictions,
                    references
                )

                bleu_value = bleu_score.item()
                rouge1_value = rouge_scores['rouge1_fmeasure'].item()
                rouge2_value = rouge_scores['rouge2_fmeasure'].item()
                rougeL_value = rouge_scores['rougeL_fmeasure'].item()
                bert_precision = bert_scores['precision'].mean().item()
                bert_recall = bert_scores['recall'].mean().item()
                bert_f1 = bert_scores['f1'].mean().item()
                train_perplexity = math.exp(train_loss)
                val_perplexity = math.exp(val_loss)

                metrics_history[f'step-{steps}'] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_perplexity': train_perplexity,
                    'val_perplexity': val_perplexity,
                    'bleu': bleu_value,
                    'rouge1': rouge1_value,
                    'rouge2': rouge2_value,
                    'rougeL': rougeL_value,
                    'bertscore_precision': bert_precision,
                    'bertscore_recall': bert_recall,
                    'bertscore_f1': bert_f1
                }

                print(f"Epoch {epoch+1} (Step {steps}):")
                print(f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                print(f"Train perplexity {train_perplexity:.3f}, Val perplexity {val_perplexity:.3f}")
                print(f"BLEU-4: {bleu_value:.4f}")
                print(f"ROUGE-1 F1: {rouge1_value:.4f}")
                print(f"ROUGE-2 F1: {rouge2_value:.4f}")
                print(f"ROUGE-L F1: {rougeL_value:.4f}")
                print(f"BERTScore Precision: {bert_precision:.4f}")
                print(f"BERTScore Recall: {bert_recall:.4f}")
                print(f"BERTScore F1: {bert_f1:.4f}")

                running_train_loss = 0.0
                running_train_steps = 0
                model.train()
            
            if save_file_name is not None and steps % checkpoint_iter == 0:
                worker_byte_offsets = {
                    i: train_position_array[i]
                    for i in range(num_workers)
                }
                worker_next_idx = {
                    i: train_idx_array[i]
                    for i in range(num_workers)
                }

                checkpoint = {
                    "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (scheduler.state_dict()
                        if scheduler is not None else None),
                    "epoch": epoch,
                    "steps": steps,
                    "metrics_history": metrics_history,
                    "worker_byte_offsets": worker_byte_offsets,
                    "worker_next_idx": worker_next_idx,
                    "num_workers": num_workers,
                }

                save_model(checkpoint, checkpoint_path, save_file_name)
                

    
    return metrics_history
import torch
from ml.src.models.common.loss_functions.cross_entropy import calculate_loss_loader
from tqdm import tqdm
import evaluate


def evaluate_model(model, train_loader, val_loader,
                    device, eval_iter, unk_id=None):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device, unk_id, eval_iter)
        val_loss = calculate_loss_loader(val_loader, model, device, unk_id, eval_iter)
    model.train()
    return train_loss, val_loss

def evaluate_protein2text(model, test_loader, text_tokenizer, device,
                           max_new_tokens=256, bert_model="dmis-lab/biobert-v1.1",
                           bert_batch_size=8):

    model.eval()

    bleu_metric = evaluate.load('bleu')
    rouge_metric = evaluate.load('rouge')
    bert_metric = evaluate.load('bertscore')

    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            protein_ids = batch['protein_ids'].to(device)
            protein_mask = batch['protein_attention_mask'].to(device)

            text_ids = batch['text_idxs']
            text_mask = batch['attention_mask']

            generated_ids = model.generate(
                protein_ids=protein_ids,
                protein_attention_mask=protein_mask,
                max_new_tokens=max_new_tokens
            )
            batch_predictions = text_tokenizer.batch_decode(
                generated_ids.cpu(),
                skip_special_tokens=True,
            )

            batch_references = []

            for i in range(text_ids.size(0)):
                valid_text_ids = text_ids[i][text_mask[i].bool()]
                refrence = text_tokenizer.decode(
                    valid_text_ids,
                    skip_special_tokens=True,
                )

                batch_references.append(refrence.strip())
            predictions.extend(
                [p.strip() for p in batch_predictions]
            )
            references.extend(batch_references)
        print(f"\nTotal evaluated samples: {len(predictions)}")

        bleu_results = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )

        rouge_results = rouge_metric.compute(
            predictions=predictions,
            references=references
        )

        bert_results = bert_metric.compute(
            predictions=predictions,
            references=references,
            model_type=bert_model,
            num_layers=12,
            device='cpu',
            batch_size=bert_batch_size
        )

        bert_precision = sum(bert_results['precision']) / len(bert_results['precision'])
        bert_recall = sum(bert_results['recall']) / len(bert_results['recall'])
        bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])

        results = {
            "num_samples": len(predictions),
            "bleu-4": bleu_results['bleu'],
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "bertscore_precision": bert_precision,
            "bertscore_recall": bert_recall,
            "bertscore_f1": bert_f1,
        }

        for metric, value in results.items():
            if metric == "num_samples":
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")
        return results, predictions, references
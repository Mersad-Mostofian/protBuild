import pickle as pkl
import pandas as pd
import re

FEATURES = [
    'FUNCTION',
    'SUBCELLULAR LOCATION',
    'SIMILARITY',
]

def parse_annotation(anno):
    row = {}
    pattern = r'(' + '|'.join(map(re.escape, FEATURES)) + r'):'
    matches = list(re.finditer(pattern, anno))

    for i, match in enumerate(matches):
        feature = match.group(1)
        start = match.end()
        if i + 1 < len(matches):
            end = matches[i+1].start()
        else:
            end = len(anno)
        value = anno[start:end].strip()
        row[feature] = value
    return row


def pkl_to_csv(seq_pkl_file, anno_pkl_file, output_path):
    rows = []
    with open(seq_pkl_file, 'rb') as seq_file, \
        open(anno_pkl_file, 'rb') as anno_file:
        seqs = pkl.load(seq_file)
        annos = pkl.load(anno_file)
    for seq, anno in zip(seqs, annos):
        row = {
            'sequence': seq
        }
        row.update(parse_annotation(anno))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

pkl_to_csv(
    'ml/datasets/clean_seq.pkl',
    'ml/datasets/clean_anno.pkl',
    'ml/datasets/prot2text.csv'
)
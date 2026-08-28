import sys, os
import csv
import re


def parse_header(header: str) -> dict:
    header = header.lstrip('>')

    parts = header.split('|', 2)

    database = parts[0]
    accession = parts[1]
    rest = parts[2]

    protein_name_match = re.match(r'(.+?)\s+OS=', rest)

    protein_name = ''
    if protein_name_match:
        protein_name = protein_name_match.group(1).strip()

    def extract(pattern):
        match = re.search(pattern, rest)
        return match.group(1) if match else ''

    organism = extract(r"OS=(.*?)\s+OX=")
    taxonomy_id = extract(r"OX=(\d+)")
    gene = extract(r"GN=(.*?)\s+PE=")
    protein_evidence = extract(r"PE=(\d+)")
    sequence_version = extract(r"SV=(\d+)")


    return {
        "database": database,
        "accession": accession,
        "protein_id": parts[2].split(" ", 1)[0],
        "protein_name": protein_name,
        "organism": organism,
        "taxonomy_id": taxonomy_id,
        "gene": gene,
        "protein_evidence": protein_evidence,
        "sequence_version": sequence_version,
    }


def fasta_to_csv(input_file: str, output_file:str):

    records = []

    current_header = None
    current_sequence = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('>'):
                if current_header is not None:
                    record = parse_header(header=current_header)

                    sequence = "".join(current_sequence)

                    record['sequence'] = sequence
                    record['length'] = len(sequence)

                    records.append(record)

                current_header = line
                current_sequence = []

            else:
                current_sequence.append(line)

        if current_header is not None:
            record = parse_header(current_header)
            sequence = "".join(current_sequence)

            record['sequence'] = sequence
            record['length'] = len(sequence)

            records.append(record)
    fieldnames = [
        'database',
        'accession',
        'protein_id',
        'protein_name',
        'organism',
        'taxonomy_id',
        'gene',
        'protein_evidence',
        'sequence_version',
        'sequence',
        'length',
    ]

    with open(output_file, 'w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(records)

    print(f'\nConverted {len(records):,} proteins')
    print(f'Output: {output_file}')


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print('\nPlease provice Fasta file.')
        print('\nUsage:\nparse_fasta.py input.fasta output.csv\n')
        sys.exit()

    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print('\nUsage:\nparse_fasta.py input.fasta output.csv\n')
        sys.exit()

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f'\nError: File {input_file} is not exist!')
        sys.exit()

    output_file = 'output.csv'

    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    fasta_to_csv(
        input_file=input_file,
        output_file=output_file
    )
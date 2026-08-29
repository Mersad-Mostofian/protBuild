import sys, os
import csv
import re

def uniport_parser(header: str) -> dict:
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
            "status": "",
            "cluster_size": "",
            "representative_id": "",
    }


def uniparc_parser(header: str) -> dict:
    parts = header.split(maxsplit=1)
    
    protein_id = parts[0]
    status = ''

    if len(parts) > 1:
        status_match = re.search(r'status=(\S+)', parts[1])
        if status_match:
            status = status_match.group(1)

    return {
        "database": "UniParc",
        "accession": protein_id,
        "protein_id": protein_id,
        "protein_name": "",
        "organism": "",
        "taxonomy_id": "",
        "gene": "",
        "protein_evidence": "",
        "sequence_version": "",
        "status": status,
        "cluster_size": "",
        "representative_id": "",
    }

def uniref_parser(header: str) -> dict:
    def extract(pattern):
        match = re.search(pattern, header)
        return match.group(1) if match else ''
    
    accession_match = re.match(r'(UniRef\d+)_(\S+)', header)

    database = ''
    accession = ''

    if accession_match:
        database = accession_match.group(1)
        accession = accession_match.group(2)

    protein_name_match = re.match(
        r'UniRef\d+_\S+\s+(.+?)\s+n=\d+\s+Tax=',
        header
    )

    protein_name = (
        protein_name_match.group(1).strip()
        if protein_name_match
        else ''
    )

    organism = extract(r'Tax=(.*?)\s+TaxID=')
    taxonomy_id = extract(r'TaxID=(\d+)')
    representative_id = extract(r'RepID=(\S+)')
    cluster_size = extract(r'n=(\d+)')

    return {
        "database": database,
        "accession": accession,
        "protein_id": accession,
        "protein_name": protein_name,
        "organism": organism,
        "taxonomy_id": taxonomy_id,
        "gene": "",
        "protein_evidence": "",
        "sequence_version": "",
        "status": "",
        "cluster_size": cluster_size,
        "representative_id": representative_id,
    }

def parse_header(header: str, dataset_category: str) -> dict:
    header = header.lstrip('>')

    if dataset_category == 'uniprot':
        return uniport_parser(header)
    elif dataset_category == 'uniparc':
        return uniparc_parser(header)
    elif dataset_category == 'uniref':
        return uniref_parser(header)
    else:
        raise ValueError(
            f"Unknown dataset category: {dataset_category}"
        )


    


def fasta_to_csv(input_file: str, output_file:str, dataset_category: str):

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
                    record = parse_header(header=current_header, dataset_category=dataset_category)

                    sequence = "".join(current_sequence)

                    record['sequence'] = sequence
                    record['length'] = len(sequence)

                    records.append(record)

                current_header = line
                current_sequence = []

            else:
                current_sequence.append(line)

        if current_header is not None:
            record = parse_header(current_header, dataset_category)
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
        'status',
        'cluster_size',
        'representative_id',
    ]

    with open(output_file, 'w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(records)

    print(f'\nConverted {len(records):,} proteins')
    print(f'Output: {output_file}')


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print('\nPlease provide dataset category and FASTA file.')
        print('\nUsage:')
        print('parse_fasta.py <dataset_category> <input.fasta> [output.csv]')
        sys.exit(1)

    if sys.argv[1] in ('-h', '--help'):
        print('\nUsage:')
        print('parse_fasta.py <dataset_category> <input.fasta> [output.csv]')
        print('\nDataset categories:')
        print('  uniprot')
        print('  uniparc')
        print('  uniref')
        sys.exit(0)

    if len(sys.argv) < 3:
        print('\nError: Dataset category and FASTA file are required.')
        print('\nUsage:')
        print('parse_fasta.py <dataset_category> <input.fasta> [output.csv]')
        sys.exit(1)

    dataset_category = sys.argv[1]
    input_file = sys.argv[2]

    valid_categories = {
        'uniprot',
        'uniparc',
        'uniref',
    }

    if dataset_category not in valid_categories:
        print(f'\nError: Unknown dataset category: {dataset_category}')
        print(f'Available categories: {", ".join(valid_categories)}')
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f'\nError: File {input_file} does not exist!')
        sys.exit(1)

    output_file = 'output.csv'

    if len(sys.argv) > 3:
        output_file = sys.argv[3]

    fasta_to_csv(
        input_file=input_file,
        output_file=output_file,
        dataset_category=dataset_category,
    )

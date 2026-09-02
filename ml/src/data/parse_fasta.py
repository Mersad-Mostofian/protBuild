import sys, os
import csv
import re
from tqdm import tqdm

FIELDNAMES = [
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

CHUNK_SIZE = 5000

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

def iter_records(input_file: str, dataset_category: str):
    current_header = None
    current_sequence_parts = []

    def build_record(header, seq_parts):
        record = parse_header(header=header, dataset_category=dataset_category)
        sequence = "".join(seq_parts)
        record['sequence'] = sequence
        record['length'] = len(sequence)
        return record

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('>'):
                if current_header is not None:
                    yield build_record(header=current_header, seq_parts=current_sequence_parts)

                current_header = line
                current_sequence_parts = []

            else:
                current_sequence_parts.append(line)
        if current_header is not None:
            yield build_record(current_header, current_sequence_parts)



def fasta_to_csv(input_file: str, output_file:str, dataset_category: str, chunk_size: int = CHUNK_SIZE):
    total = 0
    chunk = []
    with open(output_file, 'w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        with tqdm(desc='Parsing FASTA', unit=' protein', unit_scale=True) as pbar:
            for record in iter_records(input_file, dataset_category):
                chunk.append(record)
                total+=1
                pbar.update(1)
                if len(chunk) >= chunk_size:
                    writer.writerows(chunk)
                    f.flush()
                    chunk.clear()
            if chunk:
                writer.writerows(chunk)
                f.flush()

    print(f'\nConverted {total:,} proteins')
    print(f'Output: {output_file}')


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print('\nPlease provide dataset category and FASTA file.')
        print('\nUsage:')
        print('parse_fasta.py <dataset_category> <input.fasta> [output.csv] [chunk_size]')
        sys.exit(1)

    if sys.argv[1] in ('-h', '--help'):
        print('\nUsage:')
        print('parse_fasta.py <dataset_category> <input.fasta> [output.csv] [chunk_size]')
        print('\nDataset categories:')
        print('  uniprot')
        print('  uniparc')
        print('  uniref')
        sys.exit(0)

    if len(sys.argv) < 3:
        print('\nError: Dataset category and FASTA file are required.')
        print('\nUsage:')
        print('parse_fasta.py <dataset_category> <input.fasta> [output.csv] [chunk_size]')
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

    chunk_size = CHUNK_SIZE
    if len(sys.argv) > 4:
        try:
            chunk_size = int(sys.argv[4])
        except ValueError:
            print(f'\nError: chunk_size must be an integer, got: {sys.argv[4]}')
            sys.exit(1)


    fasta_to_csv(
        input_file=input_file,
        output_file=output_file,
        dataset_category=dataset_category,
        chunk_size=chunk_size
    )

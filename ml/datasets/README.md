# Datasets

This directory contains datasets used by **ProtBuild**.

## 1. UniProtKB/Swiss-Prot

**Description:** Reviewed protein sequences from UniProtKB/Swiss-Prot, including basic protein metadata such as accession, protein name, organism, gene, and amino acid sequence.


**Source:** UniProt

**Download:**

[UniProtKB/Swiss-Prot FTP](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/)

**Main FASTA file:**

```text
uniprot_sprot.fasta.gz
```

**Download with `wget`:**

```bash
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
```

Extract:

```bash
gunzip uniprot_sprot.fasta.gz
```


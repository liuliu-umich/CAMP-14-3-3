import argparse
import sys
import os
import uuid
import pandas as pd
from pathlib import Path
from typing import List, Optional
import torch,gc

# Adding PhosphoLingo to Python path
phospholingo_dir = Path("src/files/PhosphoLingo/phosphoLingo").resolve()
sys.path.insert(0, str(phospholingo_dir))

# Import predict after setting the sys.path
import predict

def add_at_after_st(seq: str, position: int) -> Optional[str]:
    try:
        if position < 1 or position > len(seq):
            raise ValueError(f"Position {position} out of range (1-{len(seq)})")

        residue = seq[position-1]
        if residue not in ['s', 't']:
            raise ValueError(f"Residue {residue} at position {position} is not S/T")

        return seq[:position] + "@" + seq[position:]

    except (IndexError, ValueError) as e:
        error_msg = f"Position: {position}, Sequence: {seq[:30]}{'...' if len(seq)>30 else ''}, Error: {str(e)}"
        with open("processing_errors.log", "a") as f:
            f.write(error_msg + "\n")
        return None

def create_fasta_file(sequence_ids: List[str], modified_sequences: List[str], output_path: Path) -> None:
    with open(output_path, 'w') as fasta_file:
        for seq_id, seq in zip(sequence_ids, modified_sequences):
            if seq is not None:
                fasta_file.write(f">{seq_id}\n{seq}\n")

def predict_fasta_file(input_fasta: Path, output_csv: Path, model_loc: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predict.run_predict(
        str(Path(model_loc).resolve()),
        str(input_fasta),
        str(output_csv)
    )

def extract_phospholingo_score(sequences, sites, model_loc, sequence_ids = ["default"]):
    if len(sequence_ids) != len(sequences) or len(sequences) != len(sites):
        raise ValueError("All input lists must have the same length")

    modified_sequences = []
    valid_entries = []
    for seq_id, seq, site in zip(sequence_ids, sequences, sites):
        modified = add_at_after_st(seq, site)
        if modified:
            modified_sequences.append(modified)
            valid_entries.append((seq_id, seq, site))

    if not modified_sequences:
        raise ValueError("No valid sequences to process")

    temp_id = uuid.uuid4().hex
    temp_fasta = Path(f"temp_{temp_id}.fasta")
    temp_output = Path(f"temp_{temp_id}_predictions.csv")

    try:
        create_fasta_file(
            [e[0] for e in valid_entries],
            modified_sequences,
            temp_fasta)

        predict_fasta_file(temp_fasta, temp_output, model_loc)

        results_df = pd.read_csv(temp_output)

        original_data = pd.DataFrame({
            'unique_id': [e[0] for e in valid_entries],
            'original_sequence': [e[1] for e in valid_entries],
            'original_site': [e[2] for e in valid_entries]})

        merged = original_data.merge(
            results_df,
            left_on=['unique_id', 'original_site'],
            right_on=['prot_id', 'position'],
            how='left')

        result_cols = ['unique_id', 'pred']
        return merged[result_cols].rename(columns={'pred': 'phospho_score'})

    finally:
        for f in [temp_fasta, temp_output]:
            if f.exists():
                f.unlink()

def clear_cuda_memory():
    """Aggressive memory cleaning for CUDA"""
    if torch.cuda.is_available():
        print("torch.cuda.is_available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        print("torch.cuda.not available")
    gc.collect()

if __name__ == "__main__":

    model_loc = r"H:\PhosphoLingo_ST_new.ckpt"
    sequence_ids = ["test1"]
    sequences = ["AsAAAARASKKKKKK"]
    sites = [2]
    clear_cuda_memory()
    scores = extract_phospholingo_score(sequences, sites, model_loc, sequence_ids)
    print(scores)
import argparse
import gzip
import os
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# Set thread limits before importing heavy ML dependencies from features.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from features import predict_all_sites


DEFAULT_FASTA_PATH = Path("/scratch/luvul_root/luvul0/luvul/proteomes/UP000005640_9606.fasta.gz")
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/luvul_root/luvul0/luvul/proteomes/human_proteome_st_site_predictions"
)
DEFAULT_FINAL_SITE_OUTPUT_NAME = "human_proteome_st_site_scores.csv"
DEFAULT_THRESHOLD = 0.58
DEFAULT_BATCH_SIZE = 25

SITE_OUTPUT_COLUMNS = [
    "row_id",
    "record_id",
    "uniprot_accession",
    "description",
    "sequence_length",
    "site",
    "residue",
    "prediction",
    "predicted_positive_model",
    "predicted_positive_threshold",
    "positive_probability",
    "negative_probability",
    "threshold",
]


def extract_accession(record_id):
    parts = record_id.split("|")
    return parts[1] if len(parts) >= 2 else record_id


def ensure_output_dirs(output_dir):
    batch_dir = output_dir / "batches"
    completed_batch_dir = output_dir / "completed_rows"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    completed_batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, completed_batch_dir


def load_fasta_records(fasta_path, max_proteins=None):
    open_fn = gzip.open if fasta_path.suffix == ".gz" else open
    rows = []
    with open_fn(fasta_path, "rt") as handle:
        for row_id, record in enumerate(SeqIO.parse(handle, "fasta")):
            sequence = str(record.seq).strip().upper()
            rows.append(
                {
                    "row_id": int(row_id),
                    "record_id": record.id,
                    "uniprot_accession": extract_accession(record.id),
                    "description": record.description,
                    "sequence": sequence,
                    "sequence_length": int(len(sequence)),
                    "candidate_site_count": int(sum(residue in {"S", "T"} for residue in sequence)),
                }
            )

    fasta_df = pd.DataFrame(rows)
    if max_proteins is not None:
        fasta_df = fasta_df.head(int(max_proteins)).copy()
    return fasta_df


def list_batch_files(batch_dir):
    return sorted(batch_dir.glob("batch_*.csv"))


def load_processed_row_ids(batch_dir, completed_batch_dir):
    processed = set()
    for search_dir in [completed_batch_dir, batch_dir]:
        for batch_file in list_batch_files(search_dir):
            batch_df = pd.read_csv(batch_file, usecols=["row_id"])
            processed.update(batch_df["row_id"].astype(int).tolist())
    return processed


def save_batch(batch_df, batch_index, batch_dir):
    batch_path = batch_dir / f"batch_{batch_index:04d}.csv"
    batch_df.to_csv(batch_path, index=False)
    return batch_path


def save_completed_rows(batch_input_df, batch_index, completed_batch_dir):
    completed_path = completed_batch_dir / f"batch_{batch_index:04d}.csv"
    batch_input_df[["row_id"]].drop_duplicates().to_csv(completed_path, index=False)
    return completed_path


def combine_saved_batches(batch_dir, final_output, sort_columns):
    batch_files = list_batch_files(batch_dir)
    if not batch_files:
        return pd.DataFrame(columns=SITE_OUTPUT_COLUMNS)

    combined_df = pd.concat((pd.read_csv(batch_file) for batch_file in batch_files), ignore_index=True)
    combined_df = combined_df.sort_values(sort_columns).reset_index(drop=True)
    combined_df.to_csv(final_output, index=False)
    return combined_df


def predict_protein_sites(row, threshold):
    row_id = int(row["row_id"])
    sequence = row["sequence"].strip()

    if not sequence or int(row["candidate_site_count"]) == 0:
        return []

    try:
        site_results_df, _, _ = predict_all_sites(sequence)
        site_results_df = site_results_df.sort_values(["site"]).reset_index(drop=True)
        site_results_df["predicted_positive_threshold"] = (
            site_results_df["positive_probability"] >= float(threshold)
        )

        site_rows = []
        for site_row in site_results_df.itertuples():
            site_rows.append(
                {
                    "row_id": row_id,
                    "record_id": row["record_id"],
                    "uniprot_accession": row["uniprot_accession"],
                    "description": row["description"],
                    "sequence_length": int(row["sequence_length"]),
                    "site": int(site_row.site),
                    "residue": site_row.residue,
                    "prediction": site_row.prediction,
                    "predicted_positive_model": bool(site_row.predicted_positive),
                    "predicted_positive_threshold": bool(site_row.predicted_positive_threshold),
                    "positive_probability": float(site_row.positive_probability),
                    "negative_probability": float(site_row.negative_probability),
                    "threshold": float(threshold),
                }
            )

        return site_rows
    except Exception as exc:
        print(f"Error scoring {row['record_id']}: {exc}")
        return []


def run_batch_scoring(
    fasta_path,
    output_dir,
    threshold=DEFAULT_THRESHOLD,
    batch_size=DEFAULT_BATCH_SIZE,
    max_proteins=None,
    final_output_name=DEFAULT_FINAL_SITE_OUTPUT_NAME,
):
    batch_dir, completed_batch_dir = ensure_output_dirs(output_dir)
    final_site_output = output_dir / final_output_name

    proteome_df = load_fasta_records(fasta_path, max_proteins=max_proteins)
    processed_row_ids = load_processed_row_ids(batch_dir, completed_batch_dir)
    remaining_df = proteome_df[~proteome_df["row_id"].isin(processed_row_ids)].copy()

    print(f"Total proteins: {len(proteome_df)}")
    print(f"Already processed: {len(processed_row_ids)}")
    print(f"Remaining: {len(remaining_df)}")
    print(f"Threshold: {threshold}")
    print(f"Batch size: {batch_size}")
    print(f"Final site output: {final_site_output}")

    if len(remaining_df) == 0:
        print("All proteins are already processed.")
    else:
        for batch_index, start in enumerate(range(0, len(proteome_df), batch_size), start=1):
            batch_path = batch_dir / f"batch_{batch_index:04d}.csv"
            completed_path = completed_batch_dir / f"batch_{batch_index:04d}.csv"

            if batch_path.exists() or completed_path.exists():
                print(f"Skipping {batch_path.name}: existing chunk file detected.")
                continue

            batch_input_df = proteome_df.iloc[start : start + batch_size].copy()
            batch_input_df = batch_input_df[~batch_input_df["row_id"].isin(processed_row_ids)].copy()

            if batch_input_df.empty:
                print(f"Skipping {batch_path.name}: all rows in this chunk are already processed.")
                continue

            site_results = []
            for _, row in batch_input_df.iterrows():
                site_results.extend(predict_protein_sites(row, threshold))

            site_batch_df = pd.DataFrame(site_results, columns=SITE_OUTPUT_COLUMNS)
            batch_path = save_batch(site_batch_df, batch_index, batch_dir)
            completed_path = save_completed_rows(batch_input_df, batch_index, completed_batch_dir)
            processed_row_ids.update(batch_input_df["row_id"].astype(int).tolist())

            print(
                f"Saved {batch_path.name} and {completed_path.name}: "
                f"rows {int(batch_input_df['row_id'].min())} to {int(batch_input_df['row_id'].max())}"
            )

    final_site_df = combine_saved_batches(
        batch_dir,
        final_site_output,
        sort_columns=["row_id", "site"],
    )

    print(f"Site-level scores saved to: {final_site_output}")
    print(f"Site rows in final output: {len(final_site_df)}")
    return final_site_df


def build_parser():
    parser = argparse.ArgumentParser(
        description="Score all S/T sites from a FASTA file with resumable batch output."
    )
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA_PATH, help="Input FASTA or FASTA.GZ file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that stores batches, completed rows, and final CSV.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Site probability threshold.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Proteins per batch.")
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Optionally score only the first N proteins from the FASTA.",
    )
    parser.add_argument(
        "--final-output-name",
        default=DEFAULT_FINAL_SITE_OUTPUT_NAME,
        help="Filename for the combined final site-level CSV inside --output-dir.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_batch_scoring(
        fasta_path=args.fasta_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_proteins=args.max_proteins,
        final_output_name=args.final_output_name,
    )


if __name__ == "__main__":
    main()

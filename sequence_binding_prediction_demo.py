import argparse
import json
import os
from pathlib import Path

import pandas as pd


# Set thread limits before importing heavy ML dependencies from features.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from features import predict_sequence_binding


def build_example_input():
    return pd.DataFrame(
        {
            "sequence_name": [
                "protein_1",
                "protein_2",
            ],
            "sequence": [
                "ASAAAAAASAAAAAAT",
                "MSSQSHPDGLSGRDQPVELLNPARVNHMPSTVD",
            ],
        }
    )


def load_input_dataframe(input_csv, name_col, sequence_col):
    if input_csv is None:
        return build_example_input()

    df = pd.read_csv(input_csv)
    missing = [col for col in [name_col, sequence_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    return df[[name_col, sequence_col]].rename(
        columns={
            name_col: "sequence_name",
            sequence_col: "sequence",
        }
    )


def predict_sequences_from_dataframe(df, threshold):
    result_rows = []

    for _, row in df.iterrows():
        sequence_name = row["sequence_name"]
        sequence = row["sequence"]
        print(f"Scoring {sequence_name}...")

        try:
            summary, site_results_df, _, _ = predict_sequence_binding(sequence, threshold=threshold)

            positive_sites = site_results_df.loc[
                site_results_df["predicted_positive"], "site"
            ].astype(int).tolist()

            positive_site_probabilities_df = site_results_df.loc[
                site_results_df["predicted_positive"], ["site", "positive_probability"]
            ].copy()
            positive_site_probabilities_df["positive_probability"] = positive_site_probabilities_df[
                "positive_probability"
            ].round(6)

            positive_site_probabilities = [
                {
                    "site": int(site_row["site"]),
                    "probability": float(site_row["positive_probability"]),
                }
                for _, site_row in positive_site_probabilities_df.iterrows()
            ]

            result_rows.append(
                {
                    "sequence_name": sequence_name,
                    "positive_sites": positive_sites,
                    "positive_site_probabilities": positive_site_probabilities,
                    "sequence_binding_14_3_3": bool(summary["predicted_positive"]),
                    "best_site": int(summary["best_site"]),
                    "best_residue": summary["best_residue"],
                    "best_site_probability": float(summary["max_probability"]),
                    "site_count": int(summary["site_count"]),
                    "prediction_status": "ok",
                    "error_message": "",
                }
            )
        except Exception as exc:
            result_rows.append(
                {
                    "sequence_name": sequence_name,
                    "positive_sites": [],
                    "positive_site_probabilities": [],
                    "sequence_binding_14_3_3": None,
                    "best_site": None,
                    "best_residue": None,
                    "best_site_probability": None,
                    "site_count": None,
                    "prediction_status": "error",
                    "error_message": str(exc),
                }
            )

    return pd.DataFrame(result_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run sequence-level 14-3-3 binding prediction outside Jupyter."
    )
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional CSV input file.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output file.")
    parser.add_argument("--threshold", type=float, default=0.58, help="Sequence-level threshold.")
    parser.add_argument(
        "--name-col",
        default="sequence_name",
        help="Column name for sequence identifiers when --input-csv is used.",
    )
    parser.add_argument(
        "--sequence-col",
        default="sequence",
        help="Column name for sequences when --input-csv is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally score only the first N rows from the input.",
    )
    args = parser.parse_args()

    input_df = load_input_dataframe(args.input_csv, args.name_col, args.sequence_col)
    if args.limit is not None:
        input_df = input_df.head(args.limit).copy()

    print(f"Rows to score: {len(input_df)}")
    print(f"Threshold: {args.threshold}")
    if args.input_csv is None:
        print("Using built-in example input.")
    else:
        print(f"Input CSV: {args.input_csv}")

    output_df = predict_sequences_from_dataframe(input_df, threshold=args.threshold)

    printable_df = output_df.copy()
    printable_df["positive_sites"] = printable_df["positive_sites"].apply(json.dumps)
    printable_df["positive_site_probabilities"] = printable_df["positive_site_probabilities"].apply(json.dumps)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        printable_df.to_csv(args.output_csv, index=False)
        print(f"Saved output to: {args.output_csv}")

    print(printable_df.to_string(index=False))


if __name__ == "__main__":
    main()

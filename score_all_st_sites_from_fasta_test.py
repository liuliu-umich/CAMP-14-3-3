import argparse
from pathlib import Path

import pandas as pd

from score_all_st_sites_from_fasta import DEFAULT_FASTA_PATH, load_fasta_records, predict_all_sites


DEFAULT_TEST_PROTEIN_COUNT = 3
DEFAULT_TEST_SITES_TO_SHOW_PER_PROTEIN = 10


def run_quick_test(
    fasta_path,
    protein_count=DEFAULT_TEST_PROTEIN_COUNT,
    sites_to_show_per_protein=DEFAULT_TEST_SITES_TO_SHOW_PER_PROTEIN,
    max_proteins=None,
):
    proteome_df = load_fasta_records(fasta_path, max_proteins=max_proteins)
    test_df = proteome_df[proteome_df["candidate_site_count"] > 0].head(protein_count).copy()

    test_summary_rows = []
    test_site_rows = []

    for _, row in test_df.iterrows():
        try:
            site_results_df, _, _ = predict_all_sites(row["sequence"])
            ranked_df = site_results_df.sort_values(
                ["positive_probability", "site"], ascending=[False, True]
            ).reset_index(drop=True)
            best_row = ranked_df.iloc[0]

            test_summary_rows.append(
                {
                    "row_id": int(row["row_id"]),
                    "uniprot_accession": row["uniprot_accession"],
                    "sequence_length": int(row["sequence_length"]),
                    "candidate_site_count": int(row["candidate_site_count"]),
                    "scored_site_count": int(len(site_results_df)),
                    "best_site": int(best_row["site"]),
                    "best_residue": best_row["residue"],
                    "best_site_probability": float(best_row["positive_probability"]),
                    "test_status": "ok",
                    "error_message": "",
                }
            )

            preview_df = ranked_df.head(sites_to_show_per_protein).copy()
            for preview_row in preview_df.itertuples():
                test_site_rows.append(
                    {
                        "row_id": int(row["row_id"]),
                        "uniprot_accession": row["uniprot_accession"],
                        "site": int(preview_row.site),
                        "residue": preview_row.residue,
                        "prediction": preview_row.prediction,
                        "predicted_positive_model": bool(preview_row.predicted_positive),
                        "positive_probability": float(preview_row.positive_probability),
                        "negative_probability": float(preview_row.negative_probability),
                    }
                )
        except Exception as exc:
            test_summary_rows.append(
                {
                    "row_id": int(row["row_id"]),
                    "uniprot_accession": row["uniprot_accession"],
                    "sequence_length": int(row["sequence_length"]),
                    "candidate_site_count": int(row["candidate_site_count"]),
                    "scored_site_count": None,
                    "best_site": None,
                    "best_residue": None,
                    "best_site_probability": None,
                    "test_status": "error",
                    "error_message": str(exc),
                }
            )

    test_results_df = pd.DataFrame(test_summary_rows)
    test_site_preview_df = pd.DataFrame(test_site_rows)

    print(f"Tested {len(test_results_df)} proteins from {Path(fasta_path).name}")
    print(f"Showing up to {sites_to_show_per_protein} scored sites per test protein")
    print()
    print("Summary:")
    print(test_results_df.to_string(index=False))
    print()
    print("Top scored sites:")
    print(test_site_preview_df.to_string(index=False))
    return test_results_df, test_site_preview_df


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the quick test block for S/T site scoring from a FASTA file."
    )
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA_PATH, help="Input FASTA or FASTA.GZ file.")
    parser.add_argument(
        "--protein-count",
        type=int,
        default=DEFAULT_TEST_PROTEIN_COUNT,
        help="Number of proteins with candidate S/T sites to test.",
    )
    parser.add_argument(
        "--sites-to-show-per-protein",
        type=int,
        default=DEFAULT_TEST_SITES_TO_SHOW_PER_PROTEIN,
        help="Top scored sites to print per test protein.",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Optionally limit the FASTA loading step to the first N proteins.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_quick_test(
        fasta_path=args.fasta_path,
        protein_count=args.protein_count,
        sites_to_show_per_protein=args.sites_to_show_per_protein,
        max_proteins=args.max_proteins,
    )


if __name__ == "__main__":
    main()

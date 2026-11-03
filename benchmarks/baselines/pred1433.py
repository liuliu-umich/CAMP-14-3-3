"""Validate and read cached original 14-3-3-Pred server scores."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=BASE / "data/sites.csv")
    parser.add_argument("--server-scores", type=Path, default=BASE / "data/server_scores.csv")
    parser.add_argument("--output", type=Path, default=BASE / "results/pred1433_predictions.csv")
    args = parser.parse_args()
    sites, scores = pd.read_csv(args.sites), pd.read_csv(args.server_scores)
    keys = ["sample_id", "uniprot_id", "site_1based", "label"]
    data = sites.merge(scores, on=keys, how="left", validate="one_to_one", sort=False)
    expected = []
    for row in data.itertuples(index=False):
        seq, pos = row.sequence.upper(), int(row.site_1based)
        expected.append(seq[max(0, pos - 7):pos - 1].rjust(6, "-") +
                        seq[pos - 1].lower() + seq[pos:pos + 4].ljust(4, "-"))
    if not data.Peptide.str.upper().eq(pd.Series(expected).str.upper()).all():
        raise ValueError("Server peptides do not match the requested sequence and site.")
    if not np.isfinite(data[["ANN", "PSSM", "SVM", "Consensus"]].to_numpy()).all():
        raise ValueError("Missing/nonfinite server scores.")
    result = data[keys + ["ANN", "PSSM", "SVM", "Consensus"]].rename(columns={
        "ANN": "score_1433pred_ann", "PSSM": "score_1433pred_pssm",
        "SVM": "score_1433pred_svm", "Consensus": "score_1433pred_consensus"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Validated {len(result)} cached server records; scores are unmodified.")


if __name__ == "__main__":
    main()

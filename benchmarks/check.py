"""Check the released inputs and reproduce the recorded benchmark values."""

import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .analysis import BASE, MODELS, load_sites, evaluate_sites, calibration, proteome_recovery
except ImportError:
    from analysis import BASE, MODELS, load_sites, evaluate_sites, calibration, proteome_recovery


def main():
    with (BASE / "data/checksums.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (BASE / row["file"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["sha256"], row["file"]
    sites = load_sites()
    metrics, _, _ = evaluate_sites(sites)
    gold = pd.read_csv(BASE / "data/reference_metrics.csv").set_index(["cohort", "model"])
    columns = ["AUROC", "average_precision", "MCC", "F1", "precision", "recall",
               "accuracy", "macro_precision", "macro_recall", "macro_F1", "TP", "TN", "FP", "FN"]
    for row in metrics.to_dict("records"):
        expected = gold.loc[(row["cohort"], row["model"])]
        for name in columns:
            assert np.isclose(row[name], expected[name], atol=1e-12, rtol=0), (row["model"], name)
        selected = sites.loc[sites.in_test77 == 1] if row["cohort"] == "test77" else sites
        values = selected[MODELS[row["model"]]].to_numpy()
        positive, negative = values[selected.label == 1], values[selected.label == 0]
        pairwise_auc = np.mean((positive[:, None] > negative) + .5 * (positive[:, None] == negative))
        assert np.isclose(pairwise_auc, row["AUROC"], atol=1e-12, rtol=0)
    _, diagnostic = calibration(sites)
    gold_calibration = pd.read_csv(BASE / "data/reference_calibration.csv").set_index("model").loc["CAMP"]
    assert np.isclose(diagnostic.iloc[0].Brier, gold_calibration.raw_score_brier, atol=1e-12)
    assert np.isclose(diagnostic.iloc[0].ECE, gold_calibration.ECE_5_fixed_bins, atol=1e-12)
    recovery, _, _ = proteome_recovery(pd.read_csv(BASE / "data/proteins.csv"))
    gold_recovery = pd.read_csv(BASE / "data/reference_recovery.csv").set_index(["model", "top_K"])
    for row in recovery.to_dict("records"):
        expected = gold_recovery.loc[(row["model"], row["top_K"])]
        assert row["tie_averaged_known_hits"] == expected.tie_averaged_known_hits
        assert np.isclose(row["matched_random_mean"], expected.matched_random_mean_hits, atol=1e-12)
    print("Passed: input hashes, eight model/cohort reports, pairwise AUROC, calibration and six recovery cutoffs.")


if __name__ == "__main__":
    main()

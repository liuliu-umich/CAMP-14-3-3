"""Combine checked server scores, IUPred2A and the authors' PTM normalization."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=BASE / "data/sites.csv")
    parser.add_argument("--server-scores", type=Path, default=BASE / "data/server_scores.csv")
    parser.add_argument("--normalization", type=Path, default=BASE / "assets/sitefinder/Normalization Database including CST.csv")
    parser.add_argument("--iupred-dir", type=Path, default=BASE.parent / "src/files/iupred2a")
    parser.add_argument("--cached-disorder", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "results/sitefinder_features.csv")
    args = parser.parse_args()
    keys = ["sample_id", "uniprot_id", "site_1based", "label"]
    sites = pd.read_csv(args.sites)
    data = sites.merge(pd.read_csv(args.server_scores), on=keys, how="left", validate="one_to_one", sort=False)
    normalization = {}
    records = pd.read_csv(args.normalization, low_memory=False)
    for row in records.to_dict("records"):
        match = re.fullmatch(r"([ST])(\d+)-p", str(row["MOD_RSD"]))
        if match:
            try:
                value = float(row["Percentages"])
                value = value if np.isfinite(value) else None
            except (TypeError, ValueError):
                value = None
            normalization.setdefault((row["ACC_ID"], int(match.group(2))), set()).add(value)
    if args.cached_disorder:
        cache = pd.read_csv(args.cached_disorder)
        cached = data[keys].merge(cache[keys + ["Iupred.score"]], on=keys, how="left", validate="one_to_one")
        disorder = dict(zip(cached.sample_id, cached["Iupred.score"]))
    else:
        sys.path.insert(0, str(args.iupred_dir.resolve()))
        import iupred2a_lib
        sequence_scores = {}
        disorder = {}
        for row in data.itertuples(index=False):
            seq = row.sequence.upper()
            if seq not in sequence_scores:
                sequence_scores[seq] = iupred2a_lib.iupred(seq, mode="long", new_smoothing=False)[0]
            disorder[row.sample_id] = sequence_scores[seq][int(row.site_1based) - 1]
    rows = []
    for row in data.to_dict("records"):
        seq, pos = row["sequence"].upper(), int(row["site_1based"])
        expected = seq[max(0, pos - 7):pos - 1].rjust(6, "-") + seq[pos - 1].lower() + seq[pos:pos + 4].ljust(4, "-")
        if expected.upper() != str(row["Peptide"]).upper():
            raise ValueError("Server peptide mismatch: " + row["sample_id"])
        observed = normalization.get((row["uniprot_id"], pos), {0.0})
        if len(observed) != 1:
            raise ValueError("Ambiguous PTM normalization: " + row["sample_id"])
        result = {k: row[k] for k in keys}
        result.update({"Amino.Acid": seq[pos - 1], "Iupred.score": disorder[row["sample_id"]],
                       "normalization": next(iter(observed)), "status": "ready",
                       **{k: row[k] for k in ["ANN", "PSSM", "SVM"]}})
        rows.append(result)
    result = pd.DataFrame(rows)
    if not np.isfinite(result[["Iupred.score", "normalization", "ANN", "PSSM", "SVM"]].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite Site-Finder features.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Checked sequence/site mapping and prepared {len(result)} Site-Finder inputs.")


if __name__ == "__main__":
    main()

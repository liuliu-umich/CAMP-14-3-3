"""Join baseline outputs by sample identity, position and label."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=BASE / "data/sites.csv")
    parser.add_argument("--camp", type=Path, default=BASE / "results/camp_predictions.csv")
    parser.add_argument("--bindsite", type=Path, default=BASE / "results/bindsite_predictions.csv")
    parser.add_argument("--sitefinder", type=Path, default=BASE / "results/sitefinder_predictions.csv")
    parser.add_argument("--pred1433", type=Path, default=BASE / "results/pred1433_predictions.csv")
    parser.add_argument("--output", type=Path, default=BASE / "results/rerun_predictions.csv")
    args = parser.parse_args()
    keys = ["sample_id", "uniprot_id", "site_1based", "label"]
    result = pd.read_csv(args.sites)[keys]
    for path, column in [(args.camp, "score_camp"), (args.bindsite, "score_bindsite_ensemble"),
                         (args.sitefinder, "score_sitefinder"), (args.pred1433, "score_1433pred_consensus")]:
        table = pd.read_csv(path)
        result = result.merge(table[keys + [column]], on=keys, how="outer", validate="one_to_one", sort=False, indicator=True)
        if not result._merge.eq("both").all():
            raise ValueError("Incomplete or inconsistent sample identities: " + str(path))
        result = result.drop(columns="_merge")
    if not np.isfinite(result.drop(columns=keys).to_numpy()).all():
        raise ValueError("Missing/nonfinite model predictions.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Joined all four tools on the same {len(result)} sites.")


if __name__ == "__main__":
    main()

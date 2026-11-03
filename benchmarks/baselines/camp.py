"""Score candidate sites with the released CAMP checkpoint."""

import argparse
import hashlib
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=BASE / "data/sites.csv")
    parser.add_argument("--features", type=Path, default=BASE / "data/camp_features.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/1433model_20260223.pkl")
    parser.add_argument("--output", type=Path, default=BASE / "results/camp_predictions.csv")
    parser.add_argument("--recompute-features", action="store_true")
    args = parser.parse_args()
    sites = pd.read_csv(args.sites)
    if sites.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs.")
    raw = args.checkpoint.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != "b89853e47d57b4665ef688cc2bcf15cc08118a68":
        raise ValueError("Checkpoint differs from the pinned released CAMP model.")
    model = joblib.load(args.checkpoint)
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    classifier.set_params(n_jobs=1, predictor="cpu_predictor")
    columns = classifier.get_booster().feature_names
    if columns is None or len(columns) != 44:
        raise ValueError("Expected the released 44-feature checkpoint.")
    if args.recompute_features:
        sys.path.insert(0, str(ROOT))
        import features
        records = []
        # Preserve neighboring lowercase S/T markers in mutant contexts.
        for sequence, group in sites.groupby("sequence", sort=False):
            positions = list(dict.fromkeys(group.site_1based.astype(int)))
            frame, index, _ = features.extract_features_for_sites(sequence, positions)
            if index.site.tolist() != positions:
                raise ValueError("Feature/site mapping failed.")
            by_site = dict(zip(positions, frame.to_dict("records")))
            records.extend(dict(sample_id=row.sample_id, **by_site[int(row.site_1based)])
                           for row in group.itertuples(index=False))
        table = pd.DataFrame(records)
    else:
        table = pd.read_csv(args.features)
    aligned = sites[["sample_id"]].merge(table, on="sample_id", how="left", validate="one_to_one", sort=False)
    if not np.isfinite(aligned[columns].to_numpy(float)).all():
        raise ValueError("Missing or nonfinite CAMP features.")
    result = sites[["sample_id", "uniprot_id", "site_1based", "label"]].copy()
    result["score_camp"] = model.predict_proba(aligned[columns])[:, list(model.classes_).index(1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} sites with the released CAMP checkpoint.")


if __name__ == "__main__":
    main()

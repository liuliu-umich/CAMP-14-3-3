"""Rebuild the frozen test inputs from the public source CSV and provenance."""

import argparse
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE.parent / "datasets/final_dataset_with_seq.csv")
    parser.add_argument("--provenance", type=Path, default=BASE / "data/provenance.csv")
    parser.add_argument("--output", type=Path, default=BASE / "results/rebuilt_sites.csv")
    args = parser.parse_args()
    with args.provenance.open(newline="") as handle:
        provenance = {r["sample_id"]: r for r in csv.DictReader(handle)}
    rows = []
    with args.source.open(newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            if row["usage"] != "independent test":
                continue
            sid = f"camp_row_{line}"
            info = provenance[sid]
            assert row["uniprot_id"] == info["uniprot_id"]
            assert row["site"] == info["site_1based"] and row["label"] == info["label"]
            sequence, site = row["sequence"], int(row["site"])
            seq = sequence.upper()
            assert 1 <= site <= len(seq) and seq[site - 1] in "ST"
            peptide = seq[max(0, site - 8):site - 1].rjust(7, "-") + seq[site - 1].lower() + seq[site:site + 7].ljust(7, "-")
            primary = (info["provenance"] == "curated_reference_match"
                       and not any(int(info[k]) for k in ["CAMP_train_protein_overlap",
                                   "BindSite_development_exact_peptide_overlap", "SiteFinder_train_protein_overlap"]))
            rows.append(dict(sample_id=sid, uniprot_id=row["uniprot_id"], site_1based=site,
                             label=int(row["label"]), sequence=sequence, peptide_15aa=peptide,
                             mutation=row["human Site and Mutation"], parent_id=row["augmentated from uniprot ID"],
                             provenance=info["provenance"], reference_PMIDs=info["reference_PMIDs"],
                             CAMP_train_protein_overlap=int(info["CAMP_train_protein_overlap"]),
                             BindSite_development_exact_peptide_overlap=int(info["BindSite_development_exact_peptide_overlap"]),
                             SiteFinder_train_protein_overlap=int(info["SiteFinder_train_protein_overlap"]), in_test77=int(primary)))
    assert len(rows) == 529 and sum(r["in_test77"] for r in rows) == 77
    assert sum(r["label"] for r in rows if r["in_test77"]) == 66
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Rebuilt 529 inputs; 77 reference-matched sites retained as the primary cohort.")


if __name__ == "__main__":
    main()

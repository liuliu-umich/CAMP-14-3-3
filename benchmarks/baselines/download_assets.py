"""Download pinned baseline assets and verify their recorded file hashes."""

import argparse
import csv
import hashlib
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["bindsite", "sitefinder"])
    parser.add_argument("--destination", type=Path, default=BASE / "assets")
    args = parser.parse_args()
    with (BASE / "data/assets.csv").open(newline="") as handle:
        records = [r for r in csv.DictReader(handle) if r["model"] == args.model]
    for row in records:
        target = args.destination / row["model"] / row["path"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]:
            continue
        with urllib.request.urlopen(row["url"], timeout=90) as response:
            raw = response.read()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != row["git_blob"]:
            raise ValueError("Source Git blob mismatch: " + row["path"])
        content = raw.replace(b"\r\n", b"\n") if row["text"] == "1" else raw
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Asset SHA256 mismatch: " + row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)
        print(row["model"], row["path"], "verified")


if __name__ == "__main__":
    main()

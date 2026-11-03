"""Score 15-residue windows with the three published BindSite models."""

import argparse
import ast
import csv
import hashlib
import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
FILENAMES = {"mlp": "single_model_lab_results_26_okt_factor_1178.h5",
             "cnn": "model_lab_30_oktober1random_seed4269.h5",
             "gru": "gru_blos_BN_False4_sep_with_end_result_lab.h5"}
CONTROLS = [("GLMQRSSsFPYTTKG", .98), ("SAKSRLQtAPVPMPD", .94),
            ("FPRARTVsVSVPPHG", .87), ("SGDRSGYsSPGSPGT", .85),
            ("NNNNRKTsNGDDSLF", .69), ("KGLPRPKsAGTATQM", .68),
            ("GRELRRMsDEFVDSF", .65), ("APGKRSEsGSPSAGG", .61),
            ("EEAGIGDtPSLEDEA", .000), ("LHEETPPtTSSDSEE", .001)]


def load_predictor(assets):
    with (BASE / "data/assets.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "bindsite":
                continue
            raw = (assets / row["path"]).read_bytes()
            raw = raw.replace(b"\r\n", b"\n") if row["text"] == "1" else raw
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Published BindSite asset mismatch: " + row["path"])
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # Load only the original encoding functions and their direct dependencies.
    tree = ast.parse((assets / "library/models/encodings.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"embed_prep", "aa_list_to_tens"}]
    if len(functions) != 2:
        raise ValueError("Published encoding functions are missing.")
    namespace = dict(np=np, pd=pd, tf=tf)
    exec(compile(ast.Module(body=functions, type_ignores=[]), "encodings.py", "exec"), namespace)
    import json
    config = json.loads((assets / "examples/configurations/MLP_embed_BN_False/best_configuration.json").read_text())
    embedding = ast.literal_eval(config[0]["dict_encoding"])
    if embedding != {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYst")}:
        raise ValueError("Unexpected published MLP encoding.")
    with (assets / "library/models/dict_encodings/blosom_dict.pkl").open("rb") as handle:
        blosum = pickle.load(handle)
    models = {name: tf.keras.models.load_model(assets / "models" / filename, compile=False)
              for name, filename in FILENAMES.items()}

    def score(peptides):
        if not all(len(p) == 15 and p[7] in "st" for p in peptides):
            raise ValueError("Expected 15-mers with a lowercase central s/t.")
        embedded = namespace["embed_prep"]([list(p) for p in peptides], list(embedding), 15, embedding)
        encoded = namespace["aa_list_to_tens"]([list(p) for p in peptides], 15, blosum)
        values = {name: model(embedded if name == "mlp" else encoded, training=False).numpy().ravel()
                  for name, model in models.items()}
        values["ensemble"] = np.mean(list(values.values()), axis=0)
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite BindSite output.")
        return values

    observed = score([p for p, _ in CONTROLS])["ensemble"]
    if np.max(np.abs(observed - [v for _, v in CONTROLS])) > .03:
        raise ValueError("Published Table 3 examples did not match within the fixed 0.03 tolerance.")
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=BASE / "data/sites.csv")
    parser.add_argument("--assets", type=Path, default=BASE / "assets/bindsite")
    parser.add_argument("--output", type=Path, default=BASE / "results/bindsite_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    sites = pd.read_csv(args.sites)
    if sites.sample_id.duplicated().any() or args.batch_size < 1:
        raise ValueError("Duplicate samples or invalid batch size.")
    score = load_predictor(args.assets)
    result = sites[["sample_id", "uniprot_id", "site_1based", "label"]].copy()
    batches = [score(sites.peptide_15aa.iloc[i:i + args.batch_size].tolist())
               for i in range(0, len(sites), args.batch_size)]
    for name in [*FILENAMES, "ensemble"]:
        result["score_bindsite_" + name] = np.concatenate([part[name] for part in batches])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Published examples checked; scored {len(result)} sites without retraining.")


if __name__ == "__main__":
    main()

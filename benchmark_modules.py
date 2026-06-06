import argparse
import os
import statistics
import time
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_SEQUENCE = "GQPKAAPSVTLFPPSSEELQANKATLVCLVSDFNPGAVTVAWKADGSPVKVGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCRVTHEGSTVEKTVAPAECS"
DEFAULT_SITE = 8
DEFAULT_REPEATS = 3
DEFAULT_PHOSPHOLINGO = "/Users/newuser/PhosphoLingo_ST_new.ckpt"


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def summarize(values):
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "best": min(values),
        "worst": max(values),
    }


def print_summary(name, cold_elapsed, warm_values, extra=""):
    summary = summarize(warm_values)
    suffix = f" {extra}" if extra else ""
    print(
        f"{name}: cold={cold_elapsed:.3f}s warm_avg={summary['avg']:.3f}s "
        f"warm_median={summary['median']:.3f}s warm_best={summary['best']:.3f}s "
        f"warm_worst={summary['worst']:.3f}s{suffix}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark module-level 14-3-3 feature pipeline timings.")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, help="Protein sequence to benchmark.")
    parser.add_argument("--site", type=int, default=DEFAULT_SITE, help="1-based S/T site to benchmark.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Warm-cache repeats.")
    parser.add_argument(
        "--phospholingo-model",
        type=Path,
        default=Path(DEFAULT_PHOSPHOLINGO),
        help="Path to the PhosphoLingo checkpoint.",
    )
    args = parser.parse_args()

    os.environ["PHOSPHOLINGO_MODEL_LOC"] = str(args.phospholingo_model.resolve())

    import features

    sequence = args.sequence.strip()
    site = int(args.site)
    wt_seq = features.filter_sequence(sequence)
    seq = wt_seq.upper()
    site_specific_sequence = features.make_site_specific_sequence(wt_seq, site)

    print(f"Sequence length: {len(sequence)}")
    print(f"Benchmark site: {site}")
    print(f"Warm repeats: {args.repeats}")
    print(f"PhosphoLingo checkpoint: {features.PHOSPHOLINGO_MODEL_LOC}")
    print()

    tokenizer, model, _ = features._get_esm_resources()

    benchmarks = []

    def run_benchmark(name, fn, fn_args, extra_builder=None):
        cold_elapsed, cold_result = timed_call(fn, *fn_args)
        warm_values = []
        warm_result = cold_result
        for _ in range(args.repeats):
            elapsed, warm_result = timed_call(fn, *fn_args)
            warm_values.append(elapsed)
        extra = extra_builder(warm_result) if extra_builder else ""
        print_summary(name, cold_elapsed, warm_values, extra)
        benchmarks.append((name, cold_elapsed, summarize(warm_values)))
        return warm_result

    deephase_score_df = run_benchmark(
        "DeePhase",
        features.extract_seq_deephase_score,
        (seq,),
        lambda df: f"deephase_score={float(df['deephase_score'].iloc[0]):.3f}",
    )

    idr_df = run_benchmark(
        "IUPred/ANCHOR",
        features.build_idr_dataframe,
        (seq,),
        lambda df: f"rows={len(df)}",
    )

    protein_embedding_df, residue_embeddings = run_benchmark(
        "ESM Embedding",
        features.extract_esm_embeddings_for_sequence,
        (seq, tokenizer, model),
        lambda result: f"residues={len(result[1])}",
    )

    phospholingo_df = run_benchmark(
        "PhosphoLingo",
        features.extract_phospholingo_score,
        ([site_specific_sequence], [site], features.PHOSPHOLINGO_MODEL_LOC, [f"site_{site}"]),
        lambda df: f"score={float(df['phospho_score'].iloc[0]):.3f}",
    )

    _ = run_benchmark(
        "One-hot Window",
        features.extract_onehot_embedding,
        (wt_seq, site),
        lambda df: f"cols={df.shape[1]}",
    )

    idr_seq = features.extract_idr_seq_from_df(idr_df, site)
    compactness_cache = {}
    compactness_result = run_benchmark(
        "Compactness",
        features.build_compactness_score_df,
        (idr_seq, compactness_cache),
        lambda df: "idr_seq=None" if idr_seq is None else f"idr_len={len(idr_seq)}",
    )

    _ = run_benchmark(
        "Bio Feature Assembly",
        features.build_bio_features_from_context,
        (site_specific_sequence, site, deephase_score_df, idr_df, compactness_cache),
        lambda df: f"cols={df.shape[1]}",
    )

    phospho_score = float(phospholingo_df["phospho_score"].iloc[0])
    _ = run_benchmark(
        "PLM Feature Assembly",
        features.build_plm_features_from_context,
        (site, protein_embedding_df, residue_embeddings, phospho_score),
        lambda df: f"cols={df.shape[1]}",
    )

    feature_df, site_df, filtered_seq = run_benchmark(
        "Full Feature Pipeline (1 site)",
        features.extract_features_for_sites,
        (sequence, [site]),
        lambda result: f"features={result[0].shape[1]}",
    )

    predict_model = features._get_predict_model()
    _ = run_benchmark(
        "Final Classifier Predict",
        predict_model.predict_proba,
        (feature_df,),
        lambda probs: f"positive_prob={float(probs[0][1]):.3f}",
    )

    candidate_sites = features.find_candidate_sites(sequence)
    _, _, _ = run_benchmark(
        "Full Feature Pipeline (all sites)",
        features.extract_features_for_sites,
        (sequence, candidate_sites),
        lambda result: f"sites={len(result[1])} features={result[0].shape[1]}",
    )

    print()
    print("Relative warm cost by module:")
    ordered = sorted(benchmarks, key=lambda item: item[2]["avg"], reverse=True)
    for name, _, summary in ordered:
        print(f"{name}: {summary['avg']:.3f}s")


if __name__ == "__main__":
    main()

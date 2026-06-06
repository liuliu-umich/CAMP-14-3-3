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


DEFAULT_SEQUENCE = "MSSQSHPDGLSGRDQPVELLNPARVNHMPSTVD"
DEFAULT_SITE = 2
DEFAULT_REPEATS = 3
DEFAULT_PHOSPHOLINGO = "/Users/newuser/PhosphoLingo_ST_new.ckpt"


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def summarize(name, values):
    avg = statistics.mean(values)
    median = statistics.median(values)
    best = min(values)
    worst = max(values)
    print(
        f"{name}: runs={len(values)} avg={avg:.3f}s median={median:.3f}s best={best:.3f}s worst={worst:.3f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark 14-3-3 inference pipeline.")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, help="Protein sequence to benchmark.")
    parser.add_argument("--site", type=int, default=DEFAULT_SITE, help="1-based S/T site for single-site benchmark.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Warm-cache repeats per function.")
    parser.add_argument(
        "--phospholingo-model",
        type=Path,
        default=Path(DEFAULT_PHOSPHOLINGO),
        help="Path to the PhosphoLingo checkpoint.",
    )
    args = parser.parse_args()

    os.environ["PHOSPHOLINGO_MODEL_LOC"] = str(args.phospholingo_model.resolve())

    import features

    print(f"Sequence length: {len(args.sequence)}")
    print(f"Benchmark site: {args.site}")
    print(f"Warm repeats: {args.repeats}")
    print(f"PhosphoLingo checkpoint: {features.PHOSPHOLINGO_MODEL_LOC}")
    print()

    functions = [
        ("predict_score", features.predict_score, (args.sequence, args.site)),
        ("predict_all_sites", features.predict_all_sites, (args.sequence,)),
        ("predict_sequence_binding", features.predict_sequence_binding, (args.sequence,)),
    ]

    for name, fn, fn_args in functions:
        cold_elapsed, result = timed_call(fn, *fn_args)
        print(f"{name} cold run: {cold_elapsed:.3f}s")

        warm_values = []
        for _ in range(args.repeats):
            elapsed, result = timed_call(fn, *fn_args)
            warm_values.append(elapsed)
        summarize(f"{name} warm runs", warm_values)

        if name == "predict_score":
            prediction, probability, prediction_features = result
            print(
                f"  output: prediction={prediction} positive_prob={float(probability[1]):.3f} "
                f"features={prediction_features.shape[1]}"
            )
        elif name == "predict_all_sites":
            results_df, prediction_features, _ = result
            print(
                f"  output: sites={len(results_df)} top_prob={float(results_df['positive_probability'].max()):.3f} "
                f"features={prediction_features.shape[1]}"
            )
        else:
            summary, results_df, prediction_features, _ = result
            print(
                f"  output: bind={bool(summary['predicted_positive'])} best_site={int(summary['best_site'])} "
                f"top_prob={float(summary['max_probability']):.3f} sites={len(results_df)} "
                f"features={prediction_features.shape[1]}"
            )
        print()


if __name__ == "__main__":
    main()

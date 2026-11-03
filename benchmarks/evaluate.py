"""Recalculate all tables and figures from the supplied frozen predictions."""

import argparse
import pandas as pd

try:
    from .analysis import (BASE, load_sites, evaluate_sites, calibration,
                           prevalence_analysis, proteome_recovery, save_tables)
    from .plot_figures import (plot_benchmark, plot_calibration, plot_prevalence,
                              plot_recovery, plot_adjusted_pr, plot_confusion)
except ImportError:
    from analysis import (BASE, load_sites, evaluate_sites, calibration,
                          prevalence_analysis, proteome_recovery, save_tables)
    from plot_figures import (plot_benchmark, plot_calibration, plot_prevalence,
                             plot_recovery, plot_adjusted_pr, plot_confusion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=str(BASE / "data"))
    parser.add_argument("--output", type=str, default=str(BASE / "results"))
    parser.add_argument("--figures", type=str, default=str(BASE / "figures"))
    parser.add_argument("--predictions", type=str, default=None)
    args = parser.parse_args()
    from pathlib import Path
    data = Path(args.data)
    import matplotlib.pyplot as plt
    sites = load_sites(data, args.predictions)
    metrics, roc, pr = evaluate_sites(sites)
    bins, diagnostic = calibration(sites)
    estimates, rates, adjusted = prevalence_analysis(sites)
    recovery, curves, ranks = proteome_recovery(pd.read_csv(data / "proteins.csv"))
    save_tables(args.output, model_metrics=metrics, calibration_summary=diagnostic,
                precision_fdr=estimates, recovery_at_K=recovery)
    for cohort in ["test77", "full_test529"]:
        plt.close(plot_benchmark(metrics, roc, pr, cohort=cohort, output=args.figures))
    plt.close(plot_calibration(bins, diagnostic, args.figures))
    plt.close(plot_prevalence(estimates, args.figures))
    plt.close(plot_adjusted_pr(adjusted, args.figures))
    plt.close(plot_confusion(metrics, args.figures))
    plt.close(plot_recovery(recovery, curves, args.figures))
    print(metrics[["cohort", "model", "AUROC", "MCC", "macro_F1"]].to_string(index=False))
    print(recovery[["model", "top_K", "tie_averaged_known_hits"]].to_string(index=False))


if __name__ == "__main__":
    main()

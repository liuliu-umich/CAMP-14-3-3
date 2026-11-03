"""Plot the frozen independent-site and proteome analyses."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analysis import BASE, MODELS, COLORS, PRIORS
except ImportError:
    from analysis import BASE, MODELS, COLORS, PRIORS

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})


def save_figure(fig, name, output=BASE / "figures"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    return fig


def plot_benchmark(metrics, roc, pr, cohort="test77", output=BASE / "figures"):
    report = metrics.loc[metrics.cohort == cohort].set_index("model")
    first = report.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot([0, 1], [0, 1], "--", color=".65", lw=1)
    axes[1].axhline(first.positive / first.n, ls="--", color=".65", lw=1)
    for name in MODELS:
        row = report.loc[name]
        r = roc.loc[(roc.cohort == cohort) & (roc.model == name)]
        p = pr.loc[(pr.cohort == cohort) & (pr.model == name)]
        axes[0].plot(r.FPR, r.TPR, color=COLORS[name], lw=1.8,
                     label=f"{name}: {row.AUROC:.3f}")
        axes[1].step(p.recall, p.precision, where="post", color=COLORS[name], lw=1.8,
                     label=f"{name}: {row.average_precision:.3f}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="A  ROC (AUROC)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="B  Precision–recall (AP)")
    for ax in axes:
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        ax.grid(alpha=.16)
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=8)
    fig.suptitle(f"Identical {int(first.n)} sites: {int(first.positive)} positive, {int(first.negative)} negative")
    fig.tight_layout()
    return save_figure(fig, "Figure5_benchmark77" if cohort == "test77" else "FigureS_benchmark529", output)


def plot_calibration(bins, summary, output=BASE / "figures"):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 6), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot([0, 1], [0, 1], "--", color=".6", label="Perfect calibration")
    axes[0].axhline(summary.iloc[0].positive_fraction, ls=":", color=".65", label="Cohort positive fraction")
    axes[0].plot(bins.mean_score, bins.observed_positive_fraction, "o-", color=COLORS["CAMP"], label="CAMP")
    for row in bins.itertuples(index=False):
        if row.n:
            axes[0].annotate(f"n={row.n}", (row.mean_score, row.observed_positive_fraction),
                             xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set(xlim=(-.03, 1.03), ylim=(-.03, 1.12), xlabel="Mean original score in bin",
                ylabel="Observed positive fraction", title="CAMP calibration: 77 held-out sites")
    axes[0].legend(fontsize=8, loc="lower right")
    centers = (bins.lower + bins.upper) / 2
    axes[1].bar(centers, bins.negative, width=.17, color="#B95757", label="Negative")
    axes[1].bar(centers, bins.positive, bottom=bins.negative, width=.17, color="#83A7CC", label="Positive")
    axes[1].set(xlim=(-.03, 1.03), xlabel="Original score", ylabel="Site count")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.text(.5, .01, "Five fixed bins; no refitting. Calibration in this selected cohort is not proteome calibration.",
             ha="center", fontsize=8)
    return save_figure(fig, "Figure6A_calibration77", output)


def plot_prevalence(estimates, output=BASE / "figures"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name in MODELS:
        data = estimates.loc[estimates.model == name].sort_values("assumed_positive_fraction")
        axes[0].plot(100 * data.assumed_positive_fraction, 100 * data.precision_estimate,
                     "o-", color=COLORS[name], label=name)
        axes[1].plot(100 * data.assumed_positive_fraction, 100 * data.FDR_estimate,
                     "o-", color=COLORS[name], label=name)
    x = np.asarray(PRIORS) * 100
    axes[0].plot(x, x, "--", color=".6", label="Random calls")
    axes[1].plot(x, 100 - x, "--", color=".6", label="Random calls")
    for ax in axes:
        ax.set(xscale="log", xlabel="Assumed binding-site prevalence (%)")
        ax.set_xticks(x, ["0.1", "0.5", "1", "5"])
        ax.grid(alpha=.16)
        ax.legend(fontsize=8)
    axes[0].set(ylabel="Estimated precision (%)")
    axes[1].set(ylabel="Estimated FDR (%)")
    fig.suptitle("Prevalence sensitivity at the fixed score > 0.5 rule")
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.text(.5, .015, "Hypothetical extrapolation from 66 positive and 11 negative sites; not measured proteome precision/FDR.\n"
             "Site-Finder and 14-3-3-Pred use an exploratory common cutoff; bootstrap intervals are in the CSV.",
             ha="center", fontsize=8)
    return save_figure(fig, "Figure6B_precision_FDR", output)


def plot_recovery(reports, curves, output=BASE / "figures"):
    n, known = int(reports.iloc[0].ranked_proteins), int(reports.iloc[0].known_covered)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name in ["CAMP", "BindSite"]:
        data = curves.loc[curves.model == name]
        for ax in axes:
            ax.plot(data.top_K, data.known_hits, color=COLORS[name], lw=1.8, label=name)
        cuts = reports.loc[reports.model == name]
        axes[1].scatter(cuts.top_K, cuts.tie_averaged_known_hits, color=COLORS[name], s=24)
    for ax, limit in zip(axes, [n, 1000]):
        ax.plot([0, limit], [0, known * limit / n], "--", color=".6", label="Uniform random")
        ax.set(xlim=(0, limit), ylim=(0, known * 1.03), xlabel="Top-ranked proteins inspected",
               ylabel=f"Known binding proteins recovered (out of {known})")
        ax.grid(alpha=.16)
    axes[0].set_title("C  Entire ranked proteome")
    axes[1].set_title("D  First 1,000 candidates")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Identical {n:,} proteins; maximum S/T-site score per protein")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.text(.5, .01, "Unknown background is unlabeled. Known-client recovery does not measure precision or FDR.",
             ha="center", fontsize=8)
    return save_figure(fig, "Figure6CD_proteome_recovery", output)


def plot_adjusted_pr(curves, output=BASE / "figures"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, prior in zip(axes.ravel(), PRIORS):
        for name in MODELS:
            data = curves.loc[(curves.model == name) & (curves.assumed_positive_fraction == prior)]
            ax.step(data.recall, data.precision_estimate, where="post", color=COLORS[name], lw=1.5, label=name)
        ax.axhline(prior, ls="--", color=".6", lw=1)
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="Recall", ylabel="Estimated precision",
               title=f"Assumed prevalence: {100 * prior:g}%")
        ax.grid(alpha=.16)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Hypothetical precision–recall curves")
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.text(.5, .01, "Fixed class-conditional rates from 77 sites; zero observed false positives do not guarantee zero FDR.",
             ha="center", fontsize=8)
    return save_figure(fig, "FigureS_adjusted_PR", output)


def plot_confusion(metrics, output=BASE / "figures"):
    report = metrics.loc[metrics.cohort == "test77"].set_index("model")
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for ax, name in zip(axes.ravel(), MODELS):
        row = report.loc[name]
        counts = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
        # Keep the matrix cells editable in vector exports.
        edges = np.arange(3) - .5
        ax.pcolormesh(edges, edges, counts, cmap="Blues", vmin=0, vmax=66,
                      shading="flat")
        ax.set(xlim=(-.5, 1.5), ylim=(1.5, -.5), aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                        color="white" if counts[i, j] > 33 else "black", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
               yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Observed", title=name)
    fig.suptitle("77-site confusion matrices: fixed score > 0.5 rule")
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.5, .01, "The common cutoff is exploratory for Site-Finder and 14-3-3-Pred.", ha="center", fontsize=8)
    return save_figure(fig, "FigureS_confusion77", output)

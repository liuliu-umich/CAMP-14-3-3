"""Evaluate frozen scores without fitting models or selecting test thresholds."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

BASE = Path(__file__).resolve().parent
MODELS = {"CAMP": "score_camp", "BindSite": "score_bindsite_ensemble",
          "Site-Finder": "score_sitefinder", "14-3-3-Pred": "score_1433pred_consensus"}
COLORS = {"CAMP": "#2474AB", "BindSite": "#DE7831",
          "Site-Finder": "#39846A", "14-3-3-Pred": "#9467BD"}
PRIORS = [.001, .005, .01, .05]
SEED = 20261026


def load_sites(data=BASE / "data", predictions=None):
    data = Path(data)
    sites = pd.read_csv(data / "sites.csv")
    scores = pd.read_csv(predictions if predictions is not None else data / "predictions.csv")
    keys = ["sample_id", "uniprot_id", "site_1based", "label"]
    if sites.sample_id.duplicated().any() or scores.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs.")
    joined = sites.merge(scores, on=keys, how="outer", validate="one_to_one", indicator=True, sort=False)
    if not joined._merge.eq("both").all():
        raise ValueError("Scores and site identities do not match.")
    joined = joined.drop(columns="_merge")
    if len(joined) != 529 or joined.label.sum() != 265:
        raise ValueError("The frozen full-test cohort must contain 265 positive and 264 negative sites.")
    primary = joined.loc[joined.in_test77 == 1]
    if len(primary) != 77 or primary.label.sum() != 66:
        raise ValueError("The frozen primary cohort must contain 66 positive and 11 negative sites.")
    if not primary.provenance.eq("curated_reference_match").all():
        raise ValueError("Primary sites must match the reference curation.")
    overlap = ["CAMP_train_protein_overlap", "BindSite_development_exact_peptide_overlap",
               "SiteFinder_train_protein_overlap"]
    if primary[overlap].to_numpy().any():
        raise ValueError("A primary site has a recorded development overlap.")
    if not np.isfinite(joined[list(MODELS.values())].to_numpy()).all():
        raise ValueError("Missing/nonfinite scores; never replace missing scores with zero.")
    return joined


def site_metrics(y, score):
    called = np.asarray(score) > .5
    tn, fp, fn, tp = confusion_matrix(y, called, labels=[0, 1]).ravel()
    return dict(n=len(y), positive=int(np.sum(y)), negative=int(len(y) - np.sum(y)),
                AUROC=roc_auc_score(y, score), average_precision=average_precision_score(y, score),
                MCC=matthews_corrcoef(y, called), accuracy=accuracy_score(y, called),
                precision=precision_score(y, called, zero_division=0),
                recall=recall_score(y, called, zero_division=0),
                F1=f1_score(y, called, zero_division=0),
                macro_precision=precision_score(y, called, average="macro", zero_division=0),
                macro_recall=recall_score(y, called, average="macro", zero_division=0),
                macro_F1=f1_score(y, called, average="macro", zero_division=0),
                threshold=.5, threshold_rule=">0.5", TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def evaluate_sites(sites):
    rows, roc_rows, pr_rows = [], [], []
    for cohort, selected in [("test77", sites.loc[sites.in_test77 == 1]), ("full_test529", sites)]:
        y = selected.label.to_numpy(int)
        for name, column in MODELS.items():
            score = selected[column].to_numpy(float)
            rows.append(dict(cohort=cohort, model=name, **site_metrics(y, score)))
            fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
            roc_rows.extend(dict(cohort=cohort, model=name, FPR=x, TPR=z, threshold=t)
                            for x, z, t in zip(fpr, tpr, thresholds))
            precision, recall, thresholds = precision_recall_curve(y, score)
            pr_rows.extend(dict(cohort=cohort, model=name, precision=p, recall=r, threshold=t)
                           for p, r, t in zip(precision, recall, np.r_[thresholds, np.nan]))
    return pd.DataFrame(rows), pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def calibration(sites):
    selected = sites.loc[sites.in_test77 == 1]
    y, score = selected.label.to_numpy(int), selected.score_camp.to_numpy(float)
    edges = np.linspace(0, 1, 6)
    assigned = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, 4)
    rows = []
    for i in range(5):
        mask = assigned == i
        n, positive = int(mask.sum()), int(y[mask].sum())
        rows.append(dict(bin=i + 1, lower=edges[i], upper=edges[i + 1], n=n,
                         positive=positive, negative=n - positive,
                         mean_score=score[mask].mean() if n else np.nan,
                         observed_positive_fraction=positive / n if n else np.nan))
    bins = pd.DataFrame(rows)
    ece = (bins.n * (bins.mean_score - bins.observed_positive_fraction).abs()).sum() / len(y)
    summary = pd.DataFrame([dict(model="CAMP", n=len(y), bins=5,
                                Brier=np.mean((score - y) ** 2), ECE=ece,
                                positive_fraction=y.mean())])
    return bins, summary


def positive_predictive_value(tpr, fpr, prior):
    numerator = prior * np.asarray(tpr)
    denominator = numerator + (1 - prior) * np.asarray(fpr)
    value = np.full(np.shape(denominator), np.nan, dtype=float)
    return np.divide(numerator, denominator, out=value, where=denominator > 0)


def prevalence_analysis(sites, n_boot=5000):
    selected = sites.loc[sites.in_test77 == 1]
    y = selected.label.to_numpy(int)
    proteins, group = np.unique(selected.uniprot_id, return_inverse=True)
    n_groups = len(proteins)
    positive = np.bincount(group, weights=y, minlength=n_groups)
    negative = np.bincount(group, weights=1 - y, minlength=n_groups)
    draws = np.random.default_rng(SEED).multinomial(
        n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    npos, nneg = draws @ positive, draws @ negative
    valid = (npos > 0) & (nneg > 0)
    rows, rates, curves = [], [], []
    for name, column in MODELS.items():
        scores = selected[column].to_numpy(float)
        called = scores > .5
        tp, fp = np.sum(called & (y == 1)), np.sum(called & (y == 0))
        tpr, fpr = tp / y.sum(), fp / (1 - y).sum()
        boot_tp = draws @ np.bincount(group, weights=called & (y == 1), minlength=n_groups)
        boot_fp = draws @ np.bincount(group, weights=called & (y == 0), minlength=n_groups)
        boot_tpr, boot_fpr = boot_tp[valid] / npos[valid], boot_fp[valid] / nneg[valid]
        rates.append(dict(model=name, TP=int(tp), FP=int(fp), positive=int(y.sum()),
                          negative=int((1 - y).sum()), recall=tpr, false_positive_rate=fpr))
        for prior in PRIORS:
            estimate = float(positive_predictive_value(tpr, fpr, prior))
            bootstrap = positive_predictive_value(boot_tpr, boot_fpr, prior)
            finite = bootstrap[np.isfinite(bootstrap)]
            low, high = np.quantile(finite, [.025, .975])
            rows.append(dict(model=name, threshold=.5, threshold_rule=">0.5",
                             assumed_positive_fraction=prior, precision_estimate=estimate,
                             FDR_estimate=1 - estimate, precision_low=low, precision_high=high,
                             FDR_low=1 - high, FDR_high=1 - low,
                             bootstrap_draws=n_boot, valid_bootstrap_draws=len(finite)))
        # Keep all ties together when sweeping raw scores.
        for threshold in np.r_[np.inf, np.unique(scores)[::-1], -np.inf]:
            calls = scores > threshold
            recall = np.sum(calls & (y == 1)) / y.sum()
            error = np.sum(calls & (y == 0)) / (1 - y).sum()
            for prior in PRIORS:
                estimate = float(positive_predictive_value(recall, error, prior))
                curves.append(dict(model=name, threshold=threshold,
                                   assumed_positive_fraction=prior, recall=recall,
                                   precision_estimate=estimate, FDR_estimate=1 - estimate))
    return pd.DataFrame(rows), pd.DataFrame(rates), pd.DataFrame(curves)


def tie_ranks(scores):
    values = np.asarray(scores)
    order = np.argsort(-values, kind="stable")
    low, size = np.empty(len(values), int), np.empty(len(values), int)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        low[order[start:stop]], size[order[start:stop]] = start + 1, stop - start
        start = stop
    return low, size


def proteome_recovery(proteins, n_random=2000):
    if len(proteins) != 19218 or proteins.accession.duplicated().any():
        raise ValueError("Expected 19,218 unique proteins in the frozen comparison.")
    known = np.flatnonzero(proteins.is_known_binder.to_numpy(int))
    if len(known) != 50:
        raise ValueError("Expected 50 mapped held-out known clients.")
    lengths = np.searchsorted([0, 200, 400, 800, 1600, np.inf], proteins.sequence_length, side="right")
    counts = np.searchsorted([0, 25, 50, 100, 200, 400, np.inf], proteins.n_ST, side="right")
    pools = {}
    for i, key in enumerate(zip(lengths, counts)):
        pools.setdefault(key, []).append(i)
    known_set = set(known)
    strata = [(np.asarray(pool), len(set(pool) & known_set)) for _, pool in sorted(pools.items())]
    rng = np.random.default_rng(SEED)
    sampled = np.asarray([np.concatenate([rng.choice(pool, count, replace=False)
                                         for pool, count in strata if count]) for _ in range(n_random)])
    n, m = len(proteins), len(known)
    grid = np.arange(n + 1)
    reports, curves, rankings = [], [], []
    for model, column in [("CAMP", "score_camp"), ("BindSite", "score_bindsite")]:
        scores = proteins[column].to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError("Missing protein scores.")
        low, size = tie_ranks(scores)
        hits = np.clip((grid[:, None] - low[known] + 1) / size[known], 0, 1).sum(axis=1)
        for k in [100, 500, 1000]:
            null = np.clip((k - low[sampled] + 1) / size[sampled], 0, 1).sum(axis=1)
            reports.append(dict(model=model, top_K=k, ranked_proteins=n, known_covered=m,
                                tie_averaged_known_hits=hits[k], recall=hits[k] / m,
                                matched_random_mean=null.mean(), matched_random_low=np.quantile(null, .025),
                                matched_random_high=np.quantile(null, .975),
                                matched_random_p=(1 + np.sum(null >= hits[k])) / (n_random + 1)))
        curves.extend(dict(model=model, top_K=k, known_hits=h) for k, h in zip(grid, hits))
        rankings.extend(dict(model=model, accession=accession, protein_score=score,
                             rank_min=a, rank_max=a + b - 1, is_known_binder=int(label))
                        for accession, score, a, b, label in zip(
                            proteins.accession, scores, low, size, proteins.is_known_binder))
    return pd.DataFrame(reports), pd.DataFrame(curves), pd.DataFrame(rankings)


def save_tables(out, **tables):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / (name + ".csv"), index=False)

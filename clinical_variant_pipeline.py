import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import SeqIO
from pandas.errors import ParserError

CLINVAR_PROTEIN_PATTERN = re.compile(
    r"\(p\.(?P<ref_aa>[A-Za-z]{3})(?P<position>\d+)(?P<alt_aa>[A-Za-z]{3})\)"
)
DEFAULT_CLINICAL_SIGNIFICANCE_FILTER = "pathogenic_or_likely_pathogenic"
DEFAULT_UNIPROT_RESOLUTION_POLICY = "first_matching"
DEFAULT_DEDUPLICATION_POLICY = "protein_change"
SEQUENCE_CACHE_DIRNAME = "sequence_prediction_cache"
VARIANT_OUTPUT_FILENAME = "clinvar_variant_summary_predictions.csv"
SITE_OUTPUT_FILENAME = "clinvar_site_level_predictions.csv"
STATS_OUTPUT_FILENAME = "clinvar_run_stats.json"
VARIANT_OUTPUT_COLUMNS = [
    "GeneID",
    "variation_id",
    "name",
    "clinical_significance",
    "accession",
    "mutation",
    "mutation_position",
    "sequence_length",
    "esm_max_sequence_length",
    "status",
    "wt_site_count",
    "wt_positive_site_count",
    "wt_best_site",
    "wt_best_probability",
    "mut_site_count",
    "mut_positive_site_count",
    "mut_best_site",
    "mut_best_probability",
]

AA3_TO_AA1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
}


def open_maybe_gzip(path: Path, mode: str = "rt"):
    path = Path(path)
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def _apply_clinical_significance_filter(clinvar_df: pd.DataFrame, significance_filter: str) -> pd.DataFrame:
    if "ClinicalSignificance" not in clinvar_df.columns or significance_filter == "all":
        return clinvar_df

    significance_series = clinvar_df["ClinicalSignificance"].astype(str).str.strip()
    primary_significance = significance_series.str.split(";", n=1).str[0].str.strip()

    if significance_filter == "pathogenic_only":
        mask = primary_significance.str.match(r"^Pathogenic(?:,|$)", case=False, na=False)
        return clinvar_df[mask].copy()

    if significance_filter == "pathogenic_or_likely_pathogenic":
        mask = primary_significance.str.match(
            r"^(Pathogenic(?:/Likely pathogenic)?|Likely pathogenic(?:/Pathogenic)?)(?:,|$)",
            case=False,
            na=False,
        )
        return clinvar_df[mask].copy()

    if significance_filter == "pathogenic_likely_pathogenic_or_vus":
        mask = primary_significance.str.match(
            r"^(Pathogenic(?:/Likely pathogenic)?|Likely pathogenic(?:/Pathogenic)?|Uncertain significance)(?:,|$)",
            case=False,
            na=False,
        )
        return clinvar_df[mask].copy()

    raise ValueError(f"Unsupported significance_filter: {significance_filter}")


def _deduplicate_clinvar_variants(clinvar_df: pd.DataFrame, deduplicate_by: str) -> pd.DataFrame:
    if deduplicate_by == "none":
        return clinvar_df

    if deduplicate_by == "variation_id":
        if "VariationID" not in clinvar_df.columns:
            return clinvar_df
        return clinvar_df.drop_duplicates(subset=["VariationID"]).copy()

    if deduplicate_by == "protein_change":
        dedupe_columns = ["GeneID", "ref_aa", "position", "alt_aa"]
        available_columns = [column for column in dedupe_columns if column in clinvar_df.columns]
        if len(available_columns) != len(dedupe_columns):
            return clinvar_df
        return clinvar_df.drop_duplicates(subset=dedupe_columns).copy()

    raise ValueError(f"Unsupported deduplicate_by: {deduplicate_by}")


def _normalize_optional_str_list(values) -> Optional[list]:
    if values is None:
        return None
    if isinstance(values, (str, int)):
        values = [values]

    normalized = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized or None


def _filter_requested_variants(
    clinvar_df: pd.DataFrame,
    selected_variation_ids=None,
    selected_gene_ids=None,
) -> pd.DataFrame:
    selected_variation_ids = _normalize_optional_str_list(selected_variation_ids)
    selected_gene_ids = _normalize_optional_str_list(selected_gene_ids)

    filtered_df = clinvar_df
    if selected_variation_ids is not None and "VariationID" in filtered_df.columns:
        variation_id_set = set(selected_variation_ids)
        filtered_df = filtered_df[
            filtered_df["VariationID"].astype(str).str.strip().isin(variation_id_set)
        ].copy()

    if selected_gene_ids is not None and "GeneID" in filtered_df.columns:
        gene_id_set = set(selected_gene_ids)
        filtered_df = filtered_df[
            filtered_df["GeneID"].astype(str).str.strip().isin(gene_id_set)
        ].copy()

    return filtered_df


def load_clinvar_variants(
    variant_summary_path: Path,
    significance_filter: str = DEFAULT_CLINICAL_SIGNIFICANCE_FILTER,
    require_germline: bool = True,
    require_grch: bool = True,
    deduplicate_by: str = DEFAULT_DEDUPLICATION_POLICY,
    selected_variation_ids=None,
    selected_gene_ids=None,
) -> pd.DataFrame:
    with open_maybe_gzip(variant_summary_path, "rt") as handle:
        clinvar_df = pd.read_csv(handle, sep="\t", low_memory=False)

    clinvar_df = clinvar_df[clinvar_df["Type"] == "single nucleotide variant"].copy()
    clinvar_df = clinvar_df[clinvar_df["Name"].astype(str).str.contains(CLINVAR_PROTEIN_PATTERN, regex=True)].copy()

    extracted = clinvar_df["Name"].astype(str).str.extract(CLINVAR_PROTEIN_PATTERN)
    clinvar_df["ref_aa3"] = extracted["ref_aa"]
    clinvar_df["position"] = extracted["position"].astype(int)
    clinvar_df["alt_aa3"] = extracted["alt_aa"]
    clinvar_df["ref_aa"] = clinvar_df["ref_aa3"].map(AA3_TO_AA1)
    clinvar_df["alt_aa"] = clinvar_df["alt_aa3"].map(AA3_TO_AA1)

    clinvar_df = clinvar_df[
        clinvar_df["ref_aa"].notna()
        & clinvar_df["alt_aa"].notna()
        & (clinvar_df["alt_aa"] != "*")
    ].copy()

    if "GeneID" in clinvar_df.columns:
        clinvar_df["GeneID"] = clinvar_df["GeneID"].astype(str).str.strip()

    if require_grch and "Assembly" in clinvar_df.columns:
        clinvar_df = clinvar_df[clinvar_df["Assembly"].astype(str).str.contains("GRCh", na=False)].copy()

    if require_germline and "OriginSimple" in clinvar_df.columns:
        clinvar_df = clinvar_df[clinvar_df["OriginSimple"].astype(str).str.contains("germline", case=False, na=False)].copy()

    clinvar_df = _apply_clinical_significance_filter(clinvar_df, significance_filter)
    clinvar_df = _filter_requested_variants(
        clinvar_df,
        selected_variation_ids=selected_variation_ids,
        selected_gene_ids=selected_gene_ids,
    )
    clinvar_df = _deduplicate_clinvar_variants(clinvar_df, deduplicate_by)
    return clinvar_df.reset_index(drop=True)


def load_geneid_to_uniprot_map(idmapping_path: Path) -> dict:
    idmapping_df = pd.read_excel(idmapping_path)
    reviewed_human_df = idmapping_df[
        (idmapping_df["Reviewed"] == "reviewed")
        & (idmapping_df["Organism"] == "Homo sapiens (Human)")
    ].copy()
    reviewed_human_df["From"] = reviewed_human_df["From"].astype(str).str.strip()
    grouped = reviewed_human_df.groupby("From")["Entry"].agg(list).reset_index()
    return dict(zip(grouped["From"], grouped["Entry"]))


def load_uniprot_sequences(fasta_path: Path) -> dict:
    sequence_by_accession = {}
    with open_maybe_gzip(fasta_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            parts = record.id.split("|")
            accession = parts[1] if len(parts) >= 2 else record.id
            sequence_by_accession[accession] = str(record.seq).strip().upper()
    return sequence_by_accession


def apply_missense_variant(sequence: str, position: int, ref_aa: str, alt_aa: str) -> Optional[str]:
    if position < 1 or position > len(sequence):
        return None
    if sequence[position - 1] != ref_aa:
        return None
    return sequence[: position - 1] + alt_aa + sequence[position:]


def find_matching_uniprot_sequence(
    uniprot_ids,
    sequence_by_accession,
    position,
    ref_aa,
    alt_aa,
    resolution_policy: str = DEFAULT_UNIPROT_RESOLUTION_POLICY,
):
    if not isinstance(uniprot_ids, list):
        return None

    matching_entries = []
    for accession in uniprot_ids:
        wt_sequence = sequence_by_accession.get(accession)
        if not wt_sequence:
            continue
        mut_sequence = apply_missense_variant(wt_sequence, position, ref_aa, alt_aa)
        if mut_sequence is not None:
            matching_entries.append((accession, wt_sequence, mut_sequence))

    if not matching_entries:
        return None
    if resolution_policy == "first_matching":
        return matching_entries[0]
    if resolution_policy == "unique_only":
        return matching_entries[0] if len(matching_entries) == 1 else None
    if resolution_policy == "all_matching":
        return matching_entries
    raise ValueError(f"Unsupported uniprot_resolution_policy: {resolution_policy}")


def summarize_site_predictions(results_df: pd.DataFrame, prefix: str) -> dict:
    summary = {
        f"{prefix}_site_count": int(len(results_df)),
        f"{prefix}_positive_site_count": int(results_df["predicted_positive"].sum()),
    }
    if results_df.empty:
        summary[f"{prefix}_best_site"] = None
        summary[f"{prefix}_best_probability"] = None
        return summary

    best_row = results_df.sort_values(["positive_probability", "site"], ascending=[False, True]).iloc[0]
    summary[f"{prefix}_best_site"] = int(best_row["site"])
    summary[f"{prefix}_best_probability"] = float(best_row["positive_probability"])
    return summary


def build_site_comparison_df(
    accession: str,
    gene_id: str,
    position: int,
    ref_aa: str,
    alt_aa: str,
    wt_results_df: pd.DataFrame,
    mut_results_df: pd.DataFrame,
) -> pd.DataFrame:
    wt_df = wt_results_df[["site", "residue", "prediction", "predicted_positive", "positive_probability"]].copy()
    wt_df = wt_df.rename(
        columns={
            "residue": "wt_residue",
            "prediction": "wt_prediction",
            "predicted_positive": "wt_predicted_positive",
            "positive_probability": "wt_positive_probability",
        }
    )
    mut_df = mut_results_df[["site", "residue", "prediction", "predicted_positive", "positive_probability"]].copy()
    mut_df = mut_df.rename(
        columns={
            "residue": "mut_residue",
            "prediction": "mut_prediction",
            "predicted_positive": "mut_predicted_positive",
            "positive_probability": "mut_positive_probability",
        }
    )

    merged = wt_df.merge(mut_df, on="site", how="outer").sort_values("site").reset_index(drop=True)
    merged["accession"] = accession
    merged["gene_id"] = gene_id
    merged["mutation_position"] = int(position)
    merged["mutation"] = f"{ref_aa}{position}{alt_aa}"
    merged["delta_positive_probability"] = (
        merged["mut_positive_probability"].fillna(0.0) - merged["wt_positive_probability"].fillna(0.0)
    )
    merged["site_distance_to_mutation"] = (merged["site"] - int(position)).abs()
    return merged


def _get_predict_all_sites():
    # Delay the heavy model stack until the final full-sequence prediction step.
    from features import predict_all_sites

    return predict_all_sites


def _configure_feature_runtime(phospholingo_model_loc: Optional[Path] = None):
    if phospholingo_model_loc is None:
        return

    from features import set_phospholingo_model_loc

    set_phospholingo_model_loc(str(Path(phospholingo_model_loc).expanduser()))


def _get_default_esm_max_sequence_length() -> int:
    from features import ESM_MAX_SEQUENCE_LENGTH

    return int(ESM_MAX_SEQUENCE_LENGTH)


def _sequence_cache_path(cache_dir: Path, sequence: str) -> Path:
    sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    return cache_dir / f"{sequence_hash}.csv"


def _predict_all_sites_with_cache(
    sequence: str,
    predict_all_sites,
    prediction_cache: dict,
    cache_dir: Optional[Path] = None,
    stats: Optional[dict] = None,
) -> pd.DataFrame:
    cached_results = prediction_cache.get(sequence)
    if cached_results is not None:
        if stats is not None:
            stats["memory_cache_hits"] += 1
        return cached_results.copy()

    if cache_dir is not None:
        cache_path = _sequence_cache_path(cache_dir, sequence)
        if cache_path.exists():
            disk_cached_results = pd.read_csv(cache_path)
            prediction_cache[sequence] = disk_cached_results.copy()
            if stats is not None:
                stats["disk_cache_hits"] += 1
            return disk_cached_results.copy()

    results_df, _, _ = predict_all_sites(sequence)
    prediction_cache[sequence] = results_df.copy()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(_sequence_cache_path(cache_dir, sequence), index=False)
    if stats is not None:
        stats["cache_misses"] += 1
    return results_df.copy()


def _make_variant_task_key(variant_row: dict) -> str:
    variation_id = variant_row.get("variation_id")
    if pd.notna(variation_id):
        return f"{variation_id}|{variant_row.get('accession', '')}|{variant_row.get('mutation', '')}"
    return "|".join(
        [
            str(variant_row.get("GeneID", "")),
            str(variant_row.get("accession", "")),
            str(variant_row.get("mutation", "")),
            str(variant_row.get("name", "")),
        ]
    )


def _append_frame_to_csv(frame: pd.DataFrame, output_path: Path):
    if frame.empty:
        return
    header = not output_path.exists() or output_path.stat().st_size == 0
    frame.to_csv(output_path, mode="a", header=header, index=False)


def _normalize_variant_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in VARIANT_OUTPUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized[VARIANT_OUTPUT_COLUMNS].copy()


def _load_completed_task_keys(variant_output: Path) -> set:
    if not variant_output.exists() or variant_output.stat().st_size == 0:
        return set()

    completed_df = pd.read_csv(variant_output)
    if completed_df.empty:
        return set()

    keys = set()
    for _, row in completed_df.iterrows():
        keys.add(
            _make_variant_task_key(
                {
                    "variation_id": row.get("variation_id"),
                    "GeneID": row.get("GeneID"),
                    "accession": row.get("accession"),
                    "mutation": row.get("mutation"),
                    "name": row.get("name"),
                }
            )
        )
    return keys


def _resolve_variant_tasks(
    clinvar_df: pd.DataFrame,
    geneid_to_uniprot: dict,
    sequence_by_accession: dict,
    uniprot_resolution_policy: str,
    max_sequence_length: Optional[int],
):
    clinvar_df = clinvar_df.copy()
    clinvar_df["uniprot_ids"] = clinvar_df["GeneID"].map(geneid_to_uniprot)

    unresolved_rows = []
    resolved_tasks = []

    for _, row in clinvar_df.iterrows():
        gene_id = row["GeneID"]
        position = int(row["position"])
        ref_aa = row["ref_aa"]
        alt_aa = row["alt_aa"]
        mutation = f"{ref_aa}{position}{alt_aa}"

        match = find_matching_uniprot_sequence(
            row["uniprot_ids"],
            sequence_by_accession,
            position,
            ref_aa,
            alt_aa,
            resolution_policy=uniprot_resolution_policy,
        )

        if match is None:
            unresolved_rows.append(
                {
                    "GeneID": gene_id,
                    "variation_id": row.get("VariationID"),
                    "name": row.get("Name"),
                    "mutation": mutation,
                    "status": "sequence_not_resolved",
                }
            )
            continue

        matches = match if uniprot_resolution_policy == "all_matching" else [match]
        for accession, wt_sequence, mut_sequence in matches:
            if max_sequence_length is not None and len(wt_sequence) > int(max_sequence_length):
                unresolved_rows.append(
                    {
                        "GeneID": gene_id,
                        "variation_id": row.get("VariationID"),
                        "name": row.get("Name"),
                        "mutation": mutation,
                        "accession": accession,
                        "sequence_length": int(len(wt_sequence)),
                        "esm_max_sequence_length": int(max_sequence_length),
                        "status": "sequence_too_long_for_esm",
                    }
                )
                continue
            resolved_tasks.append(
                {
                    "GeneID": gene_id,
                    "variation_id": row.get("VariationID"),
                    "name": row.get("Name"),
                    "clinical_significance": row.get("ClinicalSignificance"),
                    "accession": accession,
                    "mutation": mutation,
                    "mutation_position": position,
                    "ref_aa": ref_aa,
                    "alt_aa": alt_aa,
                    "wt_sequence": wt_sequence,
                    "mut_sequence": mut_sequence,
                }
            )

    return resolved_tasks, unresolved_rows


def _write_run_stats(output_path: Path, stats: dict):
    output_path.write_text(json.dumps(stats, indent=2, sort_keys=True))


def run_variant_prediction_pipeline(
    variant_summary_path: Path,
    idmapping_path: Path,
    fasta_path: Path,
    output_dir: Path,
    significance_filter: str = DEFAULT_CLINICAL_SIGNIFICANCE_FILTER,
    require_germline: bool = True,
    require_grch: bool = True,
    deduplicate_by: str = DEFAULT_DEDUPLICATION_POLICY,
    selected_variation_ids=None,
    selected_gene_ids=None,
    uniprot_resolution_policy: str = DEFAULT_UNIPROT_RESOLUTION_POLICY,
    phospholingo_model_loc: Optional[Path] = None,
    esm_max_sequence_length: Optional[int] = None,
    max_variants: Optional[int] = None,
    resume: bool = True,
):
    variant_summary_path = Path(variant_summary_path)
    idmapping_path = Path(idmapping_path)
    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_cache_dir = output_dir / SEQUENCE_CACHE_DIRNAME
    variant_output = output_dir / VARIANT_OUTPUT_FILENAME
    site_output = output_dir / SITE_OUTPUT_FILENAME
    stats_output = output_dir / STATS_OUTPUT_FILENAME

    if not resume:
        for output_path in [variant_output, site_output, stats_output]:
            if output_path.exists():
                output_path.unlink()

    clinvar_df = load_clinvar_variants(
        variant_summary_path,
        significance_filter=significance_filter,
        require_germline=require_germline,
        require_grch=require_grch,
        deduplicate_by=deduplicate_by,
        selected_variation_ids=selected_variation_ids,
        selected_gene_ids=selected_gene_ids,
    )
    if max_variants is not None:
        clinvar_df = clinvar_df.head(int(max_variants)).copy()

    geneid_to_uniprot = load_geneid_to_uniprot_map(idmapping_path)
    sequence_by_accession = load_uniprot_sequences(fasta_path)
    _configure_feature_runtime(phospholingo_model_loc=phospholingo_model_loc)
    if esm_max_sequence_length is None:
        esm_max_sequence_length = _get_default_esm_max_sequence_length()
    predict_all_sites = _get_predict_all_sites()
    prediction_cache = {}
    stats = {
        "input_variant_rows": int(len(clinvar_df)),
        "resolved_task_count": 0,
        "unresolved_variant_count": 0,
        "unique_wt_sequences": 0,
        "unique_mut_sequences": 0,
        "unique_total_sequences": 0,
        "memory_cache_hits": 0,
        "disk_cache_hits": 0,
        "cache_misses": 0,
        "variant_rows_written": 0,
        "site_rows_written": 0,
        "skipped_completed_tasks": 0,
        "skipped_too_long_for_esm": 0,
        "resume_enabled": bool(resume),
        "esm_max_sequence_length": int(esm_max_sequence_length) if esm_max_sequence_length is not None else None,
    }

    resolved_tasks, unresolved_rows = _resolve_variant_tasks(
        clinvar_df,
        geneid_to_uniprot,
        sequence_by_accession,
        uniprot_resolution_policy,
        esm_max_sequence_length,
    )
    stats["resolved_task_count"] = int(len(resolved_tasks))
    stats["unresolved_variant_count"] = int(len(unresolved_rows))
    stats["skipped_too_long_for_esm"] = int(
        sum(1 for row in unresolved_rows if row.get("status") == "sequence_too_long_for_esm")
    )

    completed_task_keys = _load_completed_task_keys(variant_output) if resume else set()
    if unresolved_rows:
        unresolved_df = _normalize_variant_output_frame(pd.DataFrame(unresolved_rows))
        if completed_task_keys:
            unresolved_df["_task_key"] = unresolved_df.apply(lambda row: _make_variant_task_key(row.to_dict()), axis=1)
            unresolved_df = unresolved_df[~unresolved_df["_task_key"].isin(completed_task_keys)].drop(columns=["_task_key"])
        _append_frame_to_csv(unresolved_df, variant_output)
        stats["variant_rows_written"] += int(len(unresolved_df))

    pending_tasks = []
    for task in resolved_tasks:
        task_key = _make_variant_task_key(task)
        if task_key in completed_task_keys:
            stats["skipped_completed_tasks"] += 1
            continue
        pending_tasks.append(task)

    unique_wt_sequences = {task["wt_sequence"] for task in pending_tasks}
    unique_mut_sequences = {task["mut_sequence"] for task in pending_tasks}
    unique_sequences = unique_wt_sequences | unique_mut_sequences
    stats["unique_wt_sequences"] = int(len(unique_wt_sequences))
    stats["unique_mut_sequences"] = int(len(unique_mut_sequences))
    stats["unique_total_sequences"] = int(len(unique_sequences))

    for sequence in unique_sequences:
        _predict_all_sites_with_cache(
            sequence,
            predict_all_sites,
            prediction_cache,
            cache_dir=sequence_cache_dir,
            stats=stats,
        )

    for task in pending_tasks:
        wt_results_df = _predict_all_sites_with_cache(
            task["wt_sequence"],
            predict_all_sites,
            prediction_cache,
            cache_dir=sequence_cache_dir,
            stats=stats,
        )
        mut_results_df = _predict_all_sites_with_cache(
            task["mut_sequence"],
            predict_all_sites,
            prediction_cache,
            cache_dir=sequence_cache_dir,
            stats=stats,
        )

        variant_summary = {
            "GeneID": task["GeneID"],
            "variation_id": task["variation_id"],
            "name": task["name"],
            "clinical_significance": task["clinical_significance"],
            "accession": task["accession"],
            "mutation": task["mutation"],
            "mutation_position": task["mutation_position"],
            "status": "ok",
        }
        variant_summary.update(summarize_site_predictions(wt_results_df, "wt"))
        variant_summary.update(summarize_site_predictions(mut_results_df, "mut"))

        comparison_df = build_site_comparison_df(
            accession=task["accession"],
            gene_id=task["GeneID"],
            position=task["mutation_position"],
            ref_aa=task["ref_aa"],
            alt_aa=task["alt_aa"],
            wt_results_df=wt_results_df,
            mut_results_df=mut_results_df,
        )
        _append_frame_to_csv(comparison_df, site_output)
        stats["site_rows_written"] += int(len(comparison_df))

        _append_frame_to_csv(_normalize_variant_output_frame(pd.DataFrame([variant_summary])), variant_output)
        stats["variant_rows_written"] += 1

        if stats["variant_rows_written"] % 100 == 0:
            _write_run_stats(stats_output, stats)

    _write_run_stats(stats_output, stats)

    try:
        variant_summary_df = (
            pd.read_csv(variant_output) if variant_output.exists() and variant_output.stat().st_size else pd.DataFrame()
        )
    except ParserError as exc:
        raise RuntimeError(
            f"Failed to read {variant_output}. The file may have been created by an older mixed-schema run. "
            "Please delete the output directory or rerun with resume=False / a fresh output_dir."
        ) from exc

    try:
        site_comparison_df = pd.read_csv(site_output) if site_output.exists() and site_output.stat().st_size else pd.DataFrame()
    except ParserError as exc:
        raise RuntimeError(
            f"Failed to read {site_output}. The file may be corrupted from an interrupted older run. "
            "Please delete the output directory or rerun with resume=False / a fresh output_dir."
        ) from exc
    return variant_summary_df, site_comparison_df, variant_output, site_output


def main():
    parser = argparse.ArgumentParser(
        description="Run ClinVar missense variants through wild-type and mutant 14-3-3 site scanning."
    )
    parser.add_argument("--variant-summary", type=Path, required=True, help="Path to ClinVar variant_summary.txt or .gz")
    parser.add_argument("--idmapping", type=Path, required=True, help="Path to UniProt ID mapping Excel file")
    parser.add_argument("--fasta", type=Path, required=True, help="Path to UniProt FASTA file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output CSV files")
    parser.add_argument(
        "--phospholingo-model",
        type=Path,
        default=None,
        help="Path to the PhosphoLingo checkpoint (.ckpt). Overrides PHOSPHOLINGO_MODEL_LOC for this run.",
    )
    parser.add_argument(
        "--esm-max-sequence-length",
        type=int,
        default=None,
        help="Skip sequences longer than this length before entering the ESM-based pipeline. Default comes from features.py.",
    )
    parser.add_argument("--max-variants", type=int, default=None, help="Optionally limit the number of variants")
    parser.add_argument(
        "--variation-id",
        dest="selected_variation_ids",
        action="append",
        default=None,
        help="Specific ClinVar VariationID to include. Repeat this flag to include multiple IDs.",
    )
    parser.add_argument(
        "--gene-id",
        dest="selected_gene_ids",
        action="append",
        default=None,
        help="Specific GeneID to include. Repeat this flag to include multiple IDs.",
    )
    parser.add_argument(
        "--deduplicate-by",
        default=DEFAULT_DEDUPLICATION_POLICY,
        choices=["protein_change", "variation_id", "none"],
        help="How to deduplicate filtered ClinVar variants before prediction.",
    )
    parser.add_argument(
        "--significance-filter",
        default=DEFAULT_CLINICAL_SIGNIFICANCE_FILTER,
        choices=[
            "pathogenic_only",
            "pathogenic_or_likely_pathogenic",
            "pathogenic_likely_pathogenic_or_vus",
            "all",
        ],
        help="Clinical significance filter mode.",
    )
    parser.add_argument(
        "--include-all-origins",
        action="store_true",
        help="If set, do not require OriginSimple to contain germline.",
    )
    parser.add_argument(
        "--include-non-grch",
        action="store_true",
        help="If set, do not require Assembly to contain GRCh.",
    )
    parser.add_argument(
        "--uniprot-resolution-policy",
        default=DEFAULT_UNIPROT_RESOLUTION_POLICY,
        choices=["first_matching", "unique_only", "all_matching"],
        help="How to resolve variants that map to multiple reviewed human UniProt entries.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="If set, overwrite previous outputs instead of resuming from existing variant summary output.",
    )
    args = parser.parse_args()

    variant_summary_df, site_comparison_df, variant_output, site_output = run_variant_prediction_pipeline(
        variant_summary_path=args.variant_summary,
        idmapping_path=args.idmapping,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        significance_filter=args.significance_filter,
        require_germline=not args.include_all_origins,
        require_grch=not args.include_non_grch,
        deduplicate_by=args.deduplicate_by,
        selected_variation_ids=args.selected_variation_ids,
        selected_gene_ids=args.selected_gene_ids,
        uniprot_resolution_policy=args.uniprot_resolution_policy,
        phospholingo_model_loc=args.phospholingo_model,
        esm_max_sequence_length=args.esm_max_sequence_length,
        max_variants=args.max_variants,
        resume=not args.no_resume,
    )

    print(f"Resolved variants: {(variant_summary_df['status'] == 'ok').sum() if not variant_summary_df.empty else 0}")
    print(f"Variant summary saved to: {variant_output}")
    print(f"Site-level output saved to: {site_output}")
    print(f"Variant rows: {len(variant_summary_df)}")
    print(f"Site rows: {len(site_comparison_df)}")


if __name__ == "__main__":
    main()

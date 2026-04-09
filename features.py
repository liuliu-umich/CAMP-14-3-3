import ast
import gc
import importlib
import itertools
import os
import platform
import shutil
import sys
import uuid
import warnings
import __main__
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Metric `AUROC` will save all targets and predictions in buffer.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from Bio import SeqIO  # noqa: F401
from IPython.display import clear_output
from joblib import load
from localcider.sequenceParameters import SequenceParameters
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

import joblib
import src.files.iupred2a.iupred2a_lib as iupred2a_lib
from gensim.models import word2vec

transformers_logging.set_verbosity_error()


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PHOSPHOLINGO_DIR = (BASE_DIR / "src/files/PhosphoLingo/phospholingo").resolve()
DEEPHASE_PROTVEC_MODEL = BASE_DIR / "src/files/DeePhase/__PREDICT/tools/Embeddings/swissprot_size200_window25.model"
MPLCONFIG_DIR = BASE_DIR / "temp" / "mplconfig"
XDG_CACHE_DIR = BASE_DIR / "temp" / "xdg-cache"
PHOSPHOLINGO_MODEL_LOC = os.environ.get(
    "PHOSPHOLINGO_MODEL_LOC",
    "/Users/newuser/PhosphoLingo_ST_new.ckpt",
)
SEQUENCE_BINDING_THRESHOLD = float(os.environ.get("SEQUENCE_BINDING_THRESHOLD", "0.5"))
PHOSPHOLINGO_SITE_CHUNK_SIZE = int(os.environ.get("PHOSPHOLINGO_SITE_CHUNK_SIZE", "16"))
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FEATURE_COLUMNS = [
    "esm_residue_embedding_491", "NCPR", "onehot_-3_V", "nuSVR",
    "esm_residue_embedding_715", "esm_residue_embedding_394", "phospho_score",
    "onehot_1_L", "onehot_-2_G", "onehot_1_R", "onehot_-3_A",
    "esm_residue_embedding_356", "esm_residue_embedding_81",
    "esm_residue_embedding_453", "onehot_-3_E", "iupred_score",
    "onehot_1_P", "onehot_-2_S", "esm_residue_embedding_1171",
    "onehot_-1_K", "esm_residue_embedding_1242", "esm_residue_embedding_53",
    "onehot_7_G", "onehot_-1_H", "onehot_3_D", "deephase_w2v_multi",
    "onehot_-4_R", "onehot_2_P", "esm_residue_embedding_385",
    "esm_residue_embedding_844", "esm_residue_embedding_197",
    "esm_residue_embedding_1215", "esm_residue_embedding_1041",
    "esm_residue_embedding_1074", "onehot_2_K", "esm_protein_embedding_68",
    "esm_residue_embedding_482", "onehot_-5_R", "esm_residue_embedding_546",
    "esm_residue_embedding_327", "onehot_-2_R", "onehot_-3_R",
    "onehot_-3_G", "esm_residue_embedding_1259",
]

MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

sys.path.insert(0, str(PHOSPHOLINGO_DIR))
import predict  # noqa: E402


_ESM_TOKENIZER = None
_ESM_MODEL = None
_ESM_DEVICE = None
_DEEPHASE_MODULE = None
_PREDICT_MODEL = None


def _get_esm_resources():
    global _ESM_TOKENIZER, _ESM_MODEL, _ESM_DEVICE
    if _ESM_TOKENIZER is None or _ESM_MODEL is None or _ESM_DEVICE is None:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

        _ESM_TOKENIZER = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        _ESM_MODEL = AutoModel.from_pretrained(ESM_MODEL_NAME)
        _ESM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ESM_MODEL.to(_ESM_DEVICE)
        _ESM_MODEL.eval()
    return _ESM_TOKENIZER, _ESM_MODEL, _ESM_DEVICE


def _get_predict_model():
    global _PREDICT_MODEL
    if _PREDICT_MODEL is None:
        _PREDICT_MODEL = joblib.load(MODEL_DIR / "1433model_20260223.pkl")
    return _PREDICT_MODEL


def split_ngrams(seq, n):
    a, b, c = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n), zip(*[iter(seq[2:])] * n)
    str_ngrams = []
    for ngrams in [a, b, c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams


class ProtVec(word2vec.Word2Vec):
    def __init__(
        self,
        fasta_fname=None,
        corpus=None,
        n=3,
        size=100,
        corpus_fname="corpus.txt",
        sg=1,
        window=25,
        min_count=1,
        workers=20,
    ):
        self.n = n
        self.size = size
        self.fasta_fname = fasta_fname

        if corpus is None and fasta_fname is None:
            raise Exception("Either fasta_fname or corpus is needed!")

        if fasta_fname is not None:
            generate_corpusfile(fasta_fname, n, corpus_fname)
            corpus = word2vec.Text8Corpus(corpus_fname)

        word2vec.Word2Vec.__init__(
            self,
            corpus,
            size=size,
            sg=sg,
            window=window,
            min_count=min_count,
            workers=workers,
        )

    def to_vecs(self, seq):
        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self.wv[ngram])
                except Exception as exc:
                    raise Exception("Model has never trained this n-gram: " + ngram) from exc
            protvecs.append(sum(ngram_vecs))
        return protvecs

    def get_vector(self, seq):
        return sum(self.to_vecs(seq))


def _get_deephase_module():
    global _DEEPHASE_MODULE
    if _DEEPHASE_MODULE is None:
        setattr(__main__, "ProtVec", ProtVec)
        _DEEPHASE_MODULE = importlib.import_module("src.files.DeePhase.__PREDICT.deephase_utils")
    return _DEEPHASE_MODULE


def calc_seq_prop(seq, residues, Nc, Cc, Hc):
    seq = list(seq).copy()
    fasta_kappa = np.array(seq.copy())
    N = len(seq)
    r = residues.copy()

    fK = sum([seq.count(a) for a in ["K"]]) / N
    fR = sum([seq.count(a) for a in ["R"]]) / N
    fE = sum([seq.count(a) for a in ["E"]]) / N
    fD = sum([seq.count(a) for a in ["D"]]) / N
    faro = sum([seq.count(a) for a in ["W", "Y", "F"]]) / N
    mean_lambda = np.mean(r.loc[seq].lambdas)

    pairs = np.array(list(itertools.combinations(seq, 2)))
    pairs_indices = np.array(list(itertools.combinations(range(N), 2)))
    ij_dist = np.diff(pairs_indices, axis=1).flatten().astype(float)
    ll = r.lambdas.loc[pairs[:, 0]].values + r.lambdas.loc[pairs[:, 1]].values
    beta = -1
    shd = np.sum(ll * np.power(np.abs(ij_dist), beta)) / N

    if Nc == 1:
        r.loc["X"] = r.loc[seq[0]]
        r.loc["X", "q"] = r.loc[seq[0], "q"] + 1.0
        seq[0] = "X"
        if r.loc["X", "q"] > 0:
            fasta_kappa[0] = "K"
        else:
            fasta_kappa[0] = "A"
    if Cc == 1:
        r.loc["Z"] = r.loc[seq[-1]]
        r.loc["Z", "q"] = r.loc[seq[-1], "q"] - 1.0
        seq[-1] = "Z"
        if r.loc["Z", "q"] < 0:
            fasta_kappa[-1] = "D"
        else:
            fasta_kappa[-1] = "A"
    if Hc < 0.5:
        r.loc["H", "q"] = 0
        fasta_kappa[np.where(np.array(seq) == "H")[0]] = "A"
    elif Hc >= 0.5:
        r.loc["H", "q"] = 1
        fasta_kappa[np.where(np.array(seq) == "H")[0]] = "K"

    pairs = np.array(list(itertools.combinations(seq, 2)))
    qq = r.q.loc[pairs[:, 0]].values * r.q.loc[pairs[:, 1]].values
    scd = np.sum(qq * np.sqrt(ij_dist)) / N
    SeqOb = SequenceParameters("".join(fasta_kappa))
    kappa = SeqOb.get_kappa()
    fcr = r.q.loc[seq].abs().mean()
    ncpr = r.q.loc[seq].mean()

    return pd.Series(
        data=[fK, fR, fE, fD, faro, scd, shd, kappa, fcr, mean_lambda, ncpr],
        index=["fK", "fR", "fE", "fD", "faro", "SCD", "SHD", "kappa", "FCR", "mean_lambda", "NCPR"],
    )


def extract_compactness_score(idr_seq):
    aa = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

    model_nu = load(MODEL_DIR / "svr_model_nu.joblib")
    model_spr = load(MODEL_DIR / "svr_model_SPR.joblib")
    features_nu = ["SCD", "SHD", "kappa", "FCR", "mean_lambda"]
    features_spr = ["SCD", "SHD", "mean_lambda"]

    residues = pd.read_csv(MODEL_DIR / "residues.csv")
    residues = residues.set_index("one")

    df = pd.DataFrame(
        columns=[
            "nuSVR", "SconfSVR/N (kB)", "mean_lambda", "SHD", "SCD", "kappa",
            "FCR", "NCPR", "fK", "fR", "fE", "fD", "faro",
        ]
    )

    fasta_dict = {"protein1": idr_seq}
    current_upload = ["protein1"]

    for x in list(current_upload):
        valid = True
        for a in fasta_dict[x]:
            if a not in aa:
                del fasta_dict[x]
                valid = False
                break
        if not valid:
            current_upload.remove(x)

    charged_N_terminal_amine = False
    charged_C_terminal_carboxyl = True
    charged_histidine = False
    Nc = 1 if charged_N_terminal_amine else 0
    Cc = 1 if charged_C_terminal_carboxyl else 0
    Hc = 0 if not charged_histidine else 1

    for k in fasta_dict.keys():
        res = calc_seq_prop(fasta_dict[k], residues, Nc, Cc, Hc)
        nu = np.around(model_nu.predict(res.loc[features_nu].values.reshape(1, -1))[0], 3)
        spr = np.around(model_spr.predict(res.loc[features_spr].values.reshape(1, -1))[0], 3)
        df.loc[k, "nuSVR"] = nu
        df.loc[k, "SconfSVR/N (kB)"] = spr
        df.loc[k, res.index.values] = np.around(res.loc[res.index.values].values, 3)

    clear_output()
    return df


def extract_idr_anchor_score(sequence, site):
    df = build_idr_dataframe(sequence)
    return extract_idr_anchor_score_from_df(df, site)


def build_idr_dataframe(sequence):
    iupred_scores = iupred2a_lib.iupred(sequence, "long")[0]
    anchor_scores = iupred2a_lib.anchor2(sequence)[0]

    df = pd.DataFrame(
        {
            "position": list(range(1, len(sequence) + 1)),
            "amino_acid": list(sequence),
            "iupred_score": iupred_scores,
            "anchor_score": anchor_scores,
        }
    )
    df["position"] = df["position"].astype(int)
    df["call_idr"] = df["iupred_score"].apply(lambda x: 1 if x > 0.5 else 0)
    df["segment_id"] = (df["call_idr"] != df["call_idr"].shift()).cumsum()
    return df


def extract_idr_anchor_score_from_df(df, site):
    return df[df["position"] == int(site)][["iupred_score", "anchor_score"]]


def add_at_after_st(seq: str, position: int) -> Optional[str]:
    try:
        if position < 1 or position > len(seq):
            raise ValueError(f"Position {position} out of range (1-{len(seq)})")

        residue = seq[position - 1]
        if residue not in ["s", "t"]:
            raise ValueError(f"Residue {residue} at position {position} is not S/T")

        return seq[:position] + "@" + seq[position:]
    except (IndexError, ValueError) as exc:
        error_msg = f"Position: {position}, Sequence: {seq[:30]}{'...' if len(seq) > 30 else ''}, Error: {str(exc)}"
        with open(BASE_DIR / "processing_errors.log", "a") as f:
            f.write(error_msg + "\n")
        return None


def create_fasta_file(sequence_ids: List[str], modified_sequences: List[str], output_path: Path) -> None:
    with open(output_path, "w") as fasta_file:
        for seq_id, seq in zip(sequence_ids, modified_sequences):
            if seq is not None:
                fasta_file.write(f">{seq_id}\n{seq}\n")


def predict_fasta_file(input_fasta: Path, output_csv: Path, model_loc: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predict.run_predict(str(Path(model_loc).resolve()), str(input_fasta), str(output_csv))


def chunk_list(items, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def extract_phospholingo_score(sequences, sites, model_loc, sequence_ids=None):
    if sequence_ids is None:
        sequence_ids = [f"default_{index}" for index in range(len(sequences))]

    if len(sequence_ids) != len(sequences) or len(sequences) != len(sites):
        raise ValueError("All input lists must have the same length")

    modified_sequences = []
    valid_entries = []
    for seq_id, seq, site in zip(sequence_ids, sequences, sites):
        modified = add_at_after_st(seq, site)
        if modified:
            modified_sequences.append(modified)
            valid_entries.append((seq_id, seq, site))

    if not modified_sequences:
        raise ValueError("No valid sequences to process")

    temp_id = uuid.uuid4().hex
    temp_fasta = BASE_DIR / f"temp_{temp_id}.fasta"
    temp_output = BASE_DIR / f"temp_{temp_id}_predictions.csv"

    try:
        create_fasta_file([e[0] for e in valid_entries], modified_sequences, temp_fasta)
        predict_fasta_file(temp_fasta, temp_output, model_loc)
        results_df = pd.read_csv(temp_output)

        original_data = pd.DataFrame(
            {
                "unique_id": [e[0] for e in valid_entries],
                "original_sequence": [e[1] for e in valid_entries],
                "original_site": [e[2] for e in valid_entries],
            }
        )

        merged = original_data.merge(
            results_df,
            left_on=["unique_id", "original_site"],
            right_on=["prot_id", "position"],
            how="left",
        )
        return merged[["unique_id", "pred"]].rename(columns={"pred": "phospho_score"})
    finally:
        for f in [temp_fasta, temp_output]:
            if f.exists():
                f.unlink()


def extract_esm_embedding(seq, site, tokenizer, model):
    import torch

    _, _, device = _get_esm_resources()
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    residue_embeddings = outputs.last_hidden_state[0, 1:-1]
    site_index = site - 1

    if site_index < 0 or site_index > len(seq):
        raise ValueError("Invalid site index. It must be within the range of the sequence length.")

    site_embedding = residue_embeddings[site_index]
    protein_embedding = torch.mean(residue_embeddings, dim=0)

    protein_embedding_df = pd.DataFrame(
        [protein_embedding.cpu().numpy()],
        columns=[f"esm_protein_embedding_{i}" for i in range(protein_embedding.shape[0])],
    )
    site_embedding_df = pd.DataFrame(
        [site_embedding.cpu().numpy()],
        columns=[f"esm_residue_embedding_{i}" for i in range(site_embedding.shape[0])],
    )

    return protein_embedding_df, site_embedding_df


def extract_esm_embeddings_for_sequence(seq, tokenizer, model):
    import torch

    _, _, device = _get_esm_resources()
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    residue_embeddings = outputs.last_hidden_state[0, 1:-1].detach().cpu().numpy()
    protein_embedding = residue_embeddings.mean(axis=0)

    protein_embedding_df = pd.DataFrame(
        [protein_embedding],
        columns=[f"esm_protein_embedding_{i}" for i in range(protein_embedding.shape[0])],
    )
    return protein_embedding_df, residue_embeddings


def extract_site_embedding_from_array(residue_embeddings, site):
    site_embedding = residue_embeddings[site - 1]
    return pd.DataFrame(
        [site_embedding],
        columns=[f"esm_residue_embedding_{i}" for i in range(site_embedding.shape[0])],
    )


def one_hot_encode(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWYst"
    one_hot = np.zeros((len(sequence), len(amino_acids)), dtype=int)

    for i, char in enumerate(sequence):
        if char in amino_acids:
            index = amino_acids.index(char)
            one_hot[i, index] = 1

    return one_hot


def extract_onehot_embedding(sequence, site):
    amino_acids = "ACDEFGHIKLMNPQRSTVWYst"
    data = {}
    sub_sequence = ["-"] * 15
    site_index = site - 1
    start = max(0, site_index - 7)
    end = min(len(sequence), site_index + 8)

    for i in range(start, end):
        sub_sequence[i - (site_index - 7)] = sequence[i]

    one_hot = one_hot_encode("".join(sub_sequence))

    for pos_offset in range(-7, 8):
        position_in_onehot = pos_offset + 7
        for aa_index, aa in enumerate(amino_acids):
            column_name = f"onehot_{pos_offset}_{aa}"
            data[column_name] = [one_hot[position_in_onehot, aa_index]]

    return pd.DataFrame(data)


def extract_idr_seq(sequence, site):
    df = build_idr_dataframe(sequence)
    return extract_idr_seq_from_df(df, site)


def extract_idr_seq_from_df(df, site):
    if df.at[site - 1, "call_idr"] != 1:
        return None

    contiguous_segments = df[df["call_idr"] == 1].groupby("segment_id")

    for _, segment in contiguous_segments:
        if site in segment["position"].values:
            start_idx = segment.index[0]
            end_idx = segment.index[-1]

            if len(segment) > 500:
                site_idx = df.index[df["position"] == site][0]
                if site_idx - start_idx > 250:
                    start_idx = max(site_idx - 250, start_idx)
                end_idx = min(start_idx + 500, end_idx)

            truncated_segment = df.loc[start_idx:end_idx]
            return "".join(truncated_segment["amino_acid"].tolist())

    return None


def filter_sequence(seq):
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWYst")
    return "".join(c if c in standard_amino_acids else "A" for c in seq)


def validate_sequence(sequence):
    sequence = (sequence or "").strip()
    if not sequence:
        raise ValueError("Sequence is required.")
    return sequence


def find_candidate_sites(sequence):
    filtered_sequence = filter_sequence(sequence)
    return [index + 1 for index, residue in enumerate(filtered_sequence) if residue.lower() in {"s", "t"}]


def make_site_specific_sequence(sequence, site):
    return sequence[: site - 1] + sequence[site - 1].lower() + sequence[site:]


def _ensure_blast_path():
    if shutil.which("segmasker"):
        return

    candidate_paths = []
    executable_dir = Path(sys.executable).resolve().parent
    conda_prefix = os.environ.get("CONDA_PREFIX")

    if executable_dir.is_dir():
        candidate_paths.append(str(executable_dir))
    if conda_prefix:
        conda_bin = Path(conda_prefix) / ("Scripts" if platform.system() == "Windows" else "bin")
        if conda_bin.is_dir():
            candidate_paths.append(str(conda_bin))

    system_name = platform.system()
    if system_name == "Darwin":
        candidate_paths.append("/Users/newuser/ncbi-blast-2.16.0+/bin")
    elif system_name == "Windows":
        candidate_paths.append(r"C:\Program Files\NCBI\blast-2.16.0+\bin")

    for candidate in dict.fromkeys(candidate_paths):
        if os.path.isdir(candidate):
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("segmasker"):
                return


_ensure_blast_path()
np.random.seed(42)


def extract_seq_deephase_score(seq):
    deephase_module = _get_deephase_module()
    df = pd.DataFrame({"sequence_final": [seq]})
    deephase_result = deephase_module.DeePhase(df)
    df_deephase_score = pd.DataFrame(
        [deephase_result],
        columns=["deephase_phys_multi", "deephase_w2v_multi", "deephase_score"],
    )
    return df_deephase_score.astype(float)


def build_compactness_score_df(idr_seq, compactness_cache=None):
    if idr_seq is None or len(idr_seq) <= 3:
        return pd.DataFrame(
            [
                {
                    "nuSVR": 0.0,
                    "SconfSVR/N (kB)": 0.0,
                    "mean_lambda": 0.0,
                    "SHD": 0.0,
                    "SCD": 0.0,
                    "kappa": 0.0,
                    "FCR": 0.0,
                    "NCPR": 0.0,
                    "fK": 0.0,
                    "fR": 0.0,
                    "fE": 0.0,
                    "fD": 0.0,
                    "faro": 0.0,
                }
            ],
            index=["protein1"],
        )

    if compactness_cache is not None and idr_seq in compactness_cache:
        return compactness_cache[idr_seq].copy()

    compactness_score_df = extract_compactness_score(idr_seq)
    if compactness_cache is not None:
        compactness_cache[idr_seq] = compactness_score_df.copy()
    return compactness_score_df


def build_bio_features_from_context(site_specific_sequence, site, deephase_score_df, idr_df, compactness_cache=None):
    wt_seq = filter_sequence(site_specific_sequence)
    idr_score_df = extract_idr_anchor_score_from_df(idr_df, site)
    onehot_embedding_df = extract_onehot_embedding(wt_seq, site)
    idr_seq = extract_idr_seq_from_df(idr_df, site)
    compactness_score_df = build_compactness_score_df(idr_seq, compactness_cache)

    return pd.concat(
        [
            deephase_score_df.reset_index(drop=True),
            idr_score_df.reset_index(drop=True),
            onehot_embedding_df.reset_index(drop=True),
            compactness_score_df.reset_index(drop=True),
        ],
        axis=1,
    )


def build_plm_features_from_context(site, protein_embedding_df, residue_embeddings, phospho_score=None):
    site_embedding_df = extract_site_embedding_from_array(residue_embeddings, site).reset_index(drop=True)
    protein_embedding_df = protein_embedding_df.reset_index(drop=True)

    if phospho_score is None:
        return pd.concat([protein_embedding_df, site_embedding_df], axis=1)

    phospholingo_score_df = pd.DataFrame([{"phospho_score": float(phospho_score)}])
    return pd.concat([protein_embedding_df, site_embedding_df, phospholingo_score_df], axis=1)


def build_bio_features_df(sequence, site):
    wt_seq = filter_sequence(sequence)
    seq = wt_seq.upper()
    site = int(site)

    deephase_score_df = extract_seq_deephase_score(seq)
    idr_df = build_idr_dataframe(seq)
    return build_bio_features_from_context(wt_seq, site, deephase_score_df, idr_df)


def build_plm_features_df(sequence, site, tokenizer, model, phospholingo=True):
    wt_seq = filter_sequence(sequence)
    seq = wt_seq.upper()
    site = int(site)

    protein_embedding_df, residue_embeddings = extract_esm_embeddings_for_sequence(seq, tokenizer, model)
    gc.collect()

    if phospholingo:
        phospholingo_score_df = extract_phospholingo_score(
            [wt_seq],
            [site],
            PHOSPHOLINGO_MODEL_LOC,
            [f"site_{site}"],
        )
        phospho_score = float(phospholingo_score_df.iloc[0]["phospho_score"])
        plm_features_df = build_plm_features_from_context(site, protein_embedding_df, residue_embeddings, phospho_score)
    else:
        plm_features_df = build_plm_features_from_context(site, protein_embedding_df, residue_embeddings)

    gc.collect()
    return plm_features_df


def validate_input(sequence, site):
    sequence = validate_sequence(sequence)

    try:
        site = int(site)
    except (TypeError, ValueError) as exc:
        raise ValueError("Site must be an integer.") from exc

    if site < 1 or site > len(sequence):
        raise ValueError(f"Site must be between 1 and {len(sequence)}.")

    residue = sequence[site - 1].upper()
    if residue not in {"S", "T"}:
        raise ValueError(f"Residue at site {site} must be S or T, found '{sequence[site - 1]}'.")

    return sequence, site


def normalize_feature_frame(result_df):
    def extract_number(x):
        try:
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                x = ast.literal_eval(x)
            if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, float)):
                return x[0]
        except Exception:
            pass
        return x

    for col in result_df.columns:
        if result_df[col].dtype == "object":
            result_df[col] = result_df[col].apply(extract_number)

    return result_df


def extract_features_for_sites(sequence, sites):
    sequence = validate_sequence(sequence)
    wt_seq = filter_sequence(sequence)
    seq = wt_seq.upper()
    sites = [int(site) for site in sites]

    if not sites:
        raise ValueError("No candidate sites were provided.")

    for site in sites:
        if site < 1 or site > len(wt_seq):
            raise ValueError(f"Site must be between 1 and {len(wt_seq)}.")
        residue = wt_seq[site - 1].upper()
        if residue not in {"S", "T"}:
            raise ValueError(f"Residue at site {site} must be S or T, found '{wt_seq[site - 1]}'.")

    tokenizer, model, _ = _get_esm_resources()
    deephase_score_df = extract_seq_deephase_score(seq).reset_index(drop=True)
    idr_df = build_idr_dataframe(seq)
    protein_embedding_df, residue_embeddings = extract_esm_embeddings_for_sequence(seq, tokenizer, model)
    compactness_cache = {}

    phospho_scores_by_site = {}
    if sites:
        site_specific_sequences = [make_site_specific_sequence(wt_seq, site) for site in sites]
        sequence_ids = [f"site_{site}" for site in sites]
        chunked_rows = []
        for sequence_chunk, site_chunk, id_chunk in zip(
            chunk_list(site_specific_sequences, PHOSPHOLINGO_SITE_CHUNK_SIZE),
            chunk_list(sites, PHOSPHOLINGO_SITE_CHUNK_SIZE),
            chunk_list(sequence_ids, PHOSPHOLINGO_SITE_CHUNK_SIZE),
        ):
            phospholingo_score_df = extract_phospholingo_score(
                sequence_chunk,
                site_chunk,
                PHOSPHOLINGO_MODEL_LOC,
                id_chunk,
            )
            chunked_rows.append(phospholingo_score_df)

        if chunked_rows:
            phospholingo_score_df = pd.concat(chunked_rows, ignore_index=True)
            for _, row in phospholingo_score_df.iterrows():
                site_value = int(str(row["unique_id"]).split("_")[-1])
                phospho_scores_by_site[site_value] = float(row["phospho_score"])

    feature_frames = []
    site_rows = []
    for site in sites:
        site_specific_sequence = make_site_specific_sequence(wt_seq, site)
        bio_features_df = build_bio_features_from_context(
            site_specific_sequence,
            site,
            deephase_score_df,
            idr_df,
            compactness_cache,
        ).reset_index(drop=True)
        plm_features_df = build_plm_features_from_context(
            site,
            protein_embedding_df,
            residue_embeddings,
            phospho_scores_by_site.get(site),
        ).reset_index(drop=True)

        feature_df = pd.concat([bio_features_df, plm_features_df], axis=1)
        feature_df = feature_df[FEATURE_COLUMNS]
        feature_df = normalize_feature_frame(feature_df)
        feature_frames.append(feature_df)
        site_rows.append({"site": site, "residue": wt_seq[site - 1].upper()})

    combined_features_df = pd.concat(feature_frames, ignore_index=True)
    site_df = pd.DataFrame(site_rows)
    return combined_features_df, site_df, wt_seq


def extract_features(sequence, site):
    sequence, site = validate_input(sequence, site)
    features_df, _, _ = extract_features_for_sites(sequence, [site])
    return features_df.iloc[[0]].reset_index(drop=True)


def check_non_numeric_values(df):
    found_non_numeric = False

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            non_finite_mask = ~np.isfinite(df[column])
            if non_finite_mask.any():
                found_non_numeric = True
        else:
            found_non_numeric = True

    for column in df.columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        non_numeric_mask = numeric_series.isna() & ~df[column].isna()
        if non_numeric_mask.any():
            found_non_numeric = True

    return not found_non_numeric


def predict_score(sequence, site):
    prediction_features = extract_features(sequence, site)
    predict_model = _get_predict_model()
    prediction = predict_model.predict(prediction_features)[0]
    probability = predict_model.predict_proba(prediction_features)[0]
    return prediction, probability, prediction_features


def is_positive_prediction(prediction):
    return bool(int(prediction)) if str(prediction).isdigit() else str(prediction).lower() in {
        "1",
        "true",
        "positive",
        "binding",
    }


def predict_all_sites(sequence):
    sequence = validate_sequence(sequence)
    candidate_sites = find_candidate_sites(sequence)
    if not candidate_sites:
        raise ValueError("Sequence does not contain any S or T residues to score.")

    prediction_features, site_df, wt_seq = extract_features_for_sites(sequence, candidate_sites)
    predict_model = _get_predict_model()
    predictions = predict_model.predict(prediction_features)
    probabilities = predict_model.predict_proba(prediction_features)

    result_rows = []
    for idx, site_row in site_df.iterrows():
        negative_probability = float(probabilities[idx][0]) if len(probabilities[idx]) > 0 else 0.0
        positive_probability = float(probabilities[idx][1]) if len(probabilities[idx]) > 1 else float(max(probabilities[idx]))
        prediction = predictions[idx]
        result_rows.append(
            {
                "site": int(site_row["site"]),
                "residue": site_row["residue"],
                "prediction": prediction,
                "predicted_positive": is_positive_prediction(prediction),
                "positive_probability": positive_probability,
                "negative_probability": negative_probability,
            }
        )

    results_df = pd.DataFrame(result_rows)
    return results_df, prediction_features, wt_seq


def predict_sequence_binding(sequence, threshold=SEQUENCE_BINDING_THRESHOLD):
    results_df, prediction_features, wt_seq = predict_all_sites(sequence)
    best_row = results_df.sort_values(["positive_probability", "site"], ascending=[False, True]).iloc[0]
    max_probability = float(best_row["positive_probability"])
    predicted_positive = max_probability >= float(threshold)

    summary = {
        "predicted_positive": predicted_positive,
        "threshold": float(threshold),
        "max_probability": max_probability,
        "best_site": int(best_row["site"]),
        "best_residue": best_row["residue"],
        "best_site_prediction": best_row["prediction"],
        "site_count": int(len(results_df)),
    }
    return summary, results_df, prediction_features, wt_seq

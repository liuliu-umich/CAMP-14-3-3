import pandas as pd
import numpy as np

import src.files.iupred2a.iupred2a_lib as iupred2a_lib
def extract_idr_anchor_score(sequence, site):
    iupred_type = "long"
    # Predict disorder using iupred2a_lib
    iupred_scores = iupred2a_lib.iupred(sequence, iupred_type)[0]

    # Predict anchor regions using iupred2a_lib if anchor is enabled
    anchor_scores = iupred2a_lib.anchor2(sequence)[0]
    # print(anchor_scores)

    # Prepare the data for saving into a CSV file
    data = {
        "position": list(range(1, len(sequence) + 1)),
        "amino_acid": list(sequence),
        "iupred_score": iupred_scores,
        "anchor_score": anchor_scores,
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    # print(df)
    df["position"] = df["position"].astype(int)
    
    idr_score_df = df[df["position"] == int(site)][["iupred_score", "anchor_score"]]
        
    return idr_score_df


#@title <b>Preliminary operations</b>
import subprocess
# subprocess.run( 'pip install wget localcider==0.1.18'.split() )
# subprocess.run('pip uninstall scikit-learn -y'.split())
# subprocess.run('pip install scikit-learn==1.0.2'.split())
import numpy as np
import itertools
from localcider.sequenceParameters import SequenceParameters
import wget
import sys
import os
from joblib import dump, load
import pandas as pd
# from google.colab import files
# from ipywidgets import IntProgress
from IPython.display import display
from IPython.display import clear_output

def calc_seq_prop(seq,residues,Nc,Cc,Hc):
    seq = list(seq).copy()
    fasta_kappa = np.array(seq.copy())
    N = len(seq)
    r = residues.copy()

    # calculate properties that do not depend on charges
    fK = sum([seq.count(a) for a in ['K']])/N
    fR = sum([seq.count(a) for a in ['R']])/N
    fE = sum([seq.count(a) for a in ['E']])/N
    fD = sum([seq.count(a) for a in ['D']])/N
    faro = sum([seq.count(a) for a in ['W','Y','F']])/N
    mean_lambda = np.mean(r.loc[seq].lambdas)
    
    pairs = np.array(list(itertools.combinations(seq,2)))
    pairs_indices = np.array(list(itertools.combinations(range(N),2)))
    # calculate sequence separations
    ij_dist = np.diff(pairs_indices,axis=1).flatten().astype(float)
    # calculate lambda sums
    ll = r.lambdas.loc[pairs[:,0]].values+r.lambdas.loc[pairs[:,1]].values
    # calculate SHD
    beta = -1
    shd = np.sum(ll*np.power(np.abs(ij_dist),beta))/N

    # fix charges
    if Nc == 1:
        r.loc['X'] = r.loc[seq[0]]
        r.loc['X','q'] = r.loc[seq[0],'q'] + 1.
        seq[0] = 'X'
        if r.loc['X','q'] > 0:
            fasta_kappa[0] = 'K'
        else:
            fasta_kappa[0] = 'A'
    if Cc == 1:
        r.loc['Z'] = r.loc[seq[-1]]
        r.loc['Z','q'] = r.loc[seq[-1],'q'] - 1.
        seq[-1] = 'Z'
        if r.loc['Z','q'] < 0:
            fasta_kappa[-1] = 'D'
        else:
            fasta_kappa[-1] = 'A'
    if Hc < 0.5:
        r.loc['H', 'q'] = 0
        fasta_kappa[np.where(np.array(seq) == 'H')[0]] = 'A'
    elif Hc >= 0.5:
        r.loc['H', 'q'] = 1
        fasta_kappa[np.where(np.array(seq) == 'H')[0]] = 'K'

    # calculate properties that depend on charges
    pairs = np.array(list(itertools.combinations(seq,2)))
    # calculate charge products
    qq = r.q.loc[pairs[:,0]].values*r.q.loc[pairs[:,1]].values
    # calculate SCD
    scd = np.sum(qq*np.sqrt(ij_dist))/N
    SeqOb = SequenceParameters(''.join(fasta_kappa))
    kappa = SeqOb.get_kappa()
    fcr = r.q.loc[seq].abs().mean()
    ncpr = r.q.loc[seq].mean()
    
    return pd.Series(data=[fK,fR,fE,fD,faro,scd,shd,kappa,fcr,mean_lambda,ncpr],
                 index=['fK','fR','fE','fD','faro','SCD','SHD','kappa','FCR','mean_lambda','NCPR'])

def extract_compactness_score(idr_seq):
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    url = 'https://github.com/KULL-Centre/_2023_Tesei_IDRome/blob/main'

    if os.path.exists('svr_model_nu.joblib') == False:
        wget.download(url+'/svr_models/svr_model_nu.joblib?raw=true')
    if os.path.exists('svr_model_SPR.joblib') == False:
        wget.download(url+'/svr_models/svr_model_SPR.joblib?raw=true')
    if os.path.exists('residues.csv') == False:
        wget.download(url+'/md_simulations/data/residues.csv?raw=true')

    model_nu = load('svr_model_nu.joblib') 
    model_spr = load('svr_model_SPR.joblib') 
    features_nu = ['SCD','SHD','kappa','FCR','mean_lambda']
    features_spr = ['SCD','SHD','mean_lambda']

    residues = pd.read_csv('residues.csv')
    residues = residues.set_index('one')

    fasta_dict = {}
    df = pd.DataFrame(columns=['nuSVR','SconfSVR/N (kB)','mean_lambda','SHD','SCD','kappa','FCR','NCPR',
                               'fK','fR','fE','fD','faro'])
    
    
    # Define the FASTA format: start with the header, then the sequence
    fasta_dict = {}
    protein_name = "protein1"  # You can specify any name, here I use "protein1"

    # Convert sequence string to FASTA format
    fasta_dict[protein_name] = idr_seq
    current_upload = [protein_name]

    # Validate the sequence
    for x in list(current_upload):  # Create a copy for safe modification
        valid = True
        for a in fasta_dict[x]:
            if a not in aa:
                print(f'WARNING: {x} sequence contains a character ({a}) not recognized as an amino acid. This sequence will be ignored.')
                del fasta_dict[x]
                valid = False
                break
        if not valid:
            current_upload.remove(x)

    # Output the results
    print("Valid sequences uploaded in FASTA format:")
    for name in fasta_dict:
        print(f">{name}")
        print(fasta_dict[name])
        
    #@title <b>Define charge states</b>
    #@markdown Define charge states:
    charged_N_terminal_amine = False #@param {type:"boolean"}
    charged_C_terminal_carboxyl = True #@param {type:"boolean"}
    charged_histidine = False #@param {type:"boolean"}
    Nc = 1 if charged_N_terminal_amine == True else 0
    Cc = 1 if charged_C_terminal_carboxyl == True else 0
    if charged_histidine == False:
        Hc = 0
        
    #@title <b>Predict $\nu$ and $S_\text{conf}/N$
    #@markdown Use this cell to calculate sequence features and predict the scaling exponent, $\nu$, and the conformational entropy per residue, $S_\text{conf}/N$. Results will be downloaded in a csv file.

    # f = IntProgress(min=0, max=len(fasta_dict), description='Progress:', bar_style='warning')
    # display(f)

    for k in fasta_dict.keys():
        res = calc_seq_prop(fasta_dict[k],residues,Nc,Cc,Hc)
        nu = np.around(model_nu.predict(res.loc[features_nu].values.reshape(1, -1))[0],3)
        spr = np.around(model_spr.predict(res.loc[features_spr].values.reshape(1, -1))[0],3)
        df.loc[k,'nuSVR'] = nu
        df.loc[k,'SconfSVR/N (kB)'] = spr
        df.loc[k,res.index.values] = np.around(res.loc[res.index.values].values,3)
        # f.value += 1

    clear_output()
#     df.to_csv('svr_pred.csv',index_label='name')
    
    return df


import argparse
import sys
import os
import uuid
import pandas as pd
from pathlib import Path
from typing import List, Optional

# 1. Add PhosphoLingo to Python path

# # Get the directory containing this script
# SCRIPT_DIR = Path(__file__).resolve().parent

# # 1. Add PhosphoLingo to Python path
# PHOSPHOLINGO_DIR = SCRIPT_DIR / "src/files/PhosphoLingo/phosphoLingo"  # Adjust relative path as needed
# sys.path.insert(0, str(PHOSPHOLINGO_DIR))


phospholingo_dir = Path("src/files/PhosphoLingo/phosphoLingo").resolve()  # Update this path!
sys.path.insert(0, str(phospholingo_dir))

# 2. Import AFTER path adjustment
import predict

def add_at_after_st(seq: str, position: int) -> Optional[str]:
    """
    Add '@' after specified position if residue is S/T with proper error handling
    
    Args:
        seq: Input protein sequence
        position: 1-based position to modify
        
    Returns:
        Modified sequence or None if invalid
    """
    try:
        if position < 1 or position > len(seq):
            raise ValueError(f"Position {position} out of range (1-{len(seq)})")
            
        residue = seq[position-1]
        if residue not in ['s', 't']:
            raise ValueError(f"Residue {residue} at position {position} is not S/T")
            
        return seq[:position] + "@" + seq[position:]
        
    except (IndexError, ValueError) as e:
        error_msg = f"Position: {position}, Sequence: {seq[:30]}{'...' if len(seq)>30 else ''}, Error: {str(e)}"
        with open("processing_errors.log", "a") as f:
            f.write(error_msg + "\n")
        return None

def create_fasta_file(sequence_ids: List[str], modified_sequences: List[str], output_path: Path) -> None:
    """
    Create FASTA file from modified sequences
    
    Args:
        sequence_ids: List of sequence identifiers
        modified_sequences: List of modified sequences
        output_path: Path to output FASTA file
    """
    with open(output_path, 'w') as fasta_file:
        for seq_id, seq in zip(sequence_ids, modified_sequences):
            if seq is not None:
                fasta_file.write(f">{seq_id}\n{seq}\n")

def predict_fasta_file(input_fasta: Path, output_csv: Path, model_loc: str) -> None:
    """
    Run PhosphoLingo prediction on a FASTA file
    
    Args:
        input_fasta: Path to input FASTA file
        output_csv: Path to output predictions CSV
        model_loc: Path to PhosphoLingo model
    """
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Run prediction
    predict.run_predict(
        str(Path(model_loc).resolve()),
        str(input_fasta),
        str(output_csv)
    )

def extract_phospholingo_score(sequences, sites, model_loc, sequence_ids = ["default"]):
    """
    Batch process multiple sequences and extract phosphorylation scores
    
    Args:
        sequence_ids: List of unique sequence identifiers
        sequences: List of protein sequences
        sites: List of 1-based positions to predict
        model_loc: Path to PhosphoLingo model
        
    Returns:
        DataFrame with predictions and original data
    """
    # Validate input
    if len(sequence_ids) != len(sequences) or len(sequences) != len(sites):
        raise ValueError("All input lists must have the same length")
        
    # Generate modified sequences
    modified_sequences = []
    valid_entries = []
    for seq_id, seq, site in zip(sequence_ids, sequences, sites):
        modified = add_at_after_st(seq, site).upper()
        print(modified)
        if modified:
            modified_sequences.append(modified)
            valid_entries.append((seq_id, seq, site))
            
    if not modified_sequences:
        raise ValueError("No valid sequences to process")
        
    # Create temporary files
    temp_id = uuid.uuid4().hex
    temp_fasta = Path(f"temp_{temp_id}.fasta")
    temp_output = Path(f"temp_{temp_id}_predictions.csv")
    
    try:
        # Create FASTA file
        create_fasta_file(
            [e[0] for e in valid_entries],
            modified_sequences,
            temp_fasta)
        
        # Run predictions
        predict_fasta_file(temp_fasta, temp_output, model_loc)
        
        # Load and process results
        results_df = pd.read_csv(temp_output)
        
        # Merge with original data
        original_data = pd.DataFrame({
            'unique_id': [e[0] for e in valid_entries],
            'original_sequence': [e[1] for e in valid_entries],
            'original_site': [e[2] for e in valid_entries]})
        
        merged = original_data.merge(
            results_df,
            left_on=['unique_id', 'original_site'],
            right_on=['prot_id', 'position'],
            how='left')
        
        # Cleanup columns
        result_cols = ['unique_id', 'pred']
        return merged[result_cols].rename(columns={'pred': 'phospho_score'})
        
    finally:
        # Cleanup temporary files
        for f in [temp_fasta, temp_output]:
            if f.exists():
                f.unlink()


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Load the ESM-2 model and tokenizer once (outside the function)
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_esm_embedding(seq, site, tokenizer, model):
    """
    Extract protein-level and site-specific embeddings using ESM-2.
    
    Args:
        seq (str): Protein sequence.
        site (int): site of the protein for extrac embeddings.
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained ESM-2 model.
    
    Returns:
        protein_embedding_df (pd.DataFrame): Protein-level embedding.
        site_embedding_df (pd.DataFrame): Site-specific embedding.
    """
    # Tokenize the sequence
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    
    # Get embeddings
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get residue-level embeddings (last layer)
    residue_embeddings = outputs.last_hidden_state[0, 1:-1]  # Remove special tokens

    # change protein site into site index
    site_index = site-1
    
    # Ensure the site is within the sequence length
    if site_index < 0 or site_index > len(seq):
        raise ValueError("Invalid site index. It must be within the range of the sequence length.")
    
    # Get the site-specific embedding
    site_embedding = residue_embeddings[site_index]
    
    # Get protein-level embedding (mean pooling)
    protein_embedding = torch.mean(residue_embeddings, dim=0)
    
    # Convert to DataFrames with dynamic column names
    protein_embedding_df = pd.DataFrame([protein_embedding.cpu().numpy()],
                                        columns=[f"esm_protein_embedding_{i}" for i in range(protein_embedding.shape[0])])
    site_embedding_df = pd.DataFrame([site_embedding.cpu().numpy()],
                                     columns=[f"esm_residue_embedding_{i}" for i in range(site_embedding.shape[0])])
    
    return protein_embedding_df, site_embedding_df


# import pandas as pd

# #DO NOT CHANGE ANYTHING IN THIS CELL. MOVE ON TO THE FOLLOWING ONE TO GET THE PREDICTION.
# import os
# os.environ['PATH'] = "/Users/newuser/ncbi-blast-2.16.0+/bin:" + os.environ['PATH']
# os.environ['PATH'] = "C:/Program Files/NCBI/blast-2.16.0+/bin" + ";" + os.environ['PATH']


# import numpy as np
# from gensim.models import word2vec

# class ProtVec(word2vec.Word2Vec):

#     def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  sg=1, window=25, min_count=1, workers=20):
#         """
#         Either fname or corpus is required.
#         fasta_fname: fasta file for corpus
#         corpus: corpus object implemented by gensim
#         n: n of n-gram
#         corpus_fname: corpus file path
#         min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
#         """

#         self.n = n
#         self.size = size
#         self.fasta_fname = fasta_fname

#         if corpus is None and fasta_fname is None:
#             raise Exception("Either fasta_fname or corpus is needed!")

#         if fasta_fname is not None:
#             print('Generate Corpus file from fasta file...')
#             generate_corpusfile(fasta_fname, n, corpus_fname)
#             corpus = word2vec.Text8Corpus(corpus_fname)

#         word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

#     def to_vecs(self, seq):
#         """
#         convert sequence to three n-length vectors
#         e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
#         """
#         ngram_patterns = split_ngrams(seq, self.n)

#         protvecs = []
#         for ngrams in ngram_patterns:
#             ngram_vecs = []
#             for ngram in ngrams:
#                 try:
#                     ngram_vecs.append(self.wv[ngram])
#                 except:
#                     raise Exception("Model has never trained this n-gram: " + ngram)
#             protvecs.append(sum(ngram_vecs))
#         return protvecs
    
    
#     def get_vector(self, seq):
#         """
#         sum and normalize the three n-length vectors returned by self.to_vecs
#         """
#         #return normalize(sum(self.to_vecs(seq)))
#         return sum(self.to_vecs(seq))

    
# def load_protvec(model_fname):
#     return word2vec.Word2Vec.load(model_fname)

# pv = load_protvec('src/files/DeePhase/__PREDICT/tools/Embeddings/swissprot_size200_window25.model')

# SEED = 42
# np.random.seed(SEED)

# from src.files.DeePhase.__PREDICT.deephase_utils import *

# def extract_seq_deephase_score(seq):
#     # # Create a DataFrame with the input sequence
#     df = pd.DataFrame({'sequence_final': [seq]})
    
#     # Call the DeePhase function (assuming it returns a string)
#     deephase_result = DeePhase(df)

#     df_deephase_score = pd.DataFrame([deephase_result], columns=['deephase_phys_multi', 'deephase_w2v_multi', 'deephase_score'])

#     df_deephase_score = df_deephase_score.astype(float)

#     return df_deephase_score


# need to run under docker enviroment
# can not run directly from here
# setup PTM-mamba into the folder 1433predictor\src\files\ptm-mamba-main, run infer_1433.py for inference.

# processing the sequence to fit for the requirement for ptmmamba under the folder 1433predictor\src\files\ptm-mamba-main
# save the processed seq and index into a csv file name as: ptmmamba_seq_index.csv
import pandas as pd
import os

# mac add path
# os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/bin/docker'

# windows add path

def modify_sequence_for_phosphorylation(seq, site):
    site_index = site - 1
    print(seq)
    if site_index < 0 or site_index > len(seq):
        raise ValueError("Site index is out of range for the sequence length.")
        return None
            
    original_residue = seq[site_index]

    if original_residue not in ["s", "t"]:
        raise ValueError(f"Residue at site {site_index+1} is neither 's' nor 't'. It's '{original_residue}'.")
        return None
        
    if seq[-1] in ["s", "t"]:
        seq += "A"
        
    modified_seq = ""
    for c in seq:    
        if c == 's':
            modified_residue = "<Phosphoserine>"
            modified_seq += modified_residue
        elif c == 't':
            modified_residue = "<Phosphothreonine>"
            modified_seq += modified_residue
        else:
            modified_seq += c
    print(modified_seq)
    return modified_seq

def save_sequences_to_csv(sequences, sites, filename="src/files/ptm-mamba-main/ptmmamba_seq_index.csv"):
    data = []
    
    for seq, site in zip(sequences, sites):
        try:
            modified_sequence = modify_sequence_for_phosphorylation(seq, site)
            data.append({"ptmmamba_seq": modified_sequence, "site_index": site})
            print(data)
        except ValueError as e:
            print(f"Skipping sequence due to error: {e}")
            continue

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# function to running ptmmamba in docker
# read the ptmmamba results into a df
def extract_ptmmamba_embedding(sequence, site):    
    save_sequences_to_csv([sequence], [site], filename="src/files/ptm-mamba-main/ptmmamba_seq_index.csv")
    # save_sequences_to_csv([sequence], [site-1], filename="H:/ptm-mamba-main/ptmmamba_seq_index.csv")
    
    # !docker start plm_benji && docker exec plm_benji python infer_1433.py && docker stop plm_benji
    
    import subprocess

    # Start the Docker container
    start_container = subprocess.run(["docker", "start", "plm_benji"], check=True)

    # Execute the Python script inside the Docker container
    exec_script = subprocess.run(["docker", "exec", "plm_benji", "python", "infer_1433.py"], check=True)

    # Stop the Docker container
    # stop_container = subprocess.run(["docker", "stop", "plm_benji"], check=True)

    # read the output or handle errors
    try:
        if start_container.returncode == 0 and exec_script.returncode == 0:
            print("All commands executed successfully.")
            try:
                df_temp = pd.read_csv("src/files/ptm-mamba-main/ptmmamba_embeddings.csv")
                phosphosite_embedding_df = df_temp.filter(like="ptmmamba_residue_embedding")
                phosphoprotein_embedding_df = df_temp.filter(like="ptmmamba_protein_embedding")
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                phosphosite_embedding_df = pd.DataFrame()  # Empty DataFrame instead of None
                phosphoprotein_embedding_df = pd.DataFrame()  # Empty DataFrame instead of None
            finally:
                # Attempt to delete the CSV file regardless of read success
                try:
                    os.remove("src/files/ptm-mamba-main/ptmmamba_embeddings.csv")
                    print("CSV file deleted successfully.")
                except OSError as e:
                    print(f"Failed to delete CSV file: {e}")
        else:
            print("An error occurred while executing the commands.")
            phosphosite_embedding_df = pd.DataFrame()  # Empty DataFrame instead of None
            phosphoprotein_embedding_df = pd.DataFrame()  # Empty DataFrame instead of None
            # Even if commands failed, try to delete CSV if it exists
            try:
                if os.path.exists("src/files/ptm-mamba-main/ptmmamba_embeddings.csv"):
                    os.remove("src/files/ptm-mamba-main/ptmmamba_embeddings.csv")
                    print("CSV file deleted successfully.")
            except OSError as e:
                print(f"Failed to delete CSV file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        phosphosite_embedding_df = pd.DataFrame()  # Empty DataFrame instead of None
        phosphoprotein_embedding_df = pd.DataFrame()  # Empty DataFrame instead of None
        
    return phosphoprotein_embedding_df, phosphosite_embedding_df


def one_hot_encode(sequence):
    """
    One-hot encodes a protein sequence.

    Parameters:
        sequence (str): The protein sequence to be one-hot encoded.

    Returns:
        np.ndarray: A one-hot encoded representation of the sequence.
    """

    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYst'
    one_hot = np.zeros((len(sequence), len(AMINO_ACIDS)), dtype=int)
    
    for i, char in enumerate(sequence):
        if char in AMINO_ACIDS:
            index = AMINO_ACIDS.index(char)
            one_hot[i, index] = 1
    
    return one_hot

def extract_onehot_embedding(sequence, site):
    """
    Extracts subsequences around specified phosphorylation sites and one-hot encodes them.

    Parameters:
        sequence (str): The protein sequence.
        site (int): The phosphorylation site (1-based index).

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded features for the sequence window.
    """
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYst'
    data = {}  # Dictionary to hold data for new columns

    # Initialize a sequence of 15 "-" characters
    sub_sequence = ['-'] * 15

    # Convert to 0-based index
    site_index = site - 1
    
    # Determine the valid window around the phosphorylation site
    start = max(0, site_index - 7)
    end = min(len(sequence), site_index + 8)

    # Fill the sub_sequence with valid residues
    for i in range(start, end):
        sub_sequence[i - (site_index - 7)] = sequence[i]

    # One-hot encode the window
    one_hot = one_hot_encode(''.join(sub_sequence))

    # Create feature columns for each position and amino acid
    for pos_offset in range(-7, 8):
        position_in_onehot = pos_offset + 7  # Convert to 0-based index (0-14)
        for aa_index, aa in enumerate(AMINO_ACIDS):
            column_name = f"onehot_{pos_offset}_{aa}"
            data[column_name] = [one_hot[position_in_onehot, aa_index]]

    return pd.DataFrame(data)    

import os
import sys
import pandas as pd
import numpy as np
import src.files.iupred2a.iupred2a_lib as iupred2a_lib
from Bio import SeqIO

def extract_idr_seq(sequence, site):
    iupred_type = "long"
    # Predict disorder using iupred2a_lib
    iupred_scores = iupred2a_lib.iupred(sequence, iupred_type)[0]

    # Predict anchor regions using iupred2a_lib if anchor is enabled
    anchor_scores = iupred2a_lib.anchor2(sequence)[0]

    # Prepare the data for saving into a CSV file
    data = {
        "position": list(range(1, len(sequence) + 1)),
        "amino_acid": list(sequence),
        "iupred_score": iupred_scores,
        "anchor_score": anchor_scores,
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Step 1: Add the 'call_idr' column
    df['call_idr'] = df['iupred_score'].apply(lambda x: 1 if x > 0.5 else 0)
    
    # Check if the provided site is within a '1' region
    if df.at[site-1, 'call_idr'] != 1:
        return None

    # Find the contiguous segments where call_idr == 1
    df['segment_id'] = (df['call_idr'] != df['call_idr'].shift()).cumsum()
    contiguous_segments = df[df['call_idr'] == 1].groupby('segment_id')
    
    for _, segment in contiguous_segments:
        # Check if the site falls within this segment
        if site in segment['position'].values:
            # Truncate the segment to a maximum length of 500 if necessary
            start_idx = segment.index[0]
            end_idx = segment.index[-1]

            if len(segment) > 500:
                # Ensure the site is within the truncated segment
                site_idx = df.index[df['position'] == site][0]
                if site_idx - start_idx > 250:
                    start_idx = max(site_idx - 250, start_idx)
                end_idx = min(start_idx + 500, end_idx)

            truncated_segment = df.loc[start_idx:end_idx]
            return "".join(truncated_segment['amino_acid'].tolist())

    return None

def filter_sequence(seq):
    # Define standard amino acids
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWYst")

    
    # Replace non-standard amino acids with 'A'
    filtered_seq = ''.join(c if c in standard_amino_acids else 'A' for c in seq)
    
    return filtered_seq

import subprocess
import platform

def is_docker_running():
    try:
        # Run a simple Docker command to check if it's responding
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        # 'docker' command not found, Docker is not installed
        return False

def start_docker():
    system = platform.system()

    if system == "Linux":
        # On most Linux systems, use systemctl to start Docker
        try:
            subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)
            print("Docker started on Linux.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start Docker on Linux: {e}")

    elif system == "Darwin":
        # On macOS, use the Docker Desktop.app
        try:
            subprocess.run(["open", "-a", "Docker"], check=True)
            print("Docker started on macOS.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start Docker on macOS: {e}")

    elif system == "Windows":
        # On Windows, use PowerShell to start Docker Desktop
        try:
            subprocess.run(["powershell", "Start-Process", "-FilePath", '"C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"'], check=True)
            print("Docker started on Windows.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start Docker on Windows: {e}")

    else:
        print("Unsupported operating system for automatic Docker start")


import pandas as pd
import numpy as np
#@title <b>Preliminary operations</b>
import subprocess
# subprocess.run( 'pip install wget localcider==0.1.18'.split() )
# subprocess.run('pip uninstall scikit-learn -y'.split())
# subprocess.run('pip install scikit-learn==1.0.2'.split())
import itertools
from localcider.sequenceParameters import SequenceParameters
import wget
import os
from joblib import dump, load
# from google.colab import files
# from ipywidgets import IntProgress
from IPython.display import display
from IPython.display import clear_output
import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModel
import src.files.iupred2a.iupred2a_lib as iupred2a_lib
from Bio import SeqIO
# from src.files.utils import *

import pandas as pd

#DO NOT CHANGE ANYTHING IN THIS CELL. MOVE ON TO THE FOLLOWING ONE TO GET THE PREDICTION.
import os
os.environ['PATH'] = "/Users/newuser/ncbi-blast-2.16.0+/bin:" + os.environ['PATH']
os.environ['PATH'] = "C:/Program Files/NCBI/blast-2.16.0+/bin" + ";" + os.environ['PATH']


import numpy as np
from gensim.models import word2vec


class ProtVec(word2vec.Word2Vec):

    def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  sg=1, window=25, min_count=1, workers=20):
        """
        Either fname or corpus is required.
        fasta_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        corpus_fname: corpus file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """

        self.n = n
        self.size = size
        self.fasta_fname = fasta_fname

        if corpus is None and fasta_fname is None:
            raise Exception("Either fasta_fname or corpus is needed!")

        if fasta_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(fasta_fname, n, corpus_fname)
            corpus = word2vec.Text8Corpus(corpus_fname)

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def to_vecs(self, seq):
        """
        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self.wv[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs
    
    
    def get_vector(self, seq):
        """
        sum and normalize the three n-length vectors returned by self.to_vecs
        """
        #return normalize(sum(self.to_vecs(seq)))
        return sum(self.to_vecs(seq))

    
# def load_protvec(model_fname):
#     return word2vec.Word2Vec.load(model_fname)

# pv = load_protvec('src/files/DeePhase/__PREDICT/tools/Embeddings/swissprot_size200_window25.model')

import __main__
__main__.ProtVec = ProtVec  # Make ProtVec available in the __main__ context

def load_protvec(model_fname):
    return ProtVec.load(model_fname)  # Use ProtVec's load method

pv = load_protvec('src/files/DeePhase/__PREDICT/tools/Embeddings/swissprot_size200_window25.model')

SEED = 42
np.random.seed(SEED)

from src.files.DeePhase.__PREDICT.deephase_utils import *

def extract_seq_deephase_score(seq):
    # # Create a DataFrame with the input sequence
    df = pd.DataFrame({'sequence_final': [seq]})
    
    # Call the DeePhase function (assuming it returns a string)
    deephase_result = DeePhase(df)

    df_deephase_score = pd.DataFrame([deephase_result], columns=['deephase_phys_multi', 'deephase_w2v_multi', 'deephase_score'])

    df_deephase_score = df_deephase_score.astype(float)

    return df_deephase_score

# build the features df:
def build_bio_features_df(sequence, site):
    
    wt_seq = filter_sequence(sequence) # sequence only have "ACDEFGHIKLMNPQRSTVWYst"
    print("filter_sequence: ", wt_seq)
    seq = wt_seq.upper()
    site = int(site)
        
    deephase_score_df = extract_seq_deephase_score(seq)
    # print("deephase_score_df:", deephase_score_df)
    
    idr_score_df = extract_idr_anchor_score(seq, site)
    # print("idr_score_df and achor_score_df:", idr_score_df)

    onehot_embedding_df = extract_onehot_embedding(wt_seq, site)
    # print("onehot_embedding_df:", onehot_embedding_df)
    
    idr_seq = extract_idr_seq(seq, site)
    # print("idr_seq:", idr_seq)

    if idr_seq is None or len(idr_seq) <= 3:
        data = {
        'nuSVR': 0.0,
        'SconfSVR/N (kB)': 0.0,
        'mean_lambda': 0.0,
        'SHD': 0.0,
        'SCD': 0.0,
        'kappa': 0.0,
        'FCR': 0.0,
        'NCPR': 0.0,
        'fK': 0.0,
        'fR': 0.0,
        'fE': 0.0,
        'fD': 0.0,
        'faro': 0.0
        }
        compactness_score_df = pd.DataFrame([data], index=['protein1'])
        # print("idr_seq is None, compactness_score_df:", compactness_score_df)        
    else:
        compactness_score_df = extract_compactness_score(idr_seq)
        # print("compactness_score_df:", compactness_score_df)

    
    # merge feature dfs
    deephase_score_df = deephase_score_df.reset_index(drop=True)
    idr_score_df = idr_score_df.reset_index(drop=True)
    onehot_embedding_df = onehot_embedding_df.reset_index(drop=True)
    compactness_score_df = compactness_score_df.reset_index(drop=True)
    
    bio_features_df = pd.concat([deephase_score_df, idr_score_df, onehot_embedding_df, compactness_score_df], axis=1)
    
    return bio_features_df



# Load the ESM-2 model and tokenizer once (outside the function)
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import gc

def build_plm_features_df(sequence, site, tokenizer, model, phospholingo=True):

    # phospholingo model location
    # model_loc = "/Users/newuser/PhosphoLingo_ST_new.ckpt" 
    model_loc = r"H:\PhosphoLingo_ST_new.ckpt"
    
    # # Move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # make sure docker is running
    if not is_docker_running():
        print("Docker is not running. Attempting to start Docker...")
        start_docker()
    else:
        print("Docker is already running.")

    
    wt_seq = filter_sequence(sequence) # sequence only have "ACDEFGHIKLMNPQRSTVWYst"
    print("filter_sequence: ", wt_seq)
    seq = wt_seq.upper()
    site = int(site)
    print(site)

    protein_embedding_df, site_embedding_df = extract_esm_embedding(seq, site, tokenizer, model)
    # print("protein_embedding_df, site_embedding_df:", protein_embedding_df, site_embedding_df)

    phosphoprotein_embedding_df, phosphosite_embedding_df = extract_ptmmamba_embedding(wt_seq, site)
    # print("phosphoprotein_embedding_df, phosphosite_embedding_df:", phosphoprotein_embedding_df, phosphosite_embedding_df)
    gc.collect()
    if phospholingo:
        phospholingo_score_df = extract_phospholingo_score([wt_seq], [site], model_loc)
        # print("phospholingo_score_df:", phospholingo_score_df)    
        # merge feature dfs
        plm_features_df = pd.concat([protein_embedding_df, site_embedding_df, phosphoprotein_embedding_df, phosphosite_embedding_df, phospholingo_score_df], axis=1)
    else:
        plm_features_df = pd.concat([protein_embedding_df, site_embedding_df, phosphoprotein_embedding_df, phosphosite_embedding_df], axis=1)
        

    gc.collect() 
    
    return plm_features_df

def extract_features(sequence, site):
    site = int(site)

    # Create a new sequence with the letter at the specified site lowercased
    if 0 <= site-1 < len(sequence):
        new_sequence = sequence[:site-1] + sequence[site-1].lower() + sequence[site:]
        protein_sequences = new_sequence              
        
    # Process the new data
    bio_features_df = build_bio_features_df(protein_sequences, site)
    plm_features_df = build_plm_features_df(protein_sequences, site, tokenizer, model, phospholingo=True)
    # plm_features_df = build_plm_features_df(protein_sequences, site, tokenizer, model, phospholingo=False)

    
    bio_features_df = bio_features_df.reset_index(drop=True)
    print(bio_features_df)
    plm_features_df = plm_features_df.reset_index(drop=True)
    print(plm_features_df)

    result_df = pd.concat([bio_features_df, plm_features_df], axis=1)
    print(len(result_df))
    
    optimal_features = ['esm_residue_embedding_482', 'esm_residue_embedding_197', 'esm_residue_embedding_844', 'onehot_1_L', 'esm_residue_embedding_715', 'esm_residue_embedding_1259', 'ptmmamba_residue_embedding_79', 'esm_residue_embedding_1074', 'iupred_score', 'esm_residue_embedding_53', 'onehot_1_P', 'onehot_-3_R', 'onehot_-1_H', 'onehot_1_R', 'esm_residue_embedding_327', 'phospho_score', 'esm_residue_embedding_1041', 'onehot_-1_K', 'onehot_-3_E', 'onehot_7_G', 'onehot_-4_R', 'esm_residue_embedding_394', 'ptmmamba_residue_embedding_408', 'onehot_2_P', 'esm_residue_embedding_546', 'onehot_-2_G', 'esm_residue_embedding_491', 'esm_residue_embedding_1215', 'onehot_-2_S', 'esm_residue_embedding_81', 'onehot_-5_R', 'onehot_3_D', 'esm_protein_embedding_68', 'nuSVR', 'esm_residue_embedding_356', 'onehot_-2_R', 'esm_residue_embedding_385', 'esm_residue_embedding_1171', 'deephase_w2v_multi', 'esm_residue_embedding_1242']

    result_df = result_df[optimal_features]

    def extract_number(x):
        try:
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                x = ast.literal_eval(x)
            if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, float)):
                return x[0]
        except:
            pass
        return x  # fallback if nothing works

    for col in result_df.columns:
        if result_df[col].dtype == 'object':
            result_df[col] = result_df[col].apply(extract_number)

    print(len(result_df))
    print(result_df)

    return result_df
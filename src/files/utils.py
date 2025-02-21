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



def extract_idr_achor_score(sequence, site):
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
    
    idr_score_df = df[df["position"] == site][["iupred_score", "anchor_score"]]
        
    return idr_score_df


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


# for a give sequence and position, add "@" after if the position is "S" or "T"
def add_at_after_st(seq, position):
    try:
        if seq[position-1] in ['S', 'T']:
            return seq[:position] + "@" + seq[position:]
    except:
        filename = "error_unique_position.txt"
        with open(filename, "a") as fasta_file:
            fasta_file.write(f"{position, seq}\n")            
    else:
        filename = "error_unique_position.txt"
        with open(filename, "a") as fasta_file:
            fasta_file.write(f"{position, seq}\n")


def extract_phospholingo_score(sequence, site, model_loc):

    # model_loc = "H:\PhosphoLingo_ST_new.ckpt"    
    dataset_fasta = "test.fasta"
    output_file = "test.csv"

    sequence_modified = add_at_after_st(sequence, site)
    # print(sequence_modified)
    # print(output_file)

    # generate
    # Open the file in write mode
    with open(dataset_fasta, 'w') as fasta_file:
        # Write the header line
        header_line = f">test\n"
        fasta_file.write(header_line)

        # Write the sequence
        fasta_file.write(sequence_modified + '\n')

    import subprocess

    # Define your command as a list of strings
    command = [
        'python',
        'src/files/PhosphoLingo/phospholingo',
        'predict',
        model_loc,
        dataset_fasta,
        output_file
    ]
    
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output
    print("Standard Output:")
    print(result.stdout)
    
    print("\nStandard Error:")
    print(result.stderr)

    df = pd.read_csv(output_file)
    # Locate the row with the specific position and select only the 'pred' column
    filtered_df = df.loc[df['position'] == site, ['pred']]
    filtered_df.columns = ['phospholingo_score']
    
    # Check if the filtered_df has exactly one row
    if len(filtered_df) == 1:
        print("Filtered DataFrame with the pred score:")
        print(filtered_df)
    else:
        print(f"Filtered DataFrame is empty or contains multiple rows for position {site}.")
        
    return filtered_df


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

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
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




# mac add path
# os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/bin/docker'
# windows add path

def modify_sequence_for_phosphorylation(seq, site):
    site_index = site - 1
    print(seq)
    if site_index < 0 or site_index > len(seq):
        raise ValueError("Site index is out of range for the sequence length.")
            
    original_residue = seq[site_index]

    if original_residue not in ["s", "t"]:
        raise ValueError(f"Residue at site {site_index} is neither 'S' nor 'T'. It's '{original_residue}'.")
        
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
    stop_container = subprocess.run(["docker", "stop", "plm_benji"], check=True)

    # Optionally, print the output or handle errors
    if start_container.returncode == 0 and exec_script.returncode == 0 and stop_container.returncode == 0:
        print("All commands executed successfully.")
    else:
        print("An error occurred while executing the commands.")
    
    df_temp = pd.read_csv("src/files/ptm-mamba-main/ptmmamba_embeddings.csv")
    
    phosphosite_embedding_df = df_temp.filter(like="ptmmamba_residue_embedding")
    
    phosphoprotein_embedding_df = df_temp.filter(like="ptmmamba_protein_embedding")
    
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



def mutate_seq(seq, mutations, site):
    """
    Mutates a sequence based on a list of mutations and checks a specific site.

    Parameters:
    seq (str): The original sequence to mutate.
    mutations (list of str): List of mutation instructions.
    site (int): The site to verify in the sequence.

    Returns:
    str: The mutated sequence.
    """
    # Convert seq to a list for mutability
    seq = list(seq)

    # Check the first mutation condition
    first_mutation = mutations[0]
    expected_char = first_mutation[0]  # Expected character at given site
    site_to_check = int(extract_numeric_part(first_mutation))

    # Check if the character at the given site matches
    if site_to_check == site and seq[site - 1] == expected_char:
        # Apply the rest of the mutations
        for mutation in mutations[1:]:            
            mut_site = int(extract_numeric_part(mutation))
            new_char = mutation[-1]  # Expected new character
            expected_char = mutation[0]

            if seq[mut_site - 1] == expected_char:
                # Apply the mutation
                seq[mut_site - 1] = new_char
            else:
                raise ValueError(f"mutation AA not match the site in the sequence ")
    else:
        raise ValueError(f"phosphorylation site in input mut_lst not match the site in sequence")
        
    # Convert the list back to a string
    mutated_seq = ''.join(seq)
    
    return mutated_seq


def build_mutation_lst(mutation_string):
    # Split the string by '/'
    parts = input_string.split('/')
    
    # Return the list of parts
    return parts



def extract_numeric_part(s):
    """Extracts and returns the numeric part from a string."""
    return ''.join(filter(str.isdigit, s))



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

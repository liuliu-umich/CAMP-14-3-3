from collections import namedtuple
import torch
import esm
from typing import List, Union, Optional
from protein_lm.modeling.scripts.train import compute_esm_embedding, load_ckpt, make_esm_input_ids
from protein_lm.tokenizer.tokenizer import PTMTokenizer
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

Output = namedtuple("output", ["logits", "hidden_states"])

class PTMMamba:
    def __init__(self, ckpt_path, device='cuda',use_esm=True) -> None:
        self.use_esm = use_esm
        self._tokenizer = PTMTokenizer()
        self._model = load_ckpt(ckpt_path, self.tokenizer, device)
        self._device = device
        self._model.to(device)
        self._model.eval()
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()

    @property
    def model(self) -> torch.nn.Module:
        return self._model


    @property
    def tokenizer(self) -> PTMTokenizer:
        return self._tokenizer
    
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    
    
    def infer(self, seq: str) -> Output:
        input_id = self.tokenizer(seq)
        input_ids = torch.tensor(input_id,device=self.device).unsqueeze(0)
        outputs = self._infer(input_ids)
        return outputs
    
    @torch.no_grad()
    def _infer(self, input_ids):
        if self.use_esm:
            esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
            embedding = compute_esm_embedding(
                self.tokenizer, self.esm_model, self.batch_converter, esm_input_ids
            )
        else:
            embedding = None
        outputs = self.model(input_ids, embedding=embedding)
        return outputs
    
    
    def infer_batch(self, seqs: list) -> Output:
        input_ids = self.tokenizer(seqs)
        input_ids = pad_sequence(
            [torch.tensor(x) for x in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = torch.tensor(input_ids,device=self.device)
        outputs = self._infer(input_ids)
        return outputs
    
    def __call__(self, seq: Union[str, List]) -> Output:
        if isinstance(seq, str):
            return self.infer(seq)
        elif isinstance(seq, list):
            return self.infer_batch(seq)
        else:
            raise ValueError("Input must be a string or a list of strings, got {}".format(type(seq)))
        


if __name__ == "__main__":
    ckpt_path = "protein_lm/modeling/scripts/ckpt/bi_mamba-esm-ptm_token_input/best.ckpt"
    mamba = PTMMamba(ckpt_path,device='cuda:0')
    print("instantiate PTMMamba successfully!")

    def get_residue_embeddings(output, site_index):
        # Call the model to get outputs
        
        # Retrieve hidden states
        hidden_states = output.hidden_states

        residue_embedding = hidden_states[0, site_index, :]
        
        
        return residue_embedding

    df = pd.read_csv("ptmmamba_seq_index.csv")
    
    # seq_lst = ['<N-acetylmethionine>EAD<Phosphoserine>DDDMSSQSHPDGLSGRDQPVELLNPARVNHMPSTV', 
    #            'LIAEFQRQHEQLSRQHEAQLHEHIKQQQEMLAMKHQQELLEHQRKLERHRQEQELEKQHREQKLQQLKNKEKGKESAVASTEVKMKLQEFVLNKKKALAHRNLNHCISSDPRYWYGKTQHSSLDQSSPPQSGVSTSYNHPVLGMYDAKDDFP']
    
    # index_lst = [0, 10]
    
    seq_lst = df["ptmmamba_seq"]
    index_lst = df["site_index"]

    data = []

    for seq, index in zip(seq_lst, index_lst):
        # output = mamba(seq)
        # print(output.logits.shape)
        # print(output.hidden_states.shape)
        # # print(output)
        # residue_embedding = get_residue_embeddings(output, index)
        # protein_embedding = output.hidden_states.mean(dim=1)[0]  # Tensor of shape [hidden_size]
        # print(residue_embedding.shape)  # Should be (1, len(seq), 768) or similar
        # print(f"residue_embedding for index : {index}, ",residue_embedding)
        output = mamba(seq)
        
        print(f" Outputs sequence {seq}:")
        print("  Logits shape:", output.logits.shape)
        print("  Hidden states shape:", output.hidden_states.shape)
        
        # Get residue embeddings
        residue_embedding = get_residue_embeddings(output, index)
        print(f"  Residue embedding for index {index}: {residue_embedding.shape}")
        
        # Compute mean of hidden states for protein embedding (mean across sequence length)
        protein_embedding = output.hidden_states.mean(dim=1)[0]  # Tensor of shape [hidden_size]
        print(f"  Protein embedding: {protein_embedding.shape}")
        
        # Create a row dictionary with sequence, site index, and embeddings
        row = {
            "ptmmamba_seq": seq,
            "site_index": index
        }
        
        # Add residue embedding dimensions as columns
        for dim in range(residue_embedding.size(0)):
            row[f"residue_embedding_{dim}"] = residue_embedding[dim].item()
        
        # Add protein embedding dimensions as columns
        for dim in range(protein_embedding.size(0)):
            row[f"ptmmamba_protein_embedding_position_{dim}"] = protein_embedding[dim].item()
        
        data.append(row)

    df = pd.DataFrame(data)
    filename = "ptmmamba_embeddings.csv"
    df.to_csv(filename, index=False)
    print(f"Embeddings have been saved to {filename}")
    
    
    
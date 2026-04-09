import input_reader as ir
# from . import input_reader as ir

from pathlib import Path

import torch
import torch.nn
from torch.utils.data.dataloader import DataLoader

from lightning_module import LightningModule
# from .lightning_module import LightningModule

import utils
# from . import utils
# from . import input_tokenizers

_PREDICT_CACHE = {}


def _get_loaded_model(model_loc: str):
    resolved_model_loc = str(Path(model_loc).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (resolved_model_loc, device.type)

    if cache_key not in _PREDICT_CACHE:
        model_d = torch.load(resolved_model_loc, map_location=device)
        print("LOADING MODEL NOW")
        config = model_d["hyper_parameters"]["config"]
        model = LightningModule(config, 0, model_d["hyper_parameters"]["tokenizer"])
        model.load_state_dict(model_d["state_dict"])
        model.to(device)
        model.eval()
        _PREDICT_CACHE[cache_key] = {
            "device": device,
            "config": config,
            "model": model,
        }

    cached = _PREDICT_CACHE[cache_key]
    return cached["device"], cached["config"], cached["model"]


def run_predict(model_loc: str, dataset_fasta: str, output_file: str) -> None:
    """
    Runs predictions on a FASTA file using an already existing model

    Parameters
    ----------
    model : str
        The file location of an already trained model

    dataset_fasta : str
        The location of the FASTA file for which predictions are to be made. Predicted residues should be succeeded in
        the file by either a '#' or '@' symbol

    output_file : str
        The output csv file to which predictions are written
    """
    device, config, model = _get_loaded_model(model_loc)
    test_set = ir.SingleFastaDataset(dataset_loc=dataset_fasta, tokenizer=model.tokenizer)
    gpu_batch_size = utils.get_gpu_max_batchsize(config['representation'], True)
    test_loader = DataLoader(
        test_set,
        gpu_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=test_set.collate_fn,
    )

    with open(output_file, 'w') as write_to:
        print('prot_id,position,pred', file=write_to)
        with torch.no_grad():
            for batch in test_loader:
                (
                    prot_ids,
                    site_positions,
                    logit_outputs,
                    predicted_probs,
                    targets,
                ) = model.process_batch(batch)

                for id, pos, prob in zip(prot_ids, site_positions, predicted_probs):
                    print(','.join([id, str(int(pos) + 1), '{:.3f}'.format(float(prob))]), file=write_to)

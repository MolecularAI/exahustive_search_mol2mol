from functools import partial
from time import time
import argparse
import os
import yaml


from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from scipy.stats import pearsonr
import numpy as np
import torch


from lib.dataset.chem import standardize_smiles
from lib.dataset.pair_dataset import PairedDataset
from lib.model.model import LitMolformer
from lib.model.sample import sample


RDLogger.DisableLog("rdApp.*")


def load_model(config_path, checkpoint_path, vocabulary_path, device="cuda"):
    hparams = yaml.load(open(config_path), Loader=yaml.FullLoader)
    hparams["vocabulary"] = vocabulary_path
    model = LitMolformer(**hparams)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    if "with_counts" in config_path:
        model.mol_to_fingerprints = partial(AllChem.GetMorganFingerprint, radius=2)
    else:
        model.mol_to_fingerprints = partial(
            AllChem.GetMorganFingerprintAsBitVect, radius=2, nBits=1024
        )
    return model


def generate_samples(model, smiles, beam_size=1000, device="cuda"):
    smiles = standardize_smiles(smiles)
    src = model.vocabulary.encode(model.tokenizer.tokenize(smiles))
    src = torch.from_numpy(src.astype(np.int64))
    src, src_mask, _, _, _ = PairedDataset.collate_fn([(src, src, torch.ones((1, 1)))])

    tin = time()
    samples = sample(
        model,
        src[:1],
        src_mask[:1],
        decode_type="beamsearch",
        beam_size=beam_size,
        beam_search_bs=512,
        device=device,
    )
    tout = time()

    print(f"Generated {beam_size:d} samples in {tout-tin:.3f} seconds.")

    return samples


def save_results(model, samples, csv_file):
    fp = None
    new_lines = []
    for input_smi, generated_smi, nll in zip(samples[0], samples[1], samples[2]):
        if fp is None:
            mol = Chem.MolFromSmiles(input_smi)
            fp = model.mol_to_fingerprints(mol)
        try:
            g_mol = Chem.MolFromSmiles(generated_smi)
            g_fp = model.mol_to_fingerprints(g_mol)
            t = TanimotoSimilarity(fp, g_fp)
            new_lines.append((input_smi, generated_smi, nll, t, 1))

        except BaseException:
            new_lines.append((input_smi, generated_smi, nll, 0, 0))
    new_lines = np.array(new_lines)

    valid = new_lines[:, 4].astype(bool)
    nlls = new_lines[valid, 2].astype(np.float32)
    tans = new_lines[valid, 3].astype(np.float32)

    with open(csv_file, "w") as wfile:
        header = "input_smiles,generated_smiles,nll,tanimoto,is_valid"
        new_lines = [header] + [",".join(line) for line in new_lines]
        wfile.write("\n".join(new_lines))

    print(f"Valid molecules = {valid.sum():d} / {len(valid):d}")
    print(f"Pearson R = {pearsonr(-nlls, tans).statistic:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample neighborhood")
    # Define three optional arguments
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--input-smiles", required=True, help="Input smiles")
    parser.add_argument("--samples", required=True, default=1000, help="Number of samples per smiles", type=int)
    parser.add_argument("--result-path", required=True, help="Result path (csv)")

    args = parser.parse_args()
    config_path = os.path.join(args.model, "config.yml")
    checkpoint_path = os.path.join(args.model, "weights.ckpt")
    input_smiles = args.input_smiles
    result_path = args.result_path
    vocabulary_path = "paper_checkpoints/vocabulary.pkl"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    samples = [[], [], []]
    for smi in open(input_smiles).readlines():
        smi = smi.strip()
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Cannot understand SMILES: {smi}")
                continue
        except BaseException:
            print(f"Cannot understand SMILES: {smi}")
            continue
        model = load_model(config_path, checkpoint_path, vocabulary_path, device)
        _samples = generate_samples(model, smi, beam_size=args.samples, device=device)
        samples[0] += _samples[0]
        samples[1] += _samples[1]
        samples[2] += _samples[2]
    save_results(model, samples, result_path)

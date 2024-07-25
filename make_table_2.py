from functools import partial
from glob import glob
from time import time
import os
import pickle
import subprocess
import sys
import yaml
import json

from natsort import natsorted
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.stats import pearsonr, kendalltau
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from lib.dataset.chem import standardize_smiles
from lib.dataset.pair_dataset import FastPairedDataset
from lib.model.model import LitMolformer
from lib.model.sample import sample as sample_from_model


RDLogger.DisableLog("rdApp.*")


class Similarity:
    def __init__(self, similarity_type=None):
        assert similarity_type in ["ae", "tan"]
        self.similarity_type = similarity_type

    def compute_similarity(self, source, targets):
        if self.similarity_type == "ae":
            res = subprocess.run(
                ["utils/compute_similarity_gan_drug_generator.sh"] + [source] + targets,
                capture_output=True,
            )
            sims = [float(s) for s in res.stdout.decode().strip().split()]
        else:
            source_mol = Chem.MolFromSmiles(source)
            target_mols = [Chem.MolFromSmiles(target) for target in targets]
            source_fp = AllChem.GetMorganFingerprint(source_mol, radius=2)
            target_fps = [
                AllChem.GetMorganFingerprint(target_mol, radius=2)
                for target_mol in target_mols
                if target_mol is not None
            ]
            sims = BulkTanimotoSimilarity(source_fp, target_fps)

        return np.array(sims, dtype=np.float32)


def load_model(config_path, checkpoint_path, vocabulary_path, device="cuda"):
    hparams = yaml.load(open(config_path), Loader=yaml.FullLoader)
    hparams["vocabulary"] = vocabulary_path
    model = LitMolformer(**hparams)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    return model


def generate_samples(model, smiles, beam_size=1000):
    # smiles = standardize_smiles(smiles)
    src = model.vocabulary.encode(model.tokenizer.tokenize(smiles))
    src = torch.from_numpy(src.astype(np.int64))
    src, src_mask, _, _, _ = FastPairedDataset.collate_fn(
        [(src, src, torch.ones((1, 1)))]
    )

    samples = sample_from_model(
        model,
        src[:1],
        src_mask[:1],
        decode_type="beamsearch",
        beam_size=beam_size,
        beam_search_bs=512,
    )

    return samples


def aggregate_same_similarity(sim, ll):
    sim_ll = {}
    for s, l in zip(sim, ll):
        if s not in sim_ll:
            sim_ll[s] = []
        sim_ll[s].append(l)

    for s in sim_ll:
        sim_ll[s] = np.mean(sim_ll[s])

    sim = np.array([x[0] for x in list(sim_ll.items())])
    ll = np.array([x[1] for x in list(sim_ll.items())])
    idx = np.argsort(sim)[::-1]
    sim = sim[idx]
    ll = ll[idx]
    return sim, ll


def compute_stats_v2(
    model, similarity, smi, samples=None, similarities=None, beam_size=100
):
    if similarities is None:
        samples = generate_samples(model, smi, beam_size)

    # filter out invalid_smiles
    valid_samples, nlls = [], []
    for sample, nll in zip(samples[1][:100], samples[2][:100]):
        try:
            if Chem.MolFromSmiles(sample) is not None:
                valid_samples.append(sample)
                nlls.append(nll)
        except BaseException:
            pass

    if len(valid_samples) == 0:
        return None

    nlls = np.array(nlls, dtype=np.float32)
    lls = -nlls
    if similarities is None:
        tans = similarity.compute_similarity(smi, valid_samples)
    else:
        tans = similarities[:100]
    if len(tans) == 0:
        return None

    idx = np.argsort(lls)[::-1]
    rank_lls = lls[idx[:10]]
    rank_sims = tans[idx[:10]]
    rank_10 = kendalltau(rank_sims, rank_lls).statistic

    agg_tans, agg_lls = aggregate_same_similarity(tans, lls)

    try:
        correlation = pearsonr(agg_tans, agg_lls).statistic
    except BaseException:
        return None

    unique_samples = []
    unique_canonicalized = []
    unique_no_stereo = []

    for vs in valid_samples:
        if vs not in unique_samples:
            unique_samples.append(vs)
        std_vs = standardize_smiles(vs)
        if std_vs is not None and std_vs not in unique_canonicalized:
            unique_canonicalized.append(std_vs)
            mol = Chem.MolFromSmiles(std_vs)
            Chem.RemoveStereochemistry(mol)
            std_no_stereo_vs = standardize_smiles(Chem.MolToSmiles(mol))
            if (
                std_no_stereo_vs is not None
                and std_no_stereo_vs not in unique_no_stereo
            ):
                unique_no_stereo.append(std_no_stereo_vs)

    mol = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(mol)
    smi_no_stereo = standardize_smiles(Chem.MolToSmiles(mol))

    return {
        "n_samples": beam_size,
        "valid": len(valid_samples),
        "unique": len(unique_samples),
        "unique_canonicalized": len(unique_canonicalized),
        "unique_no_stereo": len(unique_no_stereo),
        "top": valid_samples[0] == smi,
        "top_canonicalized": unique_canonicalized[0] == smi,
        "top_no_stereo": unique_no_stereo[0] == smi_no_stereo,
        "rank_10": rank_10,
        "correlation": correlation,
    }


if __name__ == "__main__":
    # PARAMS
    for model_name in [
        "rnn_ae_wr_chembl",
        "rnn_ae_nr_chembl",
        "tra_ae_wr_chembl",
        "tra_ae_nr_chembl",
    ]:
        config_path = f"results/{model_name}/config.yml"
        checkpoint_path = f"results/{model_name}/chkpts/epoch=29-*.ckpt"
        checkpoint_path = natsorted(glob(checkpoint_path))[-1]
        vocabulary_path = "cache/chembl/vocabulary.pkl"
        device = "cuda"
        model = load_model(config_path, checkpoint_path, vocabulary_path, device)
        similarity = Similarity(model_name.split("_")[1])
        print(checkpoint_path)
        test = pickle.load(open("data/chembl/test.pkl", "rb"))
        smiles = sorted(list(set([x[0] for x in test])))
        smiles = [
            smiles[i] for i in np.arange(0, len(smiles), len(smiles) // 1000)[:1000]
        ]
        all_res = []

        for smi in tqdm(smiles):
            res = compute_stats_v2(model, similarity, smi, beam_size=100)
            if res:
                all_res.append(res)

        all_res_json = {
            "valid": f'{np.mean([k["valid"]/k["n_samples"] for k in all_res]):.2f}',
            "unique": f'{np.mean([k["unique"]/k["valid"] for k in all_res]):.2f}',
            "unique_canonicalized": f'{np.mean([k["unique_canonicalized"]/k["valid"] for k in all_res]):.2f}',
            "unique_no_stereo": f'{np.mean([k["unique_no_stereo"]/k["valid"] for k in all_res]):.2f}',
            "top": f'{np.mean([k["top"] for k in all_res]):.2f}',
            "top_canonicalized": f'{np.mean([k["top_canonicalized"] for k in all_res]):.2f}',
            "top_no_stereo": f'{np.mean([k["top_no_stereo"] for k in all_res]):.2f}',
            "rank_10": f'{np.nanmean([k["rank_10"] for k in all_res]):.2f}±{np.nanstd([k["rank_10"] for k in all_res]):.2f}',
            "correlation": f'{np.nanmean([k["correlation"] for k in all_res]):.2f}±{np.nanstd([k["correlation"] for k in all_res]):.2f}',
        }
        with open(f"results/stats/{model_name}.json", "w") as out_file:
            json.dump(all_res_json, out_file, indent=4)

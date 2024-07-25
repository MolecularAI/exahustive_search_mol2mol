from functools import partial
from time import time
import sys
import yaml


from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from scipy.stats import pearsonr, kendalltau
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


def generate_samples(model, smiles, beam_size=1000):
    smiles = standardize_smiles(smiles)
    src = model.vocabulary.encode(model.tokenizer.tokenize(smiles))
    src = torch.from_numpy(src.astype(np.int64))
    src, src_mask, _, _, _ = PairedDataset.collate_fn([(src, src, torch.ones((1, 1)))])

    # tin = time()
    samples = sample(
        model,
        src[:1],
        src_mask[:1],
        decode_type="beamsearch",
        beam_size=beam_size,
        beam_search_bs=512,
    )
    # tout = time()

    # print(f"Generated {beam_size:d} samples in {tout-tin:.3f} seconds.")

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


def compute_stats(model, samples):
    fp = None
    valid = 0
    unique = set()
    unique_canonicalized = set()
    unique_no_stereo = set()

    top = 0
    top_canonicalized = 0
    top_no_stereo = 0

    nlls = []
    tanimotos = []
    first = True
    for input_smi, generated_smi, nll in zip(
        samples[0][:1000], samples[1][:1000], samples[2][:1000]
    ):
        if fp is None:
            mol = Chem.MolFromSmiles(input_smi)
            fp = model.mol_to_fingerprints(mol)

            std_input_smi = standardize_smiles(input_smi)
            Chem.RemoveStereochemistry(mol)
            input_smi_no_stereo = Chem.MolToSmiles(mol)
            input_smi_no_stereo = standardize_smiles(input_smi_no_stereo)
        try:
            g_mol = Chem.MolFromSmiles(generated_smi)

            g_fp = model.mol_to_fingerprints(g_mol)
            t = TanimotoSimilarity(fp, g_fp)
            Chem.RemoveStereochemistry(g_mol)
            no_stero_smi = Chem.MolToSmiles(g_mol)

            tanimotos.append(t)
            nlls.append(nll)
            unique.add(generated_smi)
            std_generated_smi = standardize_smiles(generated_smi)
            unique_canonicalized.add(std_generated_smi)
            no_stero_smi = standardize_smiles(no_stero_smi)
            unique_no_stereo.add(no_stero_smi)

            if first:
                top += input_smi == generated_smi
                top_canonicalized += std_input_smi == std_generated_smi
                top_no_stereo += input_smi_no_stereo == no_stero_smi
                first = False

            valid += 1
        except BaseException:
            pass

    nlls = np.array(nlls)
    tans = np.array(tanimotos)
    lls = -nlls

    idx = np.argsort(lls)[::-1]
    rank_lls = lls[idx[:10]]
    rank_sims = tans[idx[:10]]

    rank_10 = kendalltau(rank_sims, rank_lls).statistic
    tans, lls = aggregate_same_similarity(tans, lls)
    correlation = pearsonr(tans, lls).statistic

    return {
        "valid": valid,
        "unique": len(unique),
        "unique_canonicalized": len(unique_canonicalized),
        "unique_no_stereo": len(unique_no_stereo),
        "top": top,
        "top_canonicalized": top_canonicalized,
        "top_no_stereo": top_no_stereo,
        "rank_10": rank_10,
        "correlation": correlation,
    }


if __name__ == "__main__":
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    vocabulary_path = sys.argv[3]

    device = "cuda"

    all_res = {}

    model = load_model(config_path, checkpoint_path, vocabulary_path, device)
    with open("test_data/TTD/ttd.smi") as smiles:
        for smi_id, smi in enumerate(smiles):
            smi = smi.strip()
            smi = standardize_smiles(smi)
            samples = generate_samples(model, smi, beam_size=1000)
            results = compute_stats(model, samples)
            for k in results:
                if k not in all_res:
                    all_res[k] = []
                all_res[k].append(results[k])

    valid_den = len(all_res[k]) * 1000
    unique_den = sum(all_res["valid"])
    top_den = len(all_res[k])

    for k in all_res:
        if k == "valid":
            print(k, sum(all_res[k]) / valid_den)
        elif "unique" in k:
            print(k, sum(all_res[k]) / unique_den)
        elif "top" in k:
            print(k, sum(all_res[k]) / top_den)
        elif "rank" in k:
            print(k, np.nanmean(all_res[k]), "+/-", np.nanstd(all_res[k]))
        elif k == "correlation":
            print(k, np.nanmean(all_res[k]), "+/-", np.nanstd(all_res[k]))

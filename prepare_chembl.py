# https://zenodo.org/records/6319821
import csv
import os
import pickle
import sys
import yaml

from tqdm import tqdm
import numpy as np

from lib.dataset.chem import standardize_smiles
from lib.dataset.utils import build_vocabulary, save_vocabulary
from lib.dataset.utils import load_vocabulary
from lib.dataset.utils import smiles_to_tensor
from lib.dataset.vocabulary import SMILESTokenizer


def prepare_pairs(smiles_path, verbose=False):
    all_smiles = set()
    cache = {}
    sets = {}
    for _set in ["train", "validation", "test"]:
        pairs = []
        with open(os.path.join(smiles_path, f"{_set}.csv"), "r") as csvfile:
            # reader_variable = csv.reader(csvfile, delimiter=",")
            data = csv.DictReader(csvfile)
            sets[_set] = set()
            for row in tqdm(data):
                try:
                    src = row["Source_Mol"]
                    trg = row["Target_Mol"]
                    if src not in cache:
                        src_new = standardize_smiles(src)
                        # mol = Chem.MolFromSmiles(src_new)
                        # fp = AllChem.GetMorganFingerprint(mol, 2)
                        fp = None
                        cache[src] = (src_new, fp)
                        cache[src_new] = (src_new, fp)
                        sets[_set].add(src_new)
                        all_smiles.add(src_new)
                    if trg not in cache:
                        trg_new = standardize_smiles(trg)
                        # mol = Chem.MolFromSmiles(trg_new)
                        # fp = AllChem.GetMorganFingerprint(mol, 2)
                        fp = None
                        cache[trg] = (trg_new, fp)
                        cache[trg_new] = (trg_new, fp)
                        all_smiles.add(trg_new)
                    tanimoto = float(row["Tanimoto"])
                    if tanimoto < 0.5:
                        continue
                    pairs.append((cache[src][0], cache[trg][0], tanimoto))
                except BaseException:
                    pass
        with open(os.path.join(smiles_path, f"{_set}.pkl"), "wb") as fobj:
            pickle.dump(pairs, fobj)
    if verbose:
        print(len(sets["train"] & sets["validation"]))
        print(len(sets["train"] & sets["test"]))
        print(len(sets["validation"] & sets["test"]))
    return all_smiles


if __name__ == "__main__":
    hparams = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    smiles_path = os.path.dirname(hparams["train_smiles_path"])
    if not os.path.exists(hparams["vocabulary"]):
        all_smiles = list(prepare_pairs(smiles_path))
        std_smiles = list(all_smiles)
        print(f"# SMILES: {len(std_smiles):d}")
        std_smiles = [range(len(std_smiles)), std_smiles]
        vocabulary, tokenizer = build_vocabulary(std_smiles)
        save_vocabulary(hparams["vocabulary"], vocabulary)
    vocabulary = load_vocabulary(hparams["vocabulary"])
    tokenizer = SMILESTokenizer()
    unk_token = vocabulary.tokens()[vocabulary.unk_token]
    
    for _set in ["train", "validation", "test"]:
        pairs = pickle.load(open(os.path.join(smiles_path, f"{_set}.pkl"), "rb"))
        srcs = []
        trgs = []
        tans = []
        v = {}
        for p in tqdm(pairs):
            if p[0] not in v:
                v[p[0]] = smiles_to_tensor(p[0], vocabulary, tokenizer, unk_token)
            if p[1] not in v:
                v[p[1]] = smiles_to_tensor(p[1], vocabulary, tokenizer, unk_token)
            enc_src = v[p[0]]
            enc_trg = v[p[1]]
            tans.append(p[2])
            srcs.append(enc_src)
            trgs.append(enc_trg)
        srcs = np.array(srcs, dtype=object)
        trgs = np.array(trgs, dtype=object)
        tans = np.array(tans, dtype=np.float32)
        np.save(
            os.path.join(smiles_path, f"{_set}.npy"),
            {"src": srcs, "trg": trgs, "sims": tans},
        )

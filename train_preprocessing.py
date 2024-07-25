"""
Preprocessing
"""
from glob import glob
import os
import pickle
import sys
import yaml

from tqdm import tqdm
import numpy as np

from lib.dataset.io import read_index_binary_file_64bits
from lib.dataset.io import write_binary_encoded_smiles
from lib.dataset.utils import build_vocabulary, save_vocabulary


if __name__ == "__main__":
    hparams = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    smiles_path = hparams["smiles_path"]
    smiles = pickle.load(open(smiles_path, "rb"))
    std_smiles = [std_smi for _, _, std_smi in smiles]
    vocabulary, tokenizer = build_vocabulary(std_smiles)
    save_vocabulary(hparams["vocabulary"], vocabulary)

    index = {}
    for fname in tqdm(glob(os.path.join(hparams["pairs_path"], "*.index.dat")), ascii=True):
        index.update(read_index_binary_file_64bits(fname))

    index_np = -np.ones((max(index.keys()) + 1, 4), dtype=np.int64)
    for k in index:
        k = int(k)
        fname, pos = index[k]
        index_np[k] = [int(x) for x in fname.split("_")] + [pos]

    np.save(hparams["smiles_id_to_data"], index_np)

    # create cache for encoded smiles
    # Takes up to 1,5 hours
    write_binary_encoded_smiles(
        smiles,
        hparams["encoded_smiles"],
        hparams["smiles_id_to_encoded_smiles"],
        tokenizer,
        vocabulary,
        hparams["max_sequence_length"],
        use_pbar=True,
    )

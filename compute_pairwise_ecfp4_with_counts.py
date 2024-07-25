from concurrent import futures
from glob import glob
import os
import pickle
import sys

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm
import numpy as np

from lib.dataset.io import read_binary_file, read_index_binary_file_64bits
from lib.dataset.io import write_binary_file


RDLogger.DisableLog("rdApp.*")


class FingerprintCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size

    def add(self, k, smiles):
        if k in self.cache:
            return self.cache[k]
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 2)

        if len(self.cache) == self.max_size:
            self.cache.popitem()
        self.cache[k] = fp
        return fp

    def __getitem__(self, k):
        return self.cache[k]

    def __contains__(self, k):
        return k in self.cache


def _compute_ecfp4_wc(args):
    smiles = args["smiles"]
    dat_paths = args["dat_paths"]
    output_path = args["output_path"]
    progress_bar = args["progress_bar"]
    pid = args["pid"]

    fp_cache = FingerprintCache()

    if progress_bar:
        pbar = tqdm(dat_paths, ascii=True)
    else:
        pbar = dat_paths

    for dat_path in dat_paths:
        idx = read_index_binary_file_64bits(dat_path)
        data = {}
        for ik, k in enumerate(idx):
            fname, pos = idx[k]
            binary_file_path = os.path.join(os.path.dirname(dat_path), fname + ".dat")
            pubchem_id, pubchem_id_neighs, _ = read_binary_file(binary_file_path, pos)
            fp = fp_cache.add(pubchem_id, smiles[pubchem_id])
            fps = []
            for pin in pubchem_id_neighs:
                fps.append(fp_cache.add(pin, smiles[pin]))
            if len(fps) > 0:
                t = np.array(BulkTanimotoSimilarity(fp, fps)).astype(np.float32)
                tidx = t >= 0.5
                if tidx.sum() > 0:
                    T = t[tidx]
                    V = pubchem_id_neighs[tidx]
                    data[pubchem_id] = [V, T]
            if progress_bar:
                pbar.set_description(f"File={fname} - Processing={ik+1:d}/{len(idx):d}")
        if len(data) > 0:
            write_binary_file(data, os.path.join(output_path, fname + ".dat"))
    return {"pid": pid, "status": True}


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    ecfp4_path = sys.argv[2]
    opath = sys.argv[3]
    os.makedirs(opath, exist_ok=True)
    n_proc = int(sys.argv[4])

    smiles = pickle.load(open(dataset_path, "rb"))
    smiles = {pubchem_id: std_smi for _, pubchem_id, std_smi in smiles}
    dat_paths = glob(os.path.join(ecfp4_path, "*.index.dat"))

    n_proc = min(n_proc, len(dat_paths))
    n_proc = min(n_proc, os.cpu_count())

    split_dat_paths = np.array_split(dat_paths, n_proc)
    data_pool = []
    for pid, dp in enumerate(split_dat_paths):
        data_pool.append(
            {
                "smiles": smiles,
                "dat_paths": dp,
                "output_path": opath,
                "progress_bar": pid == 0,
                "pid": pid,
            }
        )
    pool = futures.ProcessPoolExecutor(max_workers=n_proc)
    results = list(pool.map(_compute_ecfp4_wc, data_pool))
    results = sorted(results, key=lambda x: x["pid"])
    print(results)

from concurrent import futures
import os
import pickle
import shutil
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np


def _compute_ecfp4(args):
    smiles = args["smiles"]
    output_path = args["output_path"]
    progress_bar = args["progress_bar"]
    pid = args["pid"]

    X = np.zeros((len(smiles), 128), dtype=np.uint8)
    L = np.zeros((len(smiles),), dtype=np.float32)
    N = np.zeros((len(smiles),), dtype=np.int32)

    if progress_bar:
        pbar = tqdm(smiles, ascii=True)
    else:
        pbar = smiles
    i = 0
    for _, smi_id, std_smi in pbar:
        try:
            mol = Chem.MolFromSmiles(std_smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            n_substructures = 1.0 * sum(fp.ToList())
            fp_bits = np.packbits(fp)
            X[i] = fp_bits
            L[i] = n_substructures
            N[i] = int(smi_id)
            i = i + 1
        except BaseException:
            pass
    if i > 0:
        np.save(os.path.join(output_path, f"x.{pid:d}.npy"), X[:i])
        np.save(os.path.join(output_path, f"l.{pid:d}.npy"), L[:i])
        np.save(os.path.join(output_path, f"n.{pid:d}.npy"), N[:i])
        op = os.path.join(output_path, "{}" + f".{pid:d}.npy")
    else:
        op = None

    return {"pid": pid, "output_path": op}


if __name__ == "__main__":
    path = sys.argv[1]
    opath = sys.argv[2]
    n_proc = int(sys.argv[3])
    assert not os.path.exists(os.path.join(opath, "X.npy"))
    assert not os.path.exists(os.path.join(opath, "L.npy"))
    assert not os.path.exists(os.path.join(opath, "N.npy"))
    os.makedirs(os.path.join(opath, "tmp"))

    smiles = pickle.load(open(path, "rb"))
    n_proc = min(n_proc, len(smiles))
    n_proc = min(n_proc, os.cpu_count())

    split_smiles = np.array_split(smiles, n_proc)
    data_pool = []
    for pid, smis in enumerate(split_smiles):
        data_pool.append(
            {
                "smiles": smis,
                "output_path": os.path.join(opath, "tmp"),
                "progress_bar": pid == 0,
                "pid": pid,
            }
        )
    pool = futures.ProcessPoolExecutor(max_workers=n_proc)
    results = list(pool.map(_compute_ecfp4, data_pool))
    results = sorted(results, key=lambda x: x["pid"])

    n_smiles = 0
    for res in results:
        if res["output_path"] is None:
            continue
        else:
            n_smiles += len(np.load(res["output_path"].format("l")))

    X = np.zeros((n_smiles, 128), dtype=np.uint8)
    L = np.zeros((n_smiles,), dtype=np.float32)
    N = np.zeros((n_smiles,), dtype=np.int32)
    offset = 0
    for res in results:
        if res["output_path"] is None:
            continue
        else:
            _x = np.load(res["output_path"].format("x"))
            _l = np.load(res["output_path"].format("l"))
            _n = np.load(res["output_path"].format("n"))
            X[offset:offset + len(_x)] = _x
            L[offset:offset + len(_l)] = _l
            N[offset:offset + len(_n)] = _n
            offset = offset + len(_l)
    np.save(os.path.join(opath, "X.npy"), X)
    np.save(os.path.join(opath, "L.npy"), L)
    np.save(os.path.join(opath, "N.npy"), N)

    shutil.rmtree(os.path.join(opath, "tmp"))

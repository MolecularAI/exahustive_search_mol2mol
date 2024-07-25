from concurrent import futures
from glob import glob
import gzip
import hashlib
import os
import pickle
import shutil
import sys

from natsort import natsorted
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

from lib.dataset.chem import standardize_smiles
from lib.dataset.chem import remove_isotopes

RDLogger.DisableLog("rdApp.*")


def check_path(path):
    md5sum1 = open(path + ".md5").readlines()[0].split()[0].strip()
    md5sum2 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if md5sum1 != md5sum2:
        raise ValueError(f"{os.path.basename(path)} md5sum mismatches")


def _process_sdf(args):
    paths = args["paths"]
    output_path = args["output_path"]
    progress_bar = args["progress_bar"]
    pid = args["pid"]
    output_paths = []
    all_smiles = set()
    if progress_bar:
        pbar = tqdm(paths, ascii=True)
    else:
        pbar = paths
    for path_id, path in enumerate(pbar):
        smiles = []
        try:
            check_path(path)
        except ValueError:
            continue
        fname = os.path.basename(path)
        fname = fname.replace(".", "_")
        min_id, max_id = fname.split("_")[-4:-2]
        min_id = int(min_id)
        max_id = int(max_id)
        with gzip.open(path) as fobj:
            with Chem.ForwardSDMolSupplier(fobj) as mols:
                for i, mol in enumerate(mols):
                    if mol is None:
                        continue
                    try:
                        smi = mol.GetProp("PUBCHEM_OPENEYE_CAN_SMILES")
                        std_smi = standardize_smiles(smi)
                        free_iso_smi = remove_isotopes(mol)
                        contain_iso = std_smi != free_iso_smi
                        if (
                            (std_smi is not None)
                            and (std_smi not in all_smiles)
                            and (not contain_iso)
                        ):
                            smi_cid = int(mol.GetProp("PUBCHEM_COMPOUND_CID"))
                            # hac = int(mol.GetProp("PUBCHEM_HEAVY_ATOM_COUNT"))
                            # isotope = mol.GetProp("PUBCHEM_ISOTOPIC_ATOM_COUNT")
                            # isotope = int(int(isotope) > 0)
                            smiles.append([smi, smi_cid, std_smi])
                            all_smiles.add(std_smi)
                    except BaseException:
                        # We do not care if some of the smiles
                        # raise any type of exception
                        pass
                    if progress_bar:
                        pbar.set_description(
                            f"File: {fname} Processed: {i+1:d}/{max_id-min_id+1:d}"
                        )

        try:
            if len(smiles) > 0:
                with open(output_path.format(path_id), "wb") as pkl:
                    pickle.dump(smiles, pkl)
                    output_paths.append(output_path.format(path_id))

        except BaseException:
            pass

    return {"pid": pid, "output_paths": output_paths}


if __name__ == "__main__":
    path = sys.argv[1]
    opath = sys.argv[2]
    n_proc = int(sys.argv[3])

    assert not os.path.exists(os.path.join(opath, "pubchem.pkl"))
    os.makedirs(os.path.join(opath, "tmp"))

    all_paths = natsorted(glob(os.path.join(path, "Compound_*.gz")))
    n_proc = min(n_proc, len(all_paths))
    n_proc = min(n_proc, os.cpu_count())

    split_paths = np.array_split(all_paths, n_proc)
    data_pool = []
    for pid, paths in enumerate(split_paths):
        data_pool.append(
            {
                "paths": paths,
                "output_path": os.path.join(opath, f"tmp/{pid:d}" + "_{}.pkl"),
                "progress_bar": pid == 0,
                "pid": pid,
            }
        )
    pool = futures.ProcessPoolExecutor(max_workers=n_proc)
    results = list(pool.map(_process_sdf, data_pool))
    results = sorted(results, key=lambda x: x["pid"])

    smiles = []
    all_smiles = set()
    for res in results:
        if len(res["output_paths"]) == 0:
            continue
        for op in res["output_paths"]:
            data = pickle.load(open(op, "rb"))
            for pubchem_smi, pubchem_id, std_smi in data:
                if std_smi not in all_smiles:
                    smiles.append((pubchem_smi, pubchem_id, std_smi))
                    all_smiles.add(std_smi)
    with open(os.path.join(opath, "pubchem.pkl"), "wb") as pkl:
        pickle.dump(smiles, pkl)

    shutil.rmtree(os.path.join(opath, "tmp"))

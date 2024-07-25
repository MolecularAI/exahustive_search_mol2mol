import os
import sys

from tqdm import tqdm
import cupy as cp
import numpy as np

from lib.dataset.io import write_binary_file


def split_idx(n, processes, offset=0):
    process_idx = []
    n_pairs = n * (n - 1) // 2

    idx = np.arange(n)[1:][::-1]
    cidx = np.cumsum(idx)

    remaining_pairs = (n - offset) * (n - offset - 1) // 2
    offset_pairs = n_pairs - remaining_pairs

    batch_size = int(np.ceil(remaining_pairs // processes))

    for i in range(processes):
        g = (cidx <= ((i + 1) * batch_size + offset_pairs)) & (
            cidx >= (i * batch_size + offset_pairs)
        )
        g = np.where(g)[0]
        process_idx.append((max(g[0], offset), g[-1] + 1))
    process_idx[-1] = (process_idx[-1][0], max(process_idx[-1][1], n - 1))
    for p1, p2 in zip(process_idx[:-1], process_idx[1:]):
        assert p1[1] == p2[0]

    return process_idx


def compute_pairwise_tanimoto(args):
    X = args["X"]
    L = args["L"]
    N = args["N"]

    left_i = args["left_i"]
    right_i = args["right_i"]
    gpu_id = args["gpu_id"]
    path = args["path"]

    output_prefix = os.path.join(path, "{:02d}_{:03d}_{:05d}.dat")
    # cp.cuda.Device(gpu_id).use()
    # Move to GPU
    # It requires a GPU with at least 32GB
    # to store the full dataset
    # (> 100,000,000 molecules) in memory

    Xc = cp.array(X[left_i:])
    Lc = cp.array(L[left_i:])
    Nc = cp.array(N[left_i:])

    bitmap = np.array([i for i in range(256)]).astype(np.uint8)[:, None]
    nbits = np.unpackbits(bitmap, axis=-1).sum(axis=-1).astype(np.uint8)
    nbits = cp.array(nbits)

    inner_batch_size = 10000000
    s = cp.zeros(len(X), dtype=cp.float32)
    data = {}
    size = 0
    chunk_id = 0
    for i in tqdm(range(left_i, right_i), ascii=True):
        I = cp.bitwise_and(Xc[i - left_i], Xc[i - left_i + 1:])
        lI = len(I)
        n_batches = int(np.ceil(lI / inner_batch_size))

        for b in range(n_batches):
            li, ri = b * inner_batch_size, min((b + 1) * inner_batch_size, lI)
            s[li:ri] = nbits[I[li:ri]].sum(axis=-1)
        del I
        tanimoto = s[:lI] / (Lc[i - left_i] + Lc[i - left_i + 1:] - s[:lI])
        tidx = tanimoto >= 0.5
        right_ids = Nc[i - left_i + 1:][tidx]

        if len(right_ids) > 0:
            V, T = right_ids.get(), tanimoto[tidx].get()
            data[N[i - left_i]] = [V, T]
            size += 4 + 4 + 4 * len(V) + 4 * len(T)

        if size >= 100 * 2**20:  # 100 MB:
            write_binary_file(data, output_prefix.format(0, gpu_id, chunk_id))
            data = {}
            size = 0
            chunk_id = chunk_id + 1
        del tanimoto
        del tidx
        del right_ids

    if len(data):
        write_binary_file(data, output_prefix.format(0, gpu_id, chunk_id))

    return True


if __name__ == "__main__":
    path = sys.argv[1]
    opath = sys.argv[2]
    os.makedirs(opath, exist_ok=True)
    gpu_id = int(sys.argv[3])
    all_gpus = int(sys.argv[4])
    # X contains the ecfp 1024 bit for each smiles.
    # They are represented in uint8 for saving space
    # and for efficiently compute the bitwise_and operation

    # L contains the number of 1s for each fingerprints
    # It is used to compute the tanimoto similarity

    # N contains the ids of the molecules

    X = np.load(os.path.join(path, "X.npy"))
    L = np.load(os.path.join(path, "L.npy"))
    N = np.load(os.path.join(path, "N.npy"))

    assert X.shape[-1] == 128

    idx = split_idx(len(X), processes=all_gpus)
    left_i, right_i = idx[gpu_id]

    payload = {
        "X": X,
        "L": L,
        "N": N,
        "left_i": left_i,
        "right_i": right_i,
        "gpu_id": gpu_id,
        "path": opath,
    }

    compute_pairwise_tanimoto(payload)

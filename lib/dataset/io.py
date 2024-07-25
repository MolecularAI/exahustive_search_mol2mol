import os
import struct

import numpy as np


def read_binary_file(fname, pos):
    """
    data = { k : [(v1, t1), ..., (vn, tn)]}
    """
    try:
        with open(fname, "rb") as fbin:
            fbin.seek(pos)
            key = struct.unpack("I", fbin.read(4))[0]
            length = struct.unpack("I", fbin.read(4))[0]
            values = np.zeros(length, dtype=np.int32)
            tanimotos = np.zeros(length, dtype=np.float32)
            values[:] = struct.unpack("I" * length, fbin.read(4 * length))
            tanimotos[:] = struct.unpack("f" * length, fbin.read(4 * length))
    except BaseException:
        raise ValueError(f"{fname} {pos}")
    return key, values, tanimotos


def read_index_binary_file_64bits(fname):
    """
    data = [(k1, pos1), ..., (kn, posn)]}
    """
    keys = {}
    data_prefix = os.path.splitext(os.path.splitext(os.path.basename(fname))[0])[0]

    with open(fname, "rb") as ibin:
        ibin.seek(0, 2)
        eof = ibin.tell()
        ibin.seek(0, 0)
        for _ in range(0, eof, 12):
            key = struct.unpack("I", ibin.read(4))[0]
            pos = struct.unpack("Q", ibin.read(8))[0]
            keys[key] = (data_prefix, pos)

    return keys


def write_binary_file(data, fname):
    """
    data = { k : [values, tanimotos] }
    """

    # CHECK
    for key in data:
        values, tanimotos = data[key]
        assert len(values) == len(tanimotos)

    idx_fname = os.path.splitext(fname)[0] + ".index.dat"
    offset = 0
    with open(fname, "wb") as fbin, open(idx_fname, "wb") as ibin:
        for key in data:
            values, tanimotos = data[key]
            key = int(key)
            if len(values) == 0:
                continue
            fbin.write(struct.pack("I", key))
            fbin.write(struct.pack("I", len(values)))
            fbin.write(struct.pack("I" * len(values), *values))
            fbin.write(struct.pack("f" * len(values), *tanimotos))
            ibin.write(struct.pack("I", key))
            ibin.write(struct.pack("Q", offset))
            offset += 4 + 4 + 4 * len(values) + 4 * len(values)


def read_binary_encoded_smiles(fname, pos):
    with open(fname, "rb") as fbin:
        fbin.seek(pos)
        key = struct.unpack("Q", fbin.read(8))[0]
        length = struct.unpack("H", fbin.read(2))[0]
        enc_smi = np.zeros((length,), dtype=np.uint16)
        enc_smi[:] = struct.unpack("H" * length, fbin.read(2 * length))
    return key, enc_smi


def write_binary_encoded_smiles(
    smiles, fname, index_fname, tokenizer, vocabulary, max_sequence_length, use_pbar=True
):
    from tqdm import tqdm
    from .utils import smiles_to_tensor

    offsets = {}
    offset = 0

    if use_pbar:
        pbar = tqdm(smiles, total=len(smiles), ascii=True)
    else:
        pbar = smiles

    unk_token = vocabulary.tokens()[vocabulary.unk_token]

    with open(fname, "wb") as fbin:
        for i, (_, smi_id, std_smi) in enumerate(pbar):
            smi_tensor = smiles_to_tensor(std_smi, vocabulary, tokenizer, unk_token)

            if len(smi_tensor) > max_sequence_length:
                continue

            smi_tensor = smi_tensor.astype(np.uint16)
            fbin.write(struct.pack("Q", smi_id))
            fbin.write(struct.pack("H", len(smi_tensor)))
            fbin.write(struct.pack("H" * len(smi_tensor), *smi_tensor))

            offsets[smi_id] = offset
            offset += 8 + 2 + 2 * len(smi_tensor)

    max_key = -1
    for key in offsets:
        if key > max_key:
            max_key = key

    c = -np.ones(max_key + 1, dtype=np.int64)
    for key in offsets:
        c[key] = offsets[key]
    np.save(index_fname, c)

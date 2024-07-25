from typing import List, Tuple
import os

from torch import Tensor
from torch.utils import data as tud

import numpy as np
import torch

from .io import (
    read_binary_file,
    read_binary_encoded_smiles,
)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def _mask_batch(encoded_seqs: List) -> Tuple[Tensor, Tensor]:
    """Pads a batch.

    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded and masked
    """

    # maximum length of input sequences
    max_length_source = max([seq.size(0) for seq in encoded_seqs])

    # padded source sequences with zeroes
    collated_arr_seq = torch.zeros(
        len(encoded_seqs), max_length_source, dtype=torch.long
    )

    for i, seq in enumerate(encoded_seqs):
        collated_arr_seq[i, : len(seq)] = seq

    # mask of source seqs
    seq_mask = (collated_arr_seq != 0).unsqueeze(-2)

    return collated_arr_seq, seq_mask


class PairedDataset(tud.Dataset):
    """Dataset that takes a list of (input, output) pairs."""

    def __init__(
        self,
        idx,
        data_prefix,
        smiles_id_to_data,
        excluded_idx=set(),
        encoded_smiles_path=None,
        smiles_id_to_encoded_data=None,
        no_swap=False,
        max_length=128,
        use_cache=True,
    ):
        if not use_cache:
            raise ValueError("Not implemented yet!")

        self.idx = idx
        self.max_length = max_length
        self.data_prefix = data_prefix
        self.smiles_id_to_data = smiles_id_to_data
        self.excluded_idx = excluded_idx
        self.encoded_smiles_path = encoded_smiles_path
        self.smiles_id_to_encoded_data = smiles_id_to_encoded_data
        self.no_swap = no_swap

    def push_probas(self, x, gamma):
        y_min = np.power(0.5, gamma)
        y_max = np.power(0.7, gamma)
        s = np.zeros_like(x)
        y = np.power(x[x <= 0.7], gamma)
        s[x >= 0.7] = x[x >= 0.7]
        s[x < 0.7] = (y - y_min) / (y_max - y_min) * (0.7 - 0.5) + 0.5
        return s

    def sample_pair(self, src_key, max_attempts=5):
        if (src_key >= len(self.smiles_id_to_data)) or (
            self.smiles_id_to_data[src_key][0] == -1
        ):
            return (src_key, src_key, 1.0)

        fname1, fname2, fname3, offset = self.smiles_id_to_data[src_key]
        fname = f"{fname1:02d}_{fname2:03d}_{fname3:05d}"
        k, trg_keys, tanimotos = read_binary_file(
            os.path.join(self.data_prefix, fname + ".dat"), offset
        )
        assert src_key == k

        for j in range(max_attempts):
            v = np.random.uniform(tanimotos.min(), tanimotos.max(), size=(1,))
            # Simple trick to undersample pairs which similarity closer to 0.5
            v = self.push_probas(v, -20)

            idx = np.where(tanimotos >= v)[0]
            if len(idx) == 0:
                i = np.argsort(tanimotos)[::-1][min(j, len(tanimotos) - 1)]
            else:
                i = np.random.choice(idx)
            trg_key = trg_keys[i]

            if (trg_key < len(self.smiles_id_to_encoded_data)) and (
                self.smiles_id_to_encoded_data[trg_key] >= 0
            ):
                return (src_key, trg_key, tanimotos[i])
        return (src_key, src_key, 1.0)

    def __getitem__(self, i):
        no_swap = self.no_swap
        src_key = self.idx[i]
        src_key, trg_key, tanimoto = self.sample_pair(src_key)
        tanimoto = np.array((tanimoto,), dtype=np.float32)

        src_pos = self.smiles_id_to_encoded_data[src_key]
        trg_pos = self.smiles_id_to_encoded_data[trg_key]

        _, en_input = read_binary_encoded_smiles(self.encoded_smiles_path, src_pos)
        if src_pos != trg_pos:
            _, en_output = read_binary_encoded_smiles(self.encoded_smiles_path, trg_pos)
        else:
            en_output = en_input

        en_input = en_input.astype(np.int64)
        en_output = en_output.astype(np.int64)

        no_swap = self.no_swap
        if (trg_key < len(self.excluded_idx)) and self.excluded_idx[trg_key]:
            no_swap = True

        if (np.random.rand() > 0.5) or no_swap:
            return (
                torch.from_numpy(en_input),
                torch.from_numpy(en_output),
                torch.from_numpy(tanimoto),
            )
        else:
            return (
                torch.from_numpy(en_output),
                torch.from_numpy(en_input),
                torch.from_numpy(tanimoto),
            )

    def __len__(self):
        return len(self.idx)

    @staticmethod
    def collate_fn(encoded_pairs):
        """Turns a list of encoded pairs (input, target) of sequences and turns them into two batches.

        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the input and one for the targets in the same order as given.
        """
        encoded_inputs, encoded_targets, tanimotos = list(zip(*encoded_pairs))
        tanimotos = torch.cat(tanimotos, 0)

        collated_arr_source, src_mask = _mask_batch(encoded_inputs)
        collated_arr_target, trg_mask = _mask_batch(encoded_targets)

        subseq_mask = (
            subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask).to(trg_mask)
        )
        trg_mask = torch.logical_and(trg_mask, subseq_mask)
        trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        dto = (collated_arr_source, src_mask, collated_arr_target, trg_mask, tanimotos)
        return dto


class FastPairedDataset(tud.Dataset):
    """Dataset that takes a list of (input, output) pairs."""

    def __init__(
        self,
        data,
        max_length=128,
    ):
        self.data = data
        self.max_length = max_length

    def __getitem__(self, i):
        en_input = self.data["src"][i]
        en_output = self.data["trg"][i]
        tanimoto = np.array((self.data["sims"][i],), dtype=np.float32)

        en_input = en_input.astype(np.int64)
        en_output = en_output.astype(np.int64)

        return (
            torch.from_numpy(en_input),
            torch.from_numpy(en_output),
            torch.from_numpy(tanimoto),
        )

    def __len__(self):
        return len(self.data["src"])

    @staticmethod
    def collate_fn(encoded_pairs):
        """Turns a list of encoded pairs (input, target) of sequences and turns them into two batches.

        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the input and one for the targets in the same order as given.
        """
        encoded_inputs, encoded_targets, tanimotos = list(zip(*encoded_pairs))
        tanimotos = torch.cat(tanimotos, 0)

        collated_arr_source, src_mask = _mask_batch(encoded_inputs)
        collated_arr_target, trg_mask = _mask_batch(encoded_targets)

        subseq_mask = (
            subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask).to(trg_mask)
        )
        trg_mask = torch.logical_and(trg_mask, subseq_mask)
        trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        dto = (collated_arr_source, src_mask, collated_arr_target, trg_mask, tanimotos)
        return dto

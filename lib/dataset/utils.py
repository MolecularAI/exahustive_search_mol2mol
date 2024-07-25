import pickle

from .vocabulary import Vocabulary, SMILESTokenizer


def build_vocabulary(smiles) -> Vocabulary:
    tokenizer = SMILESTokenizer()
    tokens = set()
    for smi in smiles[1]:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))
    vocabulary = Vocabulary()
    vocabulary.update(["*", "^", "$", "<UNK>"] + sorted(tokens))  # pad=0, start=1, end=2
    # For random smiles
    if "8" not in vocabulary.tokens():
        vocabulary.update(["8"])
    vocabulary.pad_token = 0  # 0 is padding
    vocabulary.bos_token = 1  # 1 is start symbol
    vocabulary.eos_token = 2  # 2 is end symbol
    vocabulary.unk_token = 3  # 3 is an unknown symbol
    return vocabulary, tokenizer


def smiles_to_tensor(smile, vocabulary, tokenizer, unk_token):
    tokenized_smi = tokenizer.tokenize(smile)
    try:
        encoded_smi = vocabulary.encode(tokenized_smi)
    except KeyError:
        tokenized_smi = [token if token in vocabulary else unk_token for token in tokenized_smi]
        encoded_smi = vocabulary.encode(tokenized_smi)
    return encoded_smi


def _load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def _save(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_vocabulary(path, vocabulary):
    _save(path, vocabulary)


def load_vocabulary(path):
    return _load(path)


def save_tensor_cache(path, tensor_cache):
    _save(path, tensor_cache)


def load_tensor_cache(path):
    return _load(path)

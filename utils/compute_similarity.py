import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from Autoencoder2_emb import Autoencoder
from Vocabulary2 import Vocabulary
import numpy as np
import sys


if __name__ == "__main__":
    vocab = Vocabulary("Vocab_complete.txt", max_len=128)
    autoencoder = Autoencoder(
        "AE/Exp9model2256_500000_biLSTM2_units512_dec_layers2-128-0.9-adam-0.1-256/",
        (vocab.max_len, vocab.vocab_size),
        256,
        512,
        vocab.vocab_size,
        True,
        0.9,
        0.1,
        2,
        256,
        vocab.vocab_size,
        vocab.max_len,
    )
    autoencoder.load_autoencoder_model(
        "AE/Exp9model2256_500000_biLSTM2_units512_dec_layers2-128-0.9-adam-0.1-256/model--86--0.0013.hdf5"
    )

    source = sys.argv[1]
    targets = sys.argv[2:]
    n_examples = len(targets)
    batch_size = 2048
    n_batches = int(np.ceil(n_examples / batch_size))
    TN = []
    for b in range(n_batches):
        Xsrc, Xtrg = [], []
        for j in range(b*batch_size, min((b+1)*batch_size, n_examples)):
            tok, _ = vocab.tokenize([source])
            try:
                x = vocab.encode(tok)[0]
            except:
                x = [21]*128        

            Xsrc.append(x)

            tok, _ = vocab.tokenize([targets[j]])
            try:
                x = vocab.encode(tok)[0]
            except:
                x = [21]*128
            Xtrg.append(x)
        Xsrc = np.reshape(Xsrc, (len(Xsrc), vocab.max_len,1))
        Xtrg = np.reshape(Xtrg, (len(Xtrg), vocab.max_len,1))
        zsrc = autoencoder.smiles_to_latent_model(Xsrc).numpy()
        ztrg = autoencoder.smiles_to_latent_model(Xtrg).numpy()
        tn = (zsrc*ztrg).sum(axis=-1)
        TN += tn.tolist()
    print(" ".join([f"{tn:f}" for tn in TN]))
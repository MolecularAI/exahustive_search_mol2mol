import torch
from torch import nn
import lightning as pl

from .transformer.encoder_decoder import EncoderDecoder
from .rnn.encoder_decoder import EncoderDecoderRNN
from ..dataset.vocabulary import Vocabulary, SMILESTokenizer

# https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
# from tqdm import tqdm

from ..dataset.utils import load_vocabulary


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def log(self, message):
        print(message)


class LitMolformer(pl.LightningModule):
    def __init__(
        self,
        vocabulary,
        use_ranking_loss,
        num_layers=6,
        num_heads=8,
        model_dimension=256,
        feedforward_dimension=2048,
        dropout=0.1,
        max_sequence_length=128,
        model_type=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = SMILESTokenizer()
        if isinstance(vocabulary, str):
            self.vocabulary = load_vocabulary(vocabulary)
        elif isinstance(vocabulary, Vocabulary):
            self.vocabulary = vocabulary
        else:
            raise ValueError("vocabulary type not understood")
        self.max_sequence_length = max_sequence_length
        
        if model_type is None:
            model_class = EncoderDecoder
        elif model_type == "transformer":
            model_class = EncoderDecoder
        elif model_type == "rnn":
            model_class = EncoderDecoderRNN
        else:
            raise ValueError("Model type is invalid")
        
        self.network = model_class(
            vocabulary_size=len(self.vocabulary),
            num_layers=num_layers,
            model_dimension=model_dimension,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
        )
        self._nll_loss = nn.NLLLoss(reduction="none", ignore_index=0)
        self.use_ranking_loss = use_ranking_loss

    def negative_loglikelihood(self, src, src_mask, trg, trg_mask):
        """
        Retrieves the likelihood of molecules.
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param trg: (batch, seq) A batch of output sequences; with start token, without end token.
        :param trg_mask: Mask of the input sequences.
        :return:  (batch) Log likelihood for each output sequence in the batch.
        """
        trg_y = trg[:, 1:]  # skip start token but keep end token
        trg = trg[:, :-1]  # save start token, skip end token
        out = self.network.forward(src, trg, src_mask, trg_mask)
        log_prob = self.network.generator(out).transpose(1, 2)  # (batch, voc, seq_len)
        nll = self._nll_loss(log_prob, trg_y).sum(dim=1)
        return nll

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98))
        self.adam = adam
        # optimizer = NoamOpt(model_size, factor, warmup, adam)
        sch = torch.optim.lr_scheduler.LambdaLR(adam, self._lr_scheduler_fn)
        return {
            "optimizer": adam,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }

    def _lr_scheduler_fn(self, step):
        factor = 1.0
        warmup = 4000
        model_size = self.network.model_dimension
        # self.network.encoder.layers[0].self_attn.linears[0].in_features
        if step < 1:
            step = 1
        return factor / 1e-4 * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    def ranking_loss(self, tanimotos, nll):
        tanimotos = torch.ravel(tanimotos)
        nll = torch.ravel(nll)
        y = 2.0 * (tanimotos[..., None] >= tanimotos[None]) - 1
        n = torch.maximum(torch.zeros_like(y), y * (nll[..., None] - nll[None]))
        n = n.sum() / (len(n) * (len(n) - 1))
        return n

    def training_step(self, train_batch, batch_idx):
        src, src_mask, trg, trg_mask, tanimotos = train_batch
        nll = self.negative_loglikelihood(src, src_mask, trg, trg_mask)
        loss = nll.mean()
        self.log(
            "lr",
            self.adam.param_groups[0]["lr"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )  # , sync_dist=True)
        if self.use_ranking_loss:
            self.log(
                "train_nll_loss",
                loss.item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )  # , sync_dist=True)
            rloss = self.ranking_loss(tanimotos, nll)
            self.log(
                "train_ranking_loss",
                rloss.item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )  # , sync_dist=True)
            loss = loss + 10.0 * rloss
            self.log(
                "train_loss",
                loss.item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )  # , sync_dist=True)
        else:
            self.log(
                "train_loss",
                loss.item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )  # , sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        src, src_mask, trg, trg_mask, tanimotos = val_batch
        nll = self.negative_loglikelihood(src, src_mask, trg, trg_mask)
        loss = nll.mean()
        if self.use_ranking_loss:
            self.log(
                "valid_nll_loss",
                loss.item(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            rloss = self.ranking_loss(tanimotos, nll)
            self.log(
                "valid_ranking_loss",
                rloss.item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            loss = loss + 10.0 * rloss
            self.log(
                "valid_loss",
                loss.item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            self.log("valid_loss", loss.item(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

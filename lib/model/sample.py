import torch
import numpy as np

from .search import beamsearch, LogicalOr, MaxLength, EOS, Node


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


@torch.no_grad()
def sample(
    model,
    src,
    src_mask,
    decode_type,
    beam_size=None,
    beam_search_bs=None,
    device="cuda",
):
    vocabulary = model.vocabulary
    tokenizer = model.tokenizer
    result = None
    if decode_type == "beamsearch":
        stop_criterion = LogicalOr((MaxLength(model.max_sequence_length - 1), EOS()))
        node = Node(
            model.network,
            (src, src_mask),
            vocabulary,
            device,
            batch_size=beam_search_bs,
            data_device="cpu",
        )  # if it explodes use 'cpu' here
        beamsearch(node, beam_size, stop_criterion)
        output_smiles_list = [
            tokenizer.untokenize(vocabulary.decode(seq)) for seq in node.y.detach().cpu().numpy()
        ]
        input_smiles_list = []
        for seq in src.detach().cpu().numpy():
            s = tokenizer.untokenize(model.vocabulary.decode(seq))
            for _ in range(beam_size):
                input_smiles_list.append(s)
        nlls = (-node.loglikelihood.detach().cpu().numpy()).ravel()
        result = (input_smiles_list, output_smiles_list, nlls.tolist())
    else:
        batch_size = src.shape[0]
        ys = model.vocabulary.bos_token * torch.ones(1).to(device)
        ys = ys.repeat(batch_size, 1).view(batch_size, 1).type_as(src.data)  # shape [batch_size, 1]
        encoder_outputs = model.network.encode(src, src_mask)
        break_condition = torch.zeros(batch_size, dtype=torch.bool).to(device)
        nlls = torch.zeros(batch_size).to(device)
        end_token = vocabulary.eos_token
        for i in range(model.max_sequence_length - 1):
            out = model.network.decode(
                encoder_outputs,
                src_mask,
                ys,
                subsequent_mask(ys.size(1)).type_as(src.data),
            )
            # (batch, seq, voc) need to exclude the probability of the start token "1"
            log_prob = model.network.generator(out[:, -1])
            prob = torch.exp(log_prob)

            if decode_type == "greedy":
                _, next_word = torch.max(prob, dim=1)
                # mask numbers after end token as 0
                next_word = next_word.masked_fill(break_condition.to(device), 0)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]

                # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                nlls += model._nll_loss(log_prob, next_word)
            elif decode_type == "multinomial":
                next_word = torch.multinomial(prob, 1)
                # mask numbers after end token as 0
                break_t = torch.unsqueeze(break_condition, 1).to(device)
                next_word = next_word.masked_fill(break_t, 0)
                ys = torch.cat([ys, next_word], dim=1)  # [batch_size, i]
                next_word = torch.reshape(next_word, (next_word.shape[0],))

                # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                nlls += model._nll_loss(log_prob, next_word)

            # next_word = np.array(next_word.to('cpu').tolist())
            break_condition = break_condition | (next_word == end_token)
            if all(break_condition):  # end token
                break

        output_smiles_list = [
            tokenizer.untokenize(vocabulary.decode(seq)) for seq in ys.detach().cpu().numpy()
        ]
        input_smiles_list = [
            tokenizer.untokenize(vocabulary.decode(seq)) for seq in src.detach().cpu().numpy()
        ]
        result = (
            input_smiles_list,
            output_smiles_list,
            nlls.detach().cpu().numpy().tolist(),
        )
    return result

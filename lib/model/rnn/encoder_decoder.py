import torch


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.embedding(x)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class EncoderDecoderRNN(torch.nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        model_dimension,
        feedforward_dimension,
        dropout,
    ):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.model_dimension = model_dimension
        self.feedforward_dimension = feedforward_dimension
        self.dropout = dropout

        self.encoder = EncoderRNN(vocabulary_size, model_dimension, dropout)
        self.decoder = DecoderRNN(model_dimension, vocabulary_size)
        self.generator = lambda x: x
        
    def encode(self, src, src_mask):
        src_enc, src_state = self.encoder(src)
        return src_state
    
    def decode(self, src, src_mask, trg, trg_mask):
        max_seq_length = trg.shape[1]

        trg_state = torch.permute(src.clone(),(1,0,2))
        trg_nexts = []

        for i in range(max_seq_length):
            trg_next, trg_state = self.decoder(trg[:, i:i + 1], trg_state)
            trg_nexts.append(trg_next)

        trg_nexts = torch.cat(trg_nexts, dim=1)
        log_proba_trg_nexts = torch.log_softmax(trg_nexts, dim=-1)
        return log_proba_trg_nexts

    def forward(self, src, trg, src_mask, trg_mask):
        src_enc, src_state = self.encoder(src)
        max_seq_length = trg.shape[1]

        trg_state = src_state
        trg_nexts = []

        for i in range(max_seq_length):
            trg_next, trg_state = self.decoder(trg[:, i:i + 1], trg_state)
            trg_nexts.append(trg_next)

        trg_nexts = torch.cat(trg_nexts, dim=1)
        log_proba_trg_nexts = torch.log_softmax(trg_nexts, dim=-1)
        return log_proba_trg_nexts

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return dict(
            vocabulary_size=self.vocabulary_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            model_dimension=self.model_dimension,
            feedforward_dimension=self.feedforward_dimension,
            dropout=self.dropout,
        )

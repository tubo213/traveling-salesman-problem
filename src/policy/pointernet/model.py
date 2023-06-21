import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerNetEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        x: [N, seq_len, input_size]
        """
        # [N, seq_len, input_size] -> [N, seq_len, hidden_size*2]
        x, hidden = self.lstm(x)
        # [N, seq_len, hidden_size*2] -> [N, seq_len, hidden_size]
        x = self.fc(x)
        return x, hidden


class PointerNetDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        search_method="probabilistic",
        num_glimpses=1,
        use_tanh=False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attn = PointerNetAttention(hidden_size, use_tanh=use_tanh)
        self.glimpse_attn = PointerNetAttention(hidden_size, use_tanh=False)
        self.num_glimpses = num_glimpses
        self.search_method = search_method
        self.bos = nn.Parameter(torch.randn(1, 1, input_size))
        self._initialize_weights()

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, r, hidden: Tuple[torch.Tensor, torch.Tensor]):
        """
        x: [N, seq_len, input_size] (decoder input)
        r: [N, seq_len, hidden_size] (encoder output)
        """
        device = r.device
        bs = r.shape[0]
        num_nodes = r.shape[1]
        idx = torch.arange(bs, device=device)
        mask = torch.zeros(bs, num_nodes, dtype=torch.float, device=device)

        pred_tour = []
        log_ll = torch.zeros(bs, device=device)
        prev_node = self.bos.expand(bs, 1, -1).to(device)  # [N, 1, input_size]
        h, c = hidden
        for i in range(r.shape[1]):
            output, (h, c) = self.lstm(prev_node, (h, c))
            q = output.squeeze()  # [N, hidden_size]

            # glimpse
            for _ in range(self.num_glimpses):
                q = self.glimpse(r, q, mask)

            # attention
            _, logits = self.attn(r, q, mask)  # [N, seq_len]
            log_prob = F.log_softmax(logits, dim=1)

            # sample next node
            position = select_node(log_prob, self.search_method)
            pred_tour.append(position)
            prev_node = x[idx, position, :][:, None, :]
            log_ll += log_prob[idx, position]

            # update mask
            mask[idx, position] = 1

        pred_tour = torch.stack(pred_tour, dim=1)  # [N, seq_len]

        return pred_tour, log_ll

    def glimpse(self, r, q, mask):
        r, logits = self.glimpse_attn(r, q, mask)  # [N, seq_len, hidden], [N, seq_len]
        r = r.transpose(1, 2)  # [N, hidden, seq_len]
        prob = F.softmax(logits, dim=1).unsqueeze(2)  # [N, seq_len, 1]
        q = torch.bmm(r, prob).squeeze(2)  # [N, hidden]

        return q


def select_node(log_prob, search_method):
    if search_method == "greedy":
        position = torch.argmax(log_prob, dim=1).squeeze().long()  # [N]
    elif search_method == "probabilistic":
        position = torch.multinomial(log_prob.exp(), 1).squeeze().long()  # [N]
    else:
        raise ValueError("search_method must be 'greedy' or 'probabilistic'")

    return position


class PointerNetAttention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, clip=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_tanh = use_tanh
        self.clip = clip
        self.w_ref = nn.Parameter(
            torch.zeros([1, hidden_size, hidden_size], dtype=torch.float), requires_grad=True
        )
        self.w_q = nn.Parameter(
            torch.zeros([hidden_size, hidden_size], dtype=torch.float), requires_grad=True
        )
        self.v = nn.Parameter(torch.zeros([1, hidden_size], dtype=torch.float), requires_grad=True)
        self._initialize_weights()
        self.inf = 1e7

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, r, q, mask):
        """
        r: [N, seq_len, hidden_size]
        q: [N, hidden_size]
        mask: [N, seq_len]
        """
        # [N, seq_len, hidden_size] * [1, hidden_size, hidden_size] = [N, seq_len, hidden_size]
        r = torch.matmul(r, self.w_ref)
        # [N, hidden_size] * [hidden_size, hidden_size] = [N, hidden_size]
        q = torch.matmul(q, self.w_q)
        logits = (self.v * F.tanh(r + q[:, None, :])).sum(dim=-1)  # [N, seq_len]
        # mask
        logits = logits - self.inf * mask

        if self.use_tanh:
            logits = self.clip * F.tanh(logits)

        return r, logits


class PointerNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        search_method="probabilistic",
        num_glimpses: int = 1,
        use_tanh: bool = False,
    ):
        super().__init__()
        self.emb = nn.Linear(input_size, hidden_size)
        self.encoder = PointerNetEncoder(hidden_size, hidden_size, num_layers)
        self.decoder = PointerNetDecoder(
            hidden_size,
            hidden_size,
            num_layers,
            search_method=search_method,
            num_glimpses=num_glimpses,
            use_tanh=use_tanh,
        )

    def forward(self, x):
        """
        x: [N, seq_len, input_size]
        """
        x = self.emb(x)
        r, hidden = self.encoder(x)
        pred_tour, log_ll = self.decoder(x, r, hidden)
        return pred_tour, log_ll


class PointerNetCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = PointerNetEncoder(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: [N, seq_len, input_size]
        """
        r, _ = self.encoder(x)
        out = self.fc(r.mean(dim=1))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=1000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[None, : x.size(1), :]
        else:
            x = x + self.pe[: x.size(0), None, :]
        return self.dropout(x)


class TransformerPointerNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        search_method="probabilistic",
        num_glimpses=1,
        use_tanh=False,
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.decoder_emb = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            PositionalEncoding(hidden_size),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=num_layers,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=num_layers,
        )
        self.attn = PointerNetAttention(hidden_size, use_tanh=use_tanh)
        self.glimpse_attn = PointerNetAttention(hidden_size, use_tanh=False)
        self.num_glimpses = num_glimpses
        self.search_method = search_method
        self.bos = nn.Parameter(torch.randn(1, 1, hidden_size))
        self._initialize_weights()

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x):
        """
        x: [N, seq_len, input_size] (decoder input)
        r: [N, seq_len, hidden_size] (encoder output)
        """
        device = x.device
        bs, num_nodes, _ = x.shape
        idx = torch.arange(bs, device=device)
        mask = torch.zeros(bs, num_nodes, dtype=torch.float, device=device)

        # transformer encoder
        x = self.fc(x)
        memory = self.transformer_encoder(x)  # [N, seq_len, hidden_size]

        pred_tour = []
        log_ll = torch.zeros(bs, device=device)
        tgt = self.bos.expand(bs, 1, -1).to(device)  # [N, 1, input_size]
        for i in range(num_nodes):
            embed_tgt = self.decoder_emb(tgt)  # [N, seq_len, hidden_size]
            output = self.transformer_decoder(embed_tgt, memory)  # [N, seq_len, hidden_size]
            q = output[:, -1, :].squeeze()  # [N, hidden_size]

            # glimpse
            for _ in range(self.num_glimpses):
                q = self.glimpse(memory, q, mask)

            # attention
            _, logits = self.attn(memory, q, mask)  # [N, seq_len]
            log_prob = F.log_softmax(logits, dim=1)

            # sample next node
            position = select_node(log_prob, self.search_method)
            pred_tour.append(position)
            pred_node = x[idx, position, :][:, None, :]
            tgt = torch.cat([tgt, pred_node], dim=1)
            log_ll += log_prob[idx, position]

            # update mask
            mask[idx, position] = 1

        pred_tour = torch.stack(pred_tour, dim=1)  # [N, seq_len]

        return pred_tour, log_ll

    def glimpse(self, r, q, mask):
        r, logits = self.glimpse_attn(r, q, mask)  # [N, seq_len, hidden], [N, seq_len]
        r = r.transpose(1, 2)  # [N, hidden, seq_len]
        prob = F.softmax(logits, dim=1).unsqueeze(2)  # [N, seq_len, 1]
        q = torch.bmm(r, prob).squeeze(2)  # [N, hidden]

        return q


class TransformerCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
                num_layers=num_layers,
            ),
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: [N, seq_len, input_size]
        """
        out = self.encoder(x)
        out = self.fc(out.mean(dim=1))
        return out


if __name__ == "__main__":
    input_size = 2
    hidden_size = 128
    seq_len = 20
    batch_size = 4
    actor_model = PointerNet(input_size, hidden_size, 3)
    ciritic_model = PointerNetCritic(input_size, hidden_size, 3)
    x = torch.randn(batch_size, seq_len, input_size)
    pred_tour, log_ll = actor_model(x)

    print(pred_tour)
    print(log_ll.exp())
    print(log_ll)

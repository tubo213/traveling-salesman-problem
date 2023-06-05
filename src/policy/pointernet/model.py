import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        """
        x: [N, seq_len, input_size]
        """
        # [N, seq_len, input_size] -> [N, seq_len, hidden_size*2]
        x, _ = self.lstm(x)
        # [N, seq_len, hidden_size*2] -> [N, seq_len, hidden_size]
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.attn = Attention(hidden_size)

        # bos parameter
        self.hx_0 = nn.Parameter(
            torch.zeros([1, hidden_size], dtype=torch.float), requires_grad=True
        )
        self.cx_0 = nn.Parameter(
            torch.zeros([1, hidden_size], dtype=torch.float), requires_grad=True
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.hx_0)
        nn.init.xavier_uniform_(self.cx_0)

    def forward(self, x, r):
        """
        x: [N, seq_len, input_size] (decoder input)
        r: [N, seq_len, hidden_size] (encoder output)
        """
        device = r.device
        bs = r.shape[0]
        idx = torch.arange(bs, device=device)
        mask = torch.zeros(r.shape[0], r.shape[1], dtype=torch.float, device=device)
        pred_tour = []
        log_prob = []

        hx = self.hx_0.expand(bs, -1)
        cx = self.cx_0.expand(bs, -1)
        for i in range(r.shape[1]):
            hx, cx = self.lstm_cell(x[:, i, :], (hx, cx))
            score = self.attn(r, hx, mask)  # [N, seq_len]
            log_prob_i = F.log_softmax(score, dim=1)
            log_prob.append(log_prob_i)

            # sample from prob
            position = torch.multinomial(log_prob_i.exp(), 1).squeeze().long()  # [N]
            pred_tour.append(position)

            # update mask
            mask[idx, position] = 1

        pred_tour = torch.stack(pred_tour, dim=1)  # [N, seq_len]
        log_prob = torch.stack(log_prob, dim=-1)  # [N, seq_len, seq_len]
        log_ll = self._calc_log_likelihood(log_prob, pred_tour)  # [N]

        return pred_tour, log_ll

    def _calc_log_likelihood(self, log_prob, tour):
        """
        log_prob: [N, seq_len, seq_len]
        tour: [N, seq_len]
        """
        log_prob = torch.gather(log_prob, dim=2, index=tour[:, :, None])  # [N, seq_len, 1]
        log_prob = log_prob.squeeze()  # [N, seq_len]
        return log_prob.sum(dim=1)  # [N]


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_ref = nn.Parameter(
            torch.zeros([1, hidden_size, hidden_size], dtype=torch.float), requires_grad=True
        )
        self.w_q = nn.Parameter(
            torch.zeros([hidden_size, hidden_size], dtype=torch.float), requires_grad=True
        )
        self.v = nn.Parameter(torch.zeros([1, hidden_size], dtype=torch.float), requires_grad=True)
        self._reset_parameters()
        self.inf = 1e8

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_ref)
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.v)

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

        score = (self.v * F.tanh(r + q[:, None, :])).sum(dim=-1)  # [N, seq_len]

        # mask
        score = score - self.inf * mask

        return score


class PointerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size)

    def forward(self, x):
        """
        x: [N, seq_len, input_size]
        """
        r = self.encoder(x)
        pred_tour, log_prob = self.decoder(x, r)
        return pred_tour, log_prob


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: [N, seq_len, input_size]
        """
        r = self.encoder(x).mean(dim=1)
        out = self.fc(r)
        return out


if __name__ == "__main__":
    input_size = 2
    hidden_size = 128
    seq_len = 10
    batch_size = 4
    actor_model = PointerNet(input_size, hidden_size, 3)
    ciritic_model = Critic(input_size, hidden_size, 3)
    x = torch.randn(batch_size, seq_len, input_size)
    pred_tour, log_prob = actor_model(x)

    print(pred_tour)
    print(log_prob)

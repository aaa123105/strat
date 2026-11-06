import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class HamiltonianBlock(nn.Module):
    def __init__(self, d_model, max_len, kernel_size, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        N_f = max_len // 2 + 1

        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 3 * d_model)

        self.m_raw = nn.Parameter(torch.zeros(d_model))
        self.c_raw = nn.Parameter(torch.zeros(d_model))
        self.k_raw = nn.Parameter(torch.zeros(d_model))

        self.psi_re = nn.Parameter(torch.ones(d_model, N_f))
        self.psi_im = nn.Parameter(torch.zeros(d_model, N_f))

        omega = 2 * math.pi * torch.arange(N_f).float() / max_len
        self.register_buffer('omega', omega)

        self.kernel_size = kernel_size
        self.impulse_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            groups=d_model, bias=False
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def _spectral_propagate(self, x):
        B, T, D = x.shape
        F_hat = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        Nf = F_hat.shape[-1]

        w = self.omega[:Nf].unsqueeze(0)
        psi = torch.complex(self.psi_re[:, :Nf], self.psi_im[:, :Nf])

        m = F.softplus(self.m_raw).unsqueeze(1) + 1e-1
        c = F.softplus(self.c_raw).unsqueeze(1) + 1e-6
        k = F.softplus(self.k_raw).unsqueeze(1) + 1e-2

        D_re = k - m * w.pow(2)
        D_im = c * w
        D_complex = torch.complex(D_re, D_im)

        G = psi / D_complex
        Q_hat = F_hat * G.unsqueeze(0)

        Q_tilde = torch.fft.irfft(Q_hat, n=T, dim=-1)
        return Q_tilde.transpose(1, 2)

    def forward(self, x, mask):
        h = self.norm1(x)
        proj = self.in_proj(h)
        F_br, U_br, A_br = proj.chunk(3, dim=-1)
        F_br, U_br = F.gelu(F_br), F.gelu(U_br)
        g = torch.sigmoid(A_br)

        Q_tilde = self._spectral_propagate(F_br)

        U_pad = F.pad(U_br.transpose(1, 2), (self.kernel_size - 1, 0))
        H_loc = self.impulse_conv(U_pad).transpose(1, 2)

        Z = (Q_tilde + H_loc) * g
        x = x + self.dropout1(self.out_proj(Z))
        x = x * mask
        x = x + self.ffn(x)
        x = x * mask

        return x


class HSR(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.n_layers = config['num_layers']
        self.max_len = config['MAX_ITEM_LIST_LENGTH']
        self.dropout_prob = config['dropout_prob']
        self.kernel_size = config['kernel_size']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.emb_norm = nn.LayerNorm(self.hidden_size)

        self.blocks = nn.ModuleList([
            HamiltonianBlock(self.hidden_size, self.max_len, kernel_size=self.kernel_size, dropout=self.dropout_prob)
            for _ in range(self.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.hidden_size)

        self.inv_mass_dt = nn.Parameter(torch.zeros(self.hidden_size))

        self.momentum_dropout = nn.Dropout(self.dropout_prob)

        omega = 2 * math.pi * torch.arange(self.max_len // 2 + 1).float() / self.max_len
        self.register_buffer('omega_final', omega)

        self.loss_type = config['loss_type']
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, item_seq, item_seq_len):
        B, T = item_seq.shape
        item_emb = self.item_embedding(item_seq)
        positions = torch.arange(T, device=item_seq.device).unsqueeze(0).expand(B, T)

        mask = (item_seq != 0).float().unsqueeze(-1)
        x = self.emb_norm(self.emb_dropout(item_emb + self.position_embedding(positions)))
        x = x * mask

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)

        last_idx = (item_seq != 0).cumsum(dim=1).argmax(dim=1)
        idx = torch.arange(B, device=x.device)
        q_L = x[idx, last_idx]

        X_hat = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        Nf = X_hat.shape[-1]

        omega_exp = self.omega_final[:Nf].unsqueeze(0).unsqueeze(0)
        freq_decay = torch.exp(-self.omega_final[:Nf]).unsqueeze(0).unsqueeze(0)

        V_hat = (1j * omega_exp) * X_hat * freq_decay
        v_seq = torch.fft.irfft(V_hat, n=T, dim=-1).transpose(1, 2)
        v_L = v_seq[idx, last_idx]

        # ：q_hat = q_L + (M^-1 * \Delta t) \odot v_L
        v_L_dropped = self.momentum_dropout(v_L)
        q_hat = q_L + v_L_dropped * self.inv_mass_dt

        return q_hat

    def _score(self, q_hat, item_emb):
        if item_emb.dim() == 2 and item_emb.shape[0] != q_hat.shape[0]:
            return torch.matmul(q_hat, item_emb.t())
        else:
            return (q_hat * item_emb).sum(dim=-1)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        q_hat = self.forward(item_seq, item_seq_len)

        if self.loss_type == "BPR":
            pos_items = interaction[self.POS_ITEM_ID]
            pos_score = self._score(q_hat, self.item_embedding(pos_items))
            neg_items = interaction[self.NEG_ITEM_ID]
            if neg_items is None:
                neg_items = torch.randint(1, self.n_items, (q_hat.shape[0],), device=q_hat.device)
            neg_score = self._score(q_hat, self.item_embedding(neg_items))
            return self.loss_fct(pos_score, neg_score)
        else:
            pos_items = interaction[self.POS_ITEM_ID]
            test_item_emb = self.item_embedding.weight
            logits = self._score(q_hat, test_item_emb)
            return self.loss_fct(logits, pos_items)

    def full_sort_predict(self, interaction):
        q_hat = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        return self._score(q_hat, self.item_embedding.weight)

    def predict(self, interaction):
        q_hat = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        return self._score(q_hat, self.item_embedding(interaction[self.ITEM_ID]))
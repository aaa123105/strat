import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


def build_dct_bank(max_len):
    bank = torch.zeros(max_len + 1, max_len, max_len, dtype=torch.float32)
    for T in range(1, max_len + 1):
        n = torch.arange(T, dtype=torch.float32).view(1, -1)
        k = torch.arange(T, dtype=torch.float32).view(-1, 1)
        C = torch.cos(math.pi / T * (n + 0.5) * k)
        C[0] *= math.sqrt(1.0 / T)
        if T > 1:
            C[1:] *= math.sqrt(2.0 / T)
        bank[T, :T, :T] = C
    return bank


def build_band_bank(max_len, num_bands=4):
    bank = torch.zeros(max_len + 1, num_bands, max_len, dtype=torch.float32)
    for T in range(1, max_len + 1):
        for k in range(T):
            band = min((num_bands * k) // T, num_bands - 1)
            bank[T, band, k] = 1.0
    return bank


def build_laplacian_bank(max_len):
    bank = torch.zeros(max_len + 1, max_len, dtype=torch.float32)
    for T in range(1, max_len + 1):
        k = torch.arange(T, dtype=torch.float32)
        bank[T, :T] = 4.0 * torch.sin(math.pi * k / (2.0 * T)).pow(2)
    return bank


class StructureMix(nn.Module):
    def __init__(self, d_model, hidden_rate=4, dropout=0.2):
        super().__init__()
        self.mix_logit = nn.Parameter(torch.full((d_model,), -2.0))
        hidden_size = int(hidden_rate * d_model)
        self.key = nn.Linear(d_model, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    @staticmethod
    def sequence_shift(x):
        xx = F.pad(x, (0, 0, 1, 0), mode="constant", value=0)
        return xx[:, :-1, :]

    def forward(self, x, s, persistence):
        prev_s = self.sequence_shift(s)
        prev2_s = self.sequence_shift(prev_s)
        s_vel = prev_s - prev2_s
        s_inertial = prev_s + persistence * s_vel

        alpha = torch.sigmoid(self.mix_logit).view(1, 1, -1)
        x_mix = x + persistence * alpha * s_inertial

        k = torch.square(torch.relu(self.key(x_mix)))
        hidden = self.dropout(self.value(k))
        return self.LayerNorm(hidden + x)


class STRATLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        lambda_freq_init=0.1,
        lambda_time_init=0.05,
        routing_strength=0.5,
        laplacian_tau_init=0.5,
        laplacian_mix_init=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bands = 4
        self.routing_strength = routing_strength

        self.in_proj = nn.Linear(d_model, d_model, bias=True)

        self.raw_lambda_freq = nn.Parameter(torch.tensor(self._inverse_sigmoid(lambda_freq_init), dtype=torch.float32))
        self.raw_lambda_time = nn.Parameter(torch.tensor(self._inverse_sigmoid(lambda_time_init), dtype=torch.float32))
        self.raw_laplacian_tau = nn.Parameter(torch.tensor(self._inverse_softplus(laplacian_tau_init), dtype=torch.float32))
        self.raw_laplacian_mix = nn.Parameter(torch.tensor(self._inverse_sigmoid(laplacian_mix_init), dtype=torch.float32))

        self.structured_proj = nn.Linear(d_model, d_model, bias=True)
        self.residual_proj = nn.Linear(d_model, d_model, bias=True)
        self.residual_gate = nn.Linear(2 * d_model, d_model, bias=True)

        self.routing_hidden = nn.Linear(self.num_bands + 2, 16, bias=True)
        self.routing_out = nn.Linear(16, self.num_bands, bias=True)

        self.input_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.structure_mix = StructureMix(
            d_model=d_model,
            hidden_rate=4,
            dropout=dropout,
        )

    @staticmethod
    def _inverse_sigmoid(x):
        return math.log(float(x) / (1.0 - float(x)))

    @staticmethod
    def _inverse_softplus(x):
        return math.log(math.expm1(float(x)))

    @staticmethod
    def batch_dct(x, dct_matrix):
        return torch.bmm(dct_matrix, x)

    @staticmethod
    def batch_idct(x, dct_matrix):
        return torch.bmm(dct_matrix.transpose(1, 2), x)

    @staticmethod
    def group_shrink(x, ratio, mask, valid_count):
        group_norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        reference = (group_norm * mask).sum(dim=1, keepdim=True) / valid_count
        threshold = ratio * reference.detach()
        scale = torch.clamp(1.0 - threshold / (group_norm + 1e-6), min=0.0)
        return x * scale

    @staticmethod
    def guided_shrink(x, ratio, mask, valid_count, mode_scale):
        group_norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        reference = (group_norm * mask).sum(dim=1, keepdim=True) / valid_count
        threshold = ratio * reference.detach() * mode_scale
        scale = torch.clamp(1.0 - threshold / (group_norm + 1e-6), min=0.0)
        return x * scale

    def behavior_profile(self, A1, S1, R1, mask, band_mask, laplacian_eigs):

        mode_energy = torch.sum(A1 * A1, dim=-1)

        # 1. Band energy distribution across 4 bands (b_1 ... b_4)
        band_energy = torch.bmm(band_mask, mode_energy.unsqueeze(-1)).squeeze(-1)
        band_energy = band_energy / (band_energy.sum(dim=-1, keepdim=True) + 1e-6)

        # 2. Residual energy ratio (rho)
        s_norm = torch.linalg.vector_norm(S1, ord=2, dim=-1, keepdim=True)
        r_norm = torch.linalg.vector_norm(R1, ord=2, dim=-1, keepdim=True)
        residual_energy = (r_norm * mask).sum(dim=1)
        total_energy = ((s_norm + r_norm) * mask).sum(dim=1)
        residual_ratio = residual_energy / (total_energy + 1e-6)

        # 3. Path roughness (r)
        norm_eigs = laplacian_eigs.squeeze(-1) / 4.0
        roughness = (mode_energy * norm_eigs).sum(dim=1, keepdim=True)
        roughness = roughness / (mode_energy.sum(dim=1, keepdim=True) + 1e-6)

        return torch.cat([band_energy, residual_ratio, roughness], dim=-1).detach()

    def adaptive_routing(self, profile, band_mask):
        routing_hidden = F.relu(self.routing_hidden(profile))
        route_delta = self.routing_out(routing_hidden)
        band_scale = 1.0 + self.routing_strength * torch.tanh(route_delta)
        return torch.bmm(band_scale.unsqueeze(1), band_mask).transpose(1, 2)

    def spectral_smooth(self, spectrum, laplacian_eigs):
        tau = F.softplus(self.raw_laplacian_tau)
        mix = torch.sigmoid(self.raw_laplacian_mix)
        transfer = torch.reciprocal(1.0 + tau * laplacian_eigs)
        return spectrum + mix * (spectrum * transfer - spectrum)

    def forward(self, x, mask, dct_matrix, band_mask, laplacian_eigs, valid_count):
        x = x * mask
        q = self.in_proj(self.input_norm(x)) * mask

        lambda_freq = torch.sigmoid(self.raw_lambda_freq)
        lambda_time = torch.sigmoid(self.raw_lambda_time)

        # ==========================================
        # TSP
        # ==========================================
        A1_raw = self.batch_dct(q, dct_matrix)
        A1 = self.group_shrink(A1_raw, lambda_freq, mask, valid_count)
        S1 = self.batch_idct(A1, dct_matrix) * mask

        R1_raw = q - S1
        R1 = self.group_shrink(R1_raw, lambda_time, mask, valid_count) * mask

        # ==========================================
        # PGR
        # ==========================================
        profile = self.behavior_profile(A1, S1, R1, mask, band_mask, laplacian_eigs)
        mode_scale = self.adaptive_routing(profile, band_mask)

        A2_raw = self.batch_dct(q - R1, dct_matrix)
        A2_regularized = self.spectral_smooth(A2_raw, laplacian_eigs)
        A2 = self.guided_shrink(A2_regularized, lambda_freq, mask, valid_count, mode_scale)

        S = self.batch_idct(A2, dct_matrix) * mask
        R = (q - S) * mask

        # ==========================================
        # PAF
        # ==========================================
        s_hidden = self.structured_proj(S) * mask
        r_hidden = self.residual_proj(R) * mask
        r_gate = torch.sigmoid(self.residual_gate(torch.cat([S, R], dim=-1))) * mask
        mixed_hidden = s_hidden + r_gate * r_hidden

        mixed_hidden = self.out_norm(mixed_hidden) * mask
        hidden = (x + self.dropout(mixed_hidden)) * mask

        s_norm = torch.linalg.vector_norm(S, ord=2, dim=-1, keepdim=True)
        r_norm = torch.linalg.vector_norm(R, ord=2, dim=-1, keepdim=True)

        conf = s_norm / (s_norm + r_norm + 1e-6)
        prev_conf = F.pad(conf[:, :-1, :], (0, 0, 1, 0), mode="constant", value=0.0)
        persistence = torch.sqrt(conf * prev_conf).detach() * mask

        hidden = self.structure_mix(hidden, s_hidden, persistence) * mask
        return hidden


class STRAT(SequentialRecommender):
    r"""STRAT: Structure-Oriented Spectral Sequential Recommendation."""
    def __init__(self, config, dataset):
        super(STRAT, self).__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.seq_len = config['MAX_ITEM_LIST_LENGTH']

        self.lambda_freq_init = config['lambda_freq_init']
        self.lambda_time_init = config['lambda_time_init']
        self.routing_strength = config['routing_strength']
        self.laplacian_tau_init = config['laplacian_tau_init']
        self.laplacian_mix_init = config['laplacian_mix_init']

        self.register_buffer("dct_bank", build_dct_bank(self.seq_len))
        self.register_buffer("band_bank", build_band_bank(self.seq_len, num_bands=4))
        self.register_buffer("laplacian_bank", build_laplacian_bank(self.seq_len))

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.layers = nn.ModuleList([
            STRATLayer(
                self.hidden_size,
                self.dropout_prob,
                self.lambda_freq_init,
                self.lambda_time_init,
                self.routing_strength,
                self.laplacian_tau_init,
                self.laplacian_mix_init,
            )
            for _ in range(self.num_layers)
        ])

        self.depth_momentum_logit = nn.Parameter(torch.full((self.num_layers,), -2.0))

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

        for layer in self.layers:
            nn.init.zeros_(layer.routing_out.weight)
            nn.init.zeros_(layer.routing_out.bias)

        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        B, L = item_seq.shape

        pos = torch.arange(L, device=item_seq.device).unsqueeze(0)
        mask = (pos < item_seq_len.unsqueeze(1)).unsqueeze(-1).to(item_emb.dtype)
        item_emb = item_emb * mask

        valid_count = mask.sum(dim=1, keepdim=True)

        dct_matrix = self.dct_bank[item_seq_len][:, :L, :L]
        band_mask = self.band_bank[item_seq_len][:, :, :L]
        laplacian_eigs = self.laplacian_bank[item_seq_len][:, :L].unsqueeze(-1)

        if dct_matrix.dtype != item_emb.dtype:
            dct_matrix = dct_matrix.to(item_emb.dtype)
            band_mask = band_mask.to(item_emb.dtype)
            laplacian_eigs = laplacian_eigs.to(item_emb.dtype)

        velocity = torch.zeros_like(item_emb)

        for i, layer in enumerate(self.layers):
            layer_out = layer(item_emb, mask, dct_matrix, band_mask, laplacian_eigs, valid_count)
            delta = layer_out - item_emb
            momentum = torch.sigmoid(self.depth_momentum_logit[i])
            velocity = momentum * velocity + delta
            velocity = velocity * mask
            item_emb = (item_emb + velocity) * mask

        return self.gather_indexes(item_emb, item_seq_len - 1)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)

            loss = self.loss_fct(pos_score, neg_score)
            return loss

        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)

        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

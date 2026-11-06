import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class ACLSRec(SequentialRecommender):
    r"""
    ACLSRec incorporates learnable perturbation and restoration networks for adaptive augmentation.
    The framework dynamically perturbs and restores user representations, thereby ensuring
    semantic consistency across augmented views.
    """

    def __init__(self, config, dataset):
        super(ACLSRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Hyperparameters for ACLSRec losses
        self.lmd = config['lmd']
        self.tau = config['tau']
        self.sim = config['sim']
        self.sim_lambda = config['sim_lambda']
        self.gamma = config['gamma']
        self.batch_size = config['train_batch_size']

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

        # Perturbation Network (MLP)
        self.perturb_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size * 4),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(self.hidden_dropout_prob)
        )

        # Restoration Network (MLP)
        self.restore_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size * 4),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(self.hidden_dropout_prob)
        )

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_raw_embeddings(self, item_seq):
        """ Get the embedding of the input sequence """
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        return input_emb

    def trm_encode_sequence(self, input_emb, item_seq):
        """ Encode the sequence using Transformer """
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output

    def perturb_augment(self, raw_emb):
        """ Apply perturbation network to generate augmented view """
        # raw_emb shape: [B, L, H]
        perturbed_emb = self.perturb_net(raw_emb)  # [B, L, H]
        return perturbed_emb

    def reversible_augment(self, perturbed_emb):
        """ Apply restoration network to reconstruct original semantics """
        restored_emb = self.restore_net(perturbed_emb)  # [B, L, H]
        return restored_emb

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # 1. Get Input Embeddings (H_in)
        raw_emb = self.get_raw_embeddings(item_seq)

        # 2. Original Branch (H_o)
        raw_out1 = self.trm_encode_sequence(raw_emb, item_seq)
        raw_out = self.gather_indexes(raw_out1, item_seq_len - 1)

        pos_items = interaction[self.POS_ITEM_ID]
        self.item_seq_len = item_seq_len

        # 3. Perturbation Branch 1 (H_a -> H_p)
        # Perturb embeddings then encode
        raw_perturb_out = self.perturb_augment(raw_out1)
        raw_perturb = self.gather_indexes(raw_perturb_out, item_seq_len - 1)

        # 4. Perturbation Branch 2 (Contrastive View construction)
        # Apply perturbation to raw embeddings first
        perturbed_emb = self.perturb_augment(raw_emb)  # [B, L, H]
        perturb_out1 = self.trm_encode_sequence(perturbed_emb, item_seq)
        perturb_out = self.gather_indexes(perturb_out1, item_seq_len - 1)

        # 5. Restoration Process (H_pr)
        # Restore from perturbed representation
        restored_out1 = self.reversible_augment(perturb_out1)
        restored_out = self.gather_indexes(restored_out1, item_seq_len - 1)

        # --- Main Task Loss (L_main) ---
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(raw_out * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(raw_out * neg_items_emb, dim=-1)  # [B]
            main_loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(raw_out, test_item_emb.transpose(0, 1))
            main_loss = self.loss_fct(logits, pos_items)

        # --- Contrastive Loss (L_cl = L_pp + L_rr) ---
        # L_pp: Contrast between two perturbation views
        nce_loss2 = self._compute_contrastive_loss(raw_perturb, perturb_out, item_seq_len.shape[0])
        # L_rr: Contrast between original and restored views
        nce_loss3 = self._compute_contrastive_loss(raw_out, restored_out, item_seq_len.shape[0])

        # --- Regularization Loss (R) ---
        regularization_loss1 = self.compute_contrastive_regularization(raw_out, restored_out)
        regularization_loss2 = self.compute_contrastive_regularization(raw_perturb, perturb_out)

        # --- Distance Alignment Loss (L_align) ---
        sim_loss1 = self.compute_distance_alignment_loss(raw_out, perturb_out, restored_out, raw_perturb)

        # Total Loss
        total_loss = (
                main_loss
                + self.lmd * (nce_loss2 + nce_loss3)
                + self.gamma * (regularization_loss1 + regularization_loss2)
                + self.sim_lambda * sim_loss1
        )
        return total_loss

    def _compute_contrastive_loss(self, z_i, z_j, batch_size):
        """ Helper to compute InfoNCE loss """
        logits, labels = self.info_nce(z_i, z_j, self.tau, batch_size, self.sim)
        return self.nce_fct(logits, labels)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, we treat the other 2(N − 1) augmented examples
        within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def compute_contrastive_regularization(self, z1, z2):
        """
        Compute Dynamic Contrast Regularization (R).
        This term adaptively adjusts the decision boundary for positive and negative sample pairs
        to prevent representation collapse and over-alignment.

        Args:
            z1 (Tensor): Augmented embedding 1 [Batch, Dim]
            z2 (Tensor): Augmented embedding 2 [Batch, Dim]
        Returns:
            Tensor: Regularization loss value
        """
        batch_size = z1.size(0)
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, embedding_dim)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T)  # (2B, 2B)

        # Mask for diagonal (self-similarity) and positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        for i in range(batch_size):
            mask[i, i + batch_size] = True
            mask[i + batch_size, i] = True

            # Extract positive pair similarities (Sigma+)
        pos_indices = torch.arange(batch_size, device=z.device)
        sigma_plus = sim_matrix[pos_indices, pos_indices + batch_size]  # (B,)

        # Extract negative pair similarities (Sigma-)
        sigma_minus = sim_matrix[~mask]

        # Dynamic thresholds based on batch distribution
        o_min = torch.min(sigma_plus)
        o_max = torch.max(sigma_minus)

        # Compute Regularization term
        term1 = torch.mean(torch.clamp(sigma_plus - o_min, min=0.0))
        term2 = torch.mean(torch.clamp(o_max - sigma_minus, min=0.0))

        R = term1 + term2
        return R

    def compute_distance_alignment_loss(self, raw_out, perturb_out, restored_out, raw_perturb):
        """
        Compute Distance Constraint Loss (L_align).
        Balances representation distributions by enforcing symmetric geometric distances
        between original and augmented views.
        """
        # L2 Normalization for Cosine distance calculation
        raw_out = F.normalize(raw_out, p=2, dim=-1)
        perturb_out = F.normalize(perturb_out, p=2, dim=-1)
        restored_out = F.normalize(restored_out, p=2, dim=-1)
        raw_perturb = F.normalize(raw_perturb, p=2, dim=-1)

        # Calculate geometric distances (1 - cosine similarity)
        distance_raw_restored = 1 - torch.sum(raw_out * restored_out, dim=-1)
        distance_perturb_rawperturb = 1 - torch.sum(perturb_out * raw_perturb, dim=-1)

        # Mean Squared Error of distances
        mse_distance = torch.mean((distance_raw_restored - distance_perturb_rawperturb) ** 2)
        return mse_distance
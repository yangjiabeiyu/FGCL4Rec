from torch import nn
import torch.nn.functional as F
from utils import *


class PointWiseFeedForward(torch.nn.Module):
    """Point-wise feed-forward network"""

    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs)))))
        outputs += inputs
        return outputs


class UpDown(torch.nn.Module):
    """Up-down projection module for feature enhancement"""

    def __init__(self, hidden_units, inner_units):
        super().__init__()
        self.hidden = hidden_units
        self.inner = inner_units
        self.activate = torch.nn.ReLU()
        self.up = torch.nn.Linear(self.hidden, self.inner)
        self.gate = torch.nn.Linear(self.hidden, self.inner)
        self.down = torch.nn.Linear(self.inner, self.hidden)
        self.dropout_layer = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        y_up = self.up(x)
        gate = self.activate(self.dropout_layer(self.gate(x)))
        out = x + self.activate(self.dropout_layer(self.down(gate * y_up)))
        return out


class FGCL4Rec(torch.nn.Module):
    """FGCL4Rec implementation: Frequency-guided Dual-view Graph Contrastive Learning for Sequential Recommendation"""

    def __init__(self, user_num, item_num, item_matrix, item_occur_matrix, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.hidden_dim = args.hidden_units
        self.max_len = args.max_len
        self.dropout_rate = args.dropout_rate

        # Precomputed matrices (fixed during training)
        self.adj_matrix = torch.FloatTensor(item_matrix.T).to(self.dev)
        self.adj_matrix.requires_grad = False
        self.occurrence_mat = torch.FloatTensor(item_occur_matrix).to(self.dev)
        self.occurrence_mat.requires_grad = False

        # Augmentation and contrastive learning parameters
        self.K_neg = args.K_neg
        self.k_neg = args.k_neg
        self.temperature = args.temperature
        self.node_aug_portion_shuffle = args.node_aug_portion_shuffle
        self.node_aug_portion_mask = args.node_aug_portion_mask
        self.node_aug_portion_add = args.node_aug_portion_add
        self.node_aug_portion_drop = args.node_aug_portion_drop
        self.mask_std = args.mask_std
        self.K_add = args.K_add
        self.attn_aug_flag = args.attn_aug_flag
        self.trans_aug_flag = args.trans_aug_flag

        # Trainable parameters
        self.W_item = nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W_item.data, gain=1.414)
        self.a_item = nn.Parameter(torch.zeros(size=(2 * self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a_item.data, gain=1.414)

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.hidden_dim)
        self.updown = UpDown(self.hidden_dim, 2 * self.hidden_dim)
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.activate = nn.LeakyReLU()
        self.layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-8)
        self.fwd_layer = PointWiseFeedForward(self.hidden_dim, 0.4)

        self.W_1 = nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W_1, gain=1.414)
        self.W_2 = nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W_2, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.b, gain=1.414)
        self.sigmoid = nn.Sigmoid()

        self.co_center = nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.co_center, gain=1.414)
        self.co_neighbor = nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.co_neighbor, gain=1.414)
        self.aggregation = nn.Parameter(torch.zeros(size=(self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.aggregation, gain=1.414)

    def node_trans_aug(self, aug_flag, item_trans, std=0.2):
        """Apply augmentation to transition view (weight-based)"""
        if aug_flag == 0:
            aug_flag = random.sample(range(1, 3), k=1)[0]
        if aug_flag == 1:
            weight_shuffle(item_trans, self.node_aug_portion_shuffle)
        elif aug_flag == 2:
            weight_mask(item_trans, dev=self.dev, std=std, portion=self.node_aug_portion_mask)

    def node_attn_aug(self, aug_flag, item_trans, item_attn, item_attn_occurrence, K_add=200):
        """Apply augmentation to attention view (structure-based)"""
        if aug_flag == 0:
            aug_flag = random.sample(range(1, 3), k=1)[0]
        if aug_flag == 1:
            node_drop(item_trans, item_attn, portion=self.node_aug_portion_drop)
        else:
            node_add(item_trans, item_attn, item_attn_occurrence, portion=self.node_aug_portion_add, topk=K_add)

    def node_augmentation(self, attn_mat, trans_mat, h_item):
        """Generate augmented views and negative samples for contrastive learning"""
        n_nodes = attn_mat.shape[0] - 1  # Exclude padding
        attn_aug = attn_mat.clone()
        trans_aug = trans_mat.clone()
        attn_occurrence_sum = attn_aug + self.occurrence_mat
        attn_occurrence_sum[trans_aug > 0] = 0
        attn_occurrence_max = torch.max(attn_mat, self.occurrence_mat)
        attn_occurrence_max[trans_mat > 0] = 1e10

        # Generate negative samples
        neg_samples = torch.zeros([n_nodes + 1, self.k_neg], dtype=torch.long)
        for i in range(1, n_nodes + 1):
            self.node_attn_aug(self.attn_aug_flag, trans_aug[i], attn_aug[i], attn_occurrence_sum[i], K_add=self.K_add)
            self.node_trans_aug(self.trans_aug_flag, trans_aug[i], std=self.mask_std)
            neg_samples[i] = neg_sample_attn(attn_occurrence_max[i], neg_num_1=self.K_neg, neg_num_2=self.k_neg)

        attn_aug = F.softmax(attn_aug, dim=1)
        return torch.mm(attn_aug, h_item), torch.mm(trans_aug, h_item), neg_samples

    def log2feats(self, log_seqs, step):
        """Transform sequence logs to feature representations"""
        gat_info, trans_info, aug_logits = self.update(step)
        gat_neighbor = gat_info[torch.tensor(log_seqs).long()]
        trans_neighbor = trans_info[torch.tensor(log_seqs).long()]
        seqs_self = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))

        # Fusion of dual-view information
        coff = self.sigmoid(torch.matmul(gat_neighbor, self.co_center) + torch.matmul(trans_neighbor, self.co_neighbor))
        seqs = coff * gat_neighbor + (1 - coff) * trans_neighbor + seqs_self
        seqs = self.emb_dropout(seqs)

        # Positional encoding
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        pos = self.pos_emb(torch.LongTensor(positions).to(self.dev))
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        pos = pos * ~timeline_mask.unsqueeze(-1)
        seqs_pos = seqs + pos

        # Self-attention mechanism
        M1 = torch.matmul(seqs_pos, self.W_1)
        M2 = torch.matmul(seqs_pos, self.W_2)
        attn_score = self.sigmoid(M1.unsqueeze(2) + M2.unsqueeze(1))
        attn_score = torch.matmul(attn_score, self.b).squeeze(-1)

        # Masked attention (causal mask)
        attention_mask = torch.tril(torch.ones((log_seqs.shape[1], log_seqs.shape[1]), device=self.dev))
        attention_mask = attention_mask.unsqueeze(0)
        attn_score = attn_score.masked_fill(attention_mask == 0, 0)

        # Feature aggregation
        final = torch.matmul(attn_score, seqs)
        final = self.fwd_layer(final)
        final = self.updown(final)
        return final, aug_logits

    def update(self, step):
        """Update graph representations and generate contrastive signals"""
        h_item = torch.matmul(self.item_emb.weight, self.W_item)
        wh1_item = torch.matmul(h_item, self.a_item[: self.hidden_dim])
        wh2_item = torch.matmul(h_item, self.a_item[self.hidden_dim:])
        e_item = self.activate(wh1_item + wh2_item.T)
        zero_item = -9e15 * torch.ones_like(e_item)

        # Attention view computation
        attn_all = torch.where(self.adj_matrix > 0, e_item, zero_item)
        attn_all = F.softmax(attn_all, dim=1)
        attn_all = F.dropout(attn_all, self.dropout_rate * 0, training=self.training)

        # Transition view computation
        trans_info = torch.matmul(self.adj_matrix, h_item)
        gat_info = torch.mm(attn_all, h_item)

        if not step:
            attn_aug, adj_aug, neg_samples = self.node_augmentation(e_item, self.adj_matrix, h_item)
            neg_samples_attn = gat_info[neg_samples]
            neg_samples_trans = trans_info[neg_samples]

            # Intra-view contrastive signals
            pos_attn_logits = (attn_aug[1:] * gat_info[1:]).sum(dim=-1) / self.temperature
            neg_attn_logits = neg_samples_attn[1:].matmul(gat_info[1:].unsqueeze(-1)).squeeze(-1) / self.temperature
            pos_trans_logits = (adj_aug[1:] * trans_info[1:]).sum(dim=-1) / self.temperature
            neg_trans_logits = neg_samples_trans[1:].matmul(trans_info[1:].unsqueeze(-1)).squeeze(-1) / self.temperature

            # Cross-view contrastive signals
            pos_attn_trans_logits = (gat_info[1:] * trans_info[1:]).sum(dim=-1) / self.temperature
            neg_attn_trans_logits = neg_samples_trans[1:].matmul(gat_info[1:].unsqueeze(-1)).squeeze(
                -1) / self.temperature

            aug_logits = [pos_attn_logits, neg_attn_logits, pos_trans_logits, neg_trans_logits,
                          pos_attn_trans_logits, neg_attn_trans_logits]
        else:
            aug_logits = []

        return gat_info, trans_info, aug_logits

    def forward(self, log_seqs, pos_seqs, neg_seqs, step):  # Removed unused user_ids
        """Forward pass for training"""
        log_feats, aug_logits = self.log2feats(log_seqs, step)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # Compute prediction logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits, aug_logits

    def predict(self, log_seqs, item_indices):
        """Forward pass for inference (prediction)"""
        log_feats, _ = self.log2feats(log_seqs, 1)
        final_feat = log_feats[:, -1, :]  # Use last position's features
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
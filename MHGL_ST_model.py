import logging
import math
import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, HeteroConv
from torch_geometric.utils import softmax


class CustomGlobalAttention(nn.Module):
    """Global-attention pooling with explicit gating and stability guards."""

    def __init__(self, gate_nn: nn.Module, nn: Optional[nn.Module] = None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        if x.numel() == 0:
            return x.new_zeros((1, x.size(-1))), x.new_zeros((0,))

        gate = self.gate_nn(x).view(-1)
        gate = torch.nan_to_num(gate, nan=0.0, posinf=1e6, neginf=-1e6)
        attn = softmax(gate, batch)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        if self.nn is not None:
            x = self.nn(x)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled = torch.zeros(num_graphs, x.size(-1), device=x.device)
        pooled.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(-1)), x * attn.unsqueeze(1))
        return pooled, attn


class HierarchicalMultiModalGNN(nn.Module):
    """Shared stage-2 backbone for both survival and classification heads."""

    def __init__(
        self,
        cell_in_channels: int,
        gene_in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        out_channels: int,
        num_shared_clusters: int,
        gnn_type: str = 'Transformer',
        num_attention_heads: int = 1,
        dropout_rate: float = 0.5,
        num_intra_modal_layers: int = 1,
        num_inter_modal_layers: int = 1,
        num_time_bins: int = 5,
        cluster_temperature: float = 0.4,
        center_repulsion_weight: float = 1e-3,
        center_separation_scale: float = 0.5,
        assignment_balance_weight: float = 0.0,
        confidence_weight: float = 3e-2,
        balance_weight: float = 0.0,
        margin_weight: float = 5e-3,
        margin_delta: float = 0.20,
        center_update_alpha: float = 0.7,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_shared_clusters = num_shared_clusters
        self.dropout_rate = dropout_rate
        self.gnn_type = gnn_type
        self.num_attention_heads = num_attention_heads
        self.num_intra_modal_layers = num_intra_modal_layers
        self.num_inter_modal_layers = num_inter_modal_layers
        self.num_time_bins = num_time_bins
        self.cluster_temperature = cluster_temperature
        self.center_repulsion_weight = max(center_repulsion_weight, 0.0)
        self.center_separation_scale = max(center_separation_scale, 1e-3)
        self.assignment_balance_weight = max(assignment_balance_weight, 0.0)
        self.confidence_weight = max(confidence_weight, 0.0)
        self.balance_weight = max(balance_weight, 0.0)
        self.margin_weight = max(margin_weight, 0.0)
        self.margin_delta = max(margin_delta, 0.0)
        self.center_update_alpha = center_update_alpha
        self._hetero_edge_kw: Optional[str] = None

        # Intra-modal blocks
        self.intra_modal_convs = nn.ModuleList()
        self.intra_modal_norms = nn.ModuleList()
        for i in range(num_intra_modal_layers):
            conv_dict = {}
            cell_current_in = cell_in_channels if i == 0 else hidden_channels
            gene_current_in = gene_in_channels if i == 0 else hidden_channels
            current_out = embedding_dim if i == num_intra_modal_layers - 1 else hidden_channels

            cell_dim = (cell_current_in, cell_current_in) if gnn_type == 'Transformer' else cell_current_in
            gene_dim = (gene_current_in, gene_current_in) if gnn_type == 'Transformer' else gene_current_in

            if gnn_type == 'GCN':
                conv_dict[('cell', 'c_c', 'cell')] = GCNConv(cell_dim, current_out)
                conv_dict[('gene', 'g_g', 'gene')] = GCNConv(gene_dim, current_out)
            elif gnn_type == 'GAT':
                conv_dict[('cell', 'c_c', 'cell')] = GATConv(cell_dim, current_out // num_attention_heads, heads=num_attention_heads, dropout=dropout_rate)
                conv_dict[('gene', 'g_g', 'gene')] = GATConv(gene_dim, current_out // num_attention_heads, heads=num_attention_heads, dropout=dropout_rate)
            elif gnn_type == 'Transformer':
                conv_dict[('cell', 'c_c', 'cell')] = TransformerConv(cell_dim, current_out // num_attention_heads, heads=num_attention_heads, edge_dim=1, dropout=dropout_rate)
                conv_dict[('gene', 'g_g', 'gene')] = TransformerConv(gene_dim, current_out // num_attention_heads, heads=num_attention_heads, edge_dim=1, dropout=dropout_rate)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}.")
            self.intra_modal_convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.intra_modal_norms.append(nn.ModuleDict({
                'cell': nn.LayerNorm(current_out),
                'gene': nn.LayerNorm(current_out),
            }))

        # Inter-modal blocks
        self.inter_modal_convs = nn.ModuleList()
        self.inter_modal_norms = nn.ModuleList()
        for _ in range(num_inter_modal_layers):
            conv_dict = {}
            in_dim = embedding_dim
            cell_dim = (in_dim, in_dim) if gnn_type == 'Transformer' else in_dim
            gene_dim = (in_dim, in_dim) if gnn_type == 'Transformer' else in_dim
            if gnn_type == 'GCN':
                conv_dict[('cell', 'c_g', 'gene')] = GCNConv(in_dim, in_dim)
                conv_dict[('gene', 'g_c', 'cell')] = GCNConv(in_dim, in_dim)
            elif gnn_type == 'GAT':
                conv_dict[('cell', 'c_g', 'gene')] = GATConv(in_dim, in_dim // num_attention_heads, heads=num_attention_heads, dropout=dropout_rate, add_self_loops=False)
                conv_dict[('gene', 'g_c', 'cell')] = GATConv(in_dim, in_dim // num_attention_heads, heads=num_attention_heads, dropout=dropout_rate, add_self_loops=False)
            elif gnn_type == 'Transformer':
                conv_dict[('cell', 'c_g', 'gene')] = TransformerConv(cell_dim, in_dim // num_attention_heads, heads=num_attention_heads, edge_dim=1, dropout=dropout_rate)
                conv_dict[('gene', 'g_c', 'cell')] = TransformerConv(gene_dim, in_dim // num_attention_heads, heads=num_attention_heads, edge_dim=1, dropout=dropout_rate)
            self.inter_modal_convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.inter_modal_norms.append(nn.ModuleDict({
                'cell': nn.LayerNorm(in_dim),
                'gene': nn.LayerNorm(in_dim),
            }))

        if num_shared_clusters > 0:
            self.shared_cluster_centers = nn.Parameter(torch.randn(num_shared_clusters, embedding_dim))
        else:
            self.register_parameter('shared_cluster_centers', None)

        self.cell_attention_pool = CustomGlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2), nn.ReLU(), nn.Linear(embedding_dim // 2, 1))
        )
        self.gene_attention_pool = CustomGlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2), nn.ReLU(), nn.Linear(embedding_dim // 2, 1))
        )

        self.readout_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, out_channels),
        )

        self._reset_last_attention()

    def _reset_last_attention(self):
        self._last_cell_attn = None
        self._last_gene_attn = None
        self._last_cell_batch = None
        self._last_gene_batch = None
        self._last_cell_embeddings = None
        self._last_gene_embeddings = None
        self._last_cluster_stats: Optional[Dict[str, float]] = None

    def _apply_hetero_conv(self, conv_layer, feat_dict, edge_index_dict, edge_attr_dict):
        if not edge_attr_dict:
            return conv_layer(feat_dict, edge_index_dict)
        if self.gnn_type == 'GCN':
            weight_dict = {k: (v.squeeze(-1) if v.dim() > 1 else v) for k, v in edge_attr_dict.items()}
            try:
                return conv_layer(feat_dict, edge_index_dict, edge_weight_dict=weight_dict)
            except TypeError:
                return conv_layer(feat_dict, edge_index_dict, edge_weight=weight_dict)
        if self._hetero_edge_kw is None:
            try:
                out = conv_layer(feat_dict, edge_index_dict, edge_attr=edge_attr_dict)
                self._hetero_edge_kw = 'edge_attr'
                return out
            except ValueError as exc:
                if "_dict" in str(exc):
                    self._hetero_edge_kw = 'edge_attr_dict'
                else:
                    raise
        if self._hetero_edge_kw == 'edge_attr':
            return conv_layer(feat_dict, edge_index_dict, edge_attr=edge_attr_dict)
        return conv_layer(feat_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

    def _resolve_cluster_params(self, base_params: dict, override: Optional[dict]) -> dict:
        merged = dict(base_params or {})
        if override:
            merged.update(override)
        mapping = {
            'confidence_weight': 'w_conf',
            'balance_weight': 'w_bal',
            'margin_weight': 'w_margin',
        }
        resolved = {}
        for key, value in merged.items():
            resolved[mapping.get(key, key)] = value
        return resolved

    def _maybe_reinit_centers(self, embeddings: torch.Tensor):
        centers = self.shared_cluster_centers
        if centers is None or centers.numel() == 0:
            return
        with torch.no_grad():
            cent = centers.data
            norms = cent.norm(dim=1)
            bad_norm = norms < 1e-3
            too_close = torch.zeros_like(bad_norm)
            if cent.size(0) > 1:
                cent_unit = F.normalize(cent, p=2, dim=1)
                pair = torch.pdist(cent_unit)
                if pair.numel() > 0 and (pair.mean() < 0.10 or pair.min() < 0.05):
                    too_close[:] = True
            reset_mask = bad_norm | too_close
            idx = reset_mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                return
            if embeddings is not None and embeddings.numel() >= idx.numel() * cent.size(1):
                perm = torch.randperm(embeddings.size(0), device=cent.device)[: idx.numel()]
                repl = F.normalize(embeddings[perm], p=2, dim=1)
            else:
                repl = F.normalize(torch.randn(idx.numel(), cent.size(1), device=cent.device), p=2, dim=1)
            cent[idx] = repl
            self.shared_cluster_centers.data.copy_(cent)

    def _center_repulsion_penalty(self) -> torch.Tensor:
        centers = self.shared_cluster_centers
        if centers is None or centers.size(0) < 2:
            device = centers.device if centers is not None else 'cpu'
            return torch.tensor(0.0, device=device)
        centers = F.normalize(centers, p=2, dim=1)
        dist = torch.cdist(centers, centers, p=2)
        mask = torch.ones_like(dist, dtype=torch.bool)
        mask.fill_diagonal_(False)
        if not mask.any():
            return torch.tensor(0.0, device=centers.device)
        dist_vals = dist[mask]
        return torch.exp(-dist_vals / self.center_separation_scale).mean()

    def _compute_clustering_loss_fn(
        self,
        embeddings: torch.Tensor,
        params: dict,
        update_centers: bool,
        tag: str,
    ) -> torch.Tensor:
        if embeddings.size(0) == 0 or self.shared_cluster_centers is None or self.shared_cluster_centers.numel() == 0:
            device = embeddings.device if embeddings.numel() > 0 else (self.shared_cluster_centers.device if self.shared_cluster_centers is not None else 'cpu')
            return torch.tensor(0.0, device=device)

        temperature = params.get('temperature', self.cluster_temperature)
        w_conf = params.get('w_conf', self.confidence_weight)
        w_bal = params.get('w_bal', self.assignment_balance_weight + self.balance_weight)
        w_margin = params.get('w_margin', self.margin_weight)
        margin_delta = params.get('margin_delta', self.margin_delta)
        alpha = params.get('alpha', self.center_update_alpha)

        embed_proc = embeddings
        embed_norms = embed_proc.norm(dim=1)
        if torch.all(embed_norms < 1e-6):
            embed_proc = embed_proc + 1e-2 * torch.randn_like(embed_proc)
        centers_norm = F.normalize(self.shared_cluster_centers, p=2, dim=1)
        distances = torch.cdist(embed_proc, centers_norm.detach(), p=2)
        logits = -distances / max(temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values
        Q = torch.softmax(logits, dim=1)
        if not torch.isfinite(Q).all():
            K = max(Q.size(1), 1)
            Q = torch.nan_to_num(Q, nan=1.0 / K).clamp_min(1e-6)
            Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-6)

        eps = 1e-8
        Q_numerator = Q ** 2
        Q_denominator = torch.sum(Q, dim=0, keepdim=True) + eps
        P_unnormalized = Q_numerator / Q_denominator
        P = P_unnormalized / (torch.sum(P_unnormalized, dim=1, keepdim=True) + eps)
        loss = torch.sum(P * torch.log((P + eps) / (Q + eps)), dim=-1).mean()

        entropy = None
        if w_conf > 0:
            entropy = -(Q.clamp_min(eps) * Q.clamp_min(eps).log()).sum(dim=1).mean()
            loss = loss + w_conf * entropy

        balance_kl = None
        if w_bal > 0:
            uniform = torch.full((1, Q.size(1)), 1.0 / Q.size(1), device=Q.device)
            q_mean = Q.mean(dim=0, keepdim=True)
            balance_kl = (q_mean * (q_mean / uniform).log()).sum()
            loss = loss + w_bal * balance_kl

        loss_margin = None
        if w_margin > 0 and Q.size(1) > 1:
            top2 = distances.topk(k=2, largest=False)
            d1 = top2.values[:, 0]
            d2 = top2.values[:, 1]
            loss_margin = F.relu(margin_delta + d1 - d2).mean()
            loss = loss + w_margin * loss_margin

        assign_conf = Q.max(dim=1).values.mean().item()
        logits_std = float(logits.std(dim=1).mean().detach())
        stats = {
            'entropy': float(entropy.detach()) if entropy is not None else None,
            'balance_kl': float(balance_kl.detach()) if balance_kl is not None else None,
            'margin': float(loss_margin.detach()) if loss_margin is not None else None,
            'logits_std': logits_std,
            'assign_conf': assign_conf,
        }
        self._last_cluster_stats = stats

        conf_th = (1.0 / max(Q.size(1), 1))
        std_th = 5e-2
        should_update = update_centers and (assign_conf >= conf_th) and (logits_std >= std_th)

        if should_update:
            with torch.no_grad():
                Q_sum = Q.sum(dim=0, keepdim=True) + eps
                new_centers = torch.mm(Q.T, F.normalize(embed_proc, p=2, dim=1)) / Q_sum.T
                blended = alpha * self.shared_cluster_centers.data + (1 - alpha) * new_centers
                self.shared_cluster_centers.data = F.normalize(blended, p=2, dim=1)
                self._maybe_reinit_centers(embed_proc)
                stats['center_delta'] = float((self.shared_cluster_centers.data - blended).norm(dim=1).mean().item())
        else:
            stats['center_delta'] = 0.0
        return loss

    def forward(
        self,
        data: HeteroData,
        compute_clustering_loss: bool = True,
        clustering_loss_params: Optional[dict] = None,
        update_centers: bool = True,
    ):
        self._reset_last_attention()
        device = data.device if hasattr(data, 'device') else (
            data['cell'].x.device if 'cell' in data.node_types and hasattr(data['cell'], 'x') else 'cpu'
        )

        cell_input_channel_size = self.intra_modal_convs[0].convs[('cell', 'c_c', 'cell')].in_channels
        if isinstance(cell_input_channel_size, tuple):
            cell_input_channel_size = cell_input_channel_size[0]
        gene_input_channel_size = self.intra_modal_convs[0].convs[('gene', 'g_g', 'gene')].in_channels
        if isinstance(gene_input_channel_size, tuple):
            gene_input_channel_size = gene_input_channel_size[0]

        x_dict = {}
        if 'cell' in data.node_types and hasattr(data['cell'], 'x') and data['cell'].x is not None:
            x_dict['cell'] = data['cell'].x
        else:
            x_dict['cell'] = torch.empty((0, cell_input_channel_size), device=device)
        if 'gene' in data.node_types and hasattr(data['gene'], 'x') and data['gene'].x is not None:
            x_dict['gene'] = data['gene'].x
        else:
            x_dict['gene'] = torch.empty((0, gene_input_channel_size), device=device)

        # Intra-modal propagation
        h_intra = x_dict
        for i, conv in enumerate(self.intra_modal_convs):
            edge_attr_dict = {}
            if ('cell', 'c_c', 'cell') in data.edge_types and hasattr(data['cell', 'c_c', 'cell'], 'edge_attr'):
                edge_attr_dict[('cell', 'c_c', 'cell')] = data['cell', 'c_c', 'cell'].edge_attr
            if ('gene', 'g_g', 'gene') in data.edge_types and hasattr(data['gene', 'g_g', 'gene'], 'edge_attr'):
                edge_attr_dict[('gene', 'g_g', 'gene')] = data['gene', 'g_g', 'gene'].edge_attr
            out = self._apply_hetero_conv(conv, h_intra, data.edge_index_dict, edge_attr_dict)
            norms = self.intra_modal_norms[i]
            new_dict = {}
            for ntype, tensor in out.items():
                if tensor.numel() > 0:
                    new_dict[ntype] = F.dropout(norms[ntype](F.relu(tensor)), p=self.dropout_rate, training=self.training)
            for ntype in h_intra:
                if ntype not in new_dict:
                    new_dict[ntype] = h_intra[ntype]
            h_intra = new_dict

        # Inter-modal propagation
        h_inter = {k: v.clone() if v.numel() > 0 else v for k, v in h_intra.items()}
        for i, conv in enumerate(self.inter_modal_convs):
            edge_attr_dict = {}
            if ('cell', 'c_g', 'gene') in data.edge_types and hasattr(data['cell', 'c_g', 'gene'], 'edge_attr'):
                edge_attr_dict[('cell', 'c_g', 'gene')] = data['cell', 'c_g', 'gene'].edge_attr
            if ('gene', 'g_c', 'cell') in data.edge_types and hasattr(data['gene', 'g_c', 'cell'], 'edge_attr'):
                edge_attr_dict[('gene', 'g_c', 'cell')] = data['gene', 'g_c', 'cell'].edge_attr
            hout = self._apply_hetero_conv(conv, h_inter, data.edge_index_dict, edge_attr_dict)
            norms = self.inter_modal_norms[i]
            for ntype, tensor in hout.items():
                if tensor.numel() > 0:
                    base = h_inter.get(ntype)
                    z = base + tensor if (base is not None and base.shape == tensor.shape) else tensor
                    h_inter[ntype] = F.dropout(norms[ntype](F.relu(z)), p=self.dropout_rate, training=self.training)
        fused_cell_embeddings = h_inter.get('cell', torch.empty((0, self.embedding_dim), device=device))
        fused_gene_embeddings = h_inter.get('gene', torch.empty((0, self.embedding_dim), device=device))

        num_graphs = data.num_graphs if hasattr(data, 'num_graphs') and data.num_graphs is not None else 1
        cell_batch = data['cell'].batch if 'cell' in data.node_types and hasattr(data['cell'], 'batch') and data['cell'].batch is not None else torch.zeros(fused_cell_embeddings.size(0), dtype=torch.long, device=device)
        gene_batch = data['gene'].batch if 'gene' in data.node_types and hasattr(data['gene'], 'batch') and data['gene'].batch is not None else torch.zeros(fused_gene_embeddings.size(0), dtype=torch.long, device=device)

        graph_embedding_cells = torch.zeros(num_graphs, self.embedding_dim, device=device)
        cell_attention_scores = None
        if fused_cell_embeddings.size(0) > 0 and cell_batch.numel() == fused_cell_embeddings.size(0):
            graph_embedding_cells, cell_attention_scores = self.cell_attention_pool(fused_cell_embeddings, cell_batch)

        graph_embedding_genes = torch.zeros(num_graphs, self.embedding_dim, device=device)
        gene_attention_scores = None
        if fused_gene_embeddings.size(0) > 0 and gene_batch.numel() == fused_gene_embeddings.size(0):
            graph_embedding_genes, gene_attention_scores = self.gene_attention_pool(fused_gene_embeddings, gene_batch)

        self._last_cell_attn = cell_attention_scores.detach() if cell_attention_scores is not None else None
        self._last_gene_attn = gene_attention_scores.detach() if gene_attention_scores is not None else None
        self._last_cell_batch = cell_batch.detach() if cell_attention_scores is not None else None
        self._last_gene_batch = gene_batch.detach() if gene_attention_scores is not None else None
        self._last_cell_embeddings = fused_cell_embeddings.detach() if fused_cell_embeddings.numel() > 0 else None
        self._last_gene_embeddings = fused_gene_embeddings.detach() if fused_gene_embeddings.numel() > 0 else None

        fused_graph_embedding = torch.cat([graph_embedding_cells, graph_embedding_genes], dim=-1)
        logits = self.readout_mlp(fused_graph_embedding)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        total_cluster_loss = torch.tensor(0.0, device=device)
        if compute_clustering_loss and self.shared_cluster_centers is not None and self.shared_cluster_centers.numel() > 0:
            params = clustering_loss_params or {}
            base_defaults = {
                'temperature': self.cluster_temperature,
                'confidence_weight': self.confidence_weight,
                'balance_weight': self.assignment_balance_weight + self.balance_weight,
                'margin_weight': self.margin_weight,
                'margin_delta': self.margin_delta,
                'alpha': self.center_update_alpha,
            }
            base_params = self._resolve_cluster_params(base_defaults, params.get('base'))
            cell_params = self._resolve_cluster_params(base_params, params.get('cell'))
            gene_params = self._resolve_cluster_params(base_params, params.get('gene'))
            total_cluster_loss = torch.tensor(0.0, device=device)
            if fused_cell_embeddings.size(0) > 0:
                total_cluster_loss = total_cluster_loss + self._compute_clustering_loss_fn(
                    fused_cell_embeddings, cell_params, update_centers, 'cell'
                )
            if fused_gene_embeddings.size(0) > 0:
                gene_stride = max(1, int(params.get('gene_update_stride', 1)))
                allow_gene_update = update_centers and (self.training is False or (self.training and getattr(self, '_cluster_epoch', 0) % gene_stride == 0))
                total_cluster_loss = total_cluster_loss + self._compute_clustering_loss_fn(
                    fused_gene_embeddings, gene_params, allow_gene_update, 'gene'
                )
            if compute_clustering_loss and self.center_repulsion_weight > 0:
                total_cluster_loss = total_cluster_loss + self.center_repulsion_weight * self._center_repulsion_penalty()
        return logits, total_cluster_loss


class GraphAugmentations:
    """Shared geometric augmentations for heterogeneous graphs."""

    def __init__(self, jitter_strength: float = 0.0, rotation_prob: float = 0.2, flip_prob: float = 0.2, update_edge_attr: bool = False):
        self.jitter_strength = float(jitter_strength)
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.update_edge_attr = update_edge_attr
        if self.jitter_strength != 0.0 and not update_edge_attr:
            logging.info("GraphAugmentations: jitter is enabled but edge attributes are not updated; jitter may be ignored.")

    def __call__(self, data: HeteroData) -> HeteroData:
        new_data = data.clone()
        pos_stores = [store for store in new_data.node_stores if hasattr(store, 'pos')]
        if not pos_stores:
            return new_data

        center = torch.cat([s.pos for s in pos_stores], dim=0).mean(dim=0)
        device = center.device
        dtype = center.dtype

        if random.random() < self.rotation_prob:
            angle = random.uniform(0.0, 360.0) * math.pi / 180.0
            angle = torch.tensor(angle, dtype=dtype, device=device)
            c, s = torch.cos(angle), torch.sin(angle)
            rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
            for store in pos_stores:
                store.pos = (store.pos - center) @ rot.T + center

        if random.random() < self.flip_prob:
            for store in pos_stores:
                store.pos[:, 0] = 2 * center[0] - store.pos[:, 0]

        if self.jitter_strength != 0.0:
            for store in pos_stores:
                store.pos = store.pos + torch.randn_like(store.pos) * self.jitter_strength

        if self.update_edge_attr:
            for store in new_data.edge_stores:
                if 'edge_index' not in store:
                    continue
                src_type, _, dst_type = store._key
                if not hasattr(new_data[src_type], 'pos') or not hasattr(new_data[dst_type], 'pos'):
                    continue
                src_pos = new_data[src_type].pos
                dst_pos = new_data[dst_type].pos
                row, col = store.edge_index
                new_dist = (src_pos[row] - dst_pos[col]).pow(2).sum(dim=-1).sqrt()
                edge_attr_raw = new_dist.unsqueeze(1)
                min_d, max_d = edge_attr_raw.min(), edge_attr_raw.max()
                if float(max_d - min_d) > 1e-6:
                    store.edge_attr = 1.0 - ((edge_attr_raw - min_d) / (max_d - min_d))
                else:
                    store.edge_attr = torch.ones_like(edge_attr_raw) * 0.5
        return new_data

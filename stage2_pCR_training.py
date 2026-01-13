import argparse
import logging
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial import Delaunay
from openslide import OpenSlide
from tqdm import tqdm
import traceback
import random
import matplotlib.pyplot as plt # 导入 matplotlib
from sklearn.model_selection import StratifiedShuffleSplit 
import gc

from MHGL_ST_model import HierarchicalMultiModalGNN, GraphAugmentations


def _set_global_seed(seed: int = 233) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stage 2 pCR predictor on multi-modal graphs.")
    parser.add_argument("--root-dir", type=Path, required=True, help="Directory to store processed graphs and checkpoints.")
    parser.add_argument("--cell-output-dir", type=Path, required=True, help="Directory containing CellViT outputs (per-sample subdirs).")
    parser.add_argument("--gene-dir", type=Path, required=True, help="Directory containing predicted gene expression CSVs.")
    parser.add_argument("--svs-dir", type=Path, required=True, help="Directory containing raw SVS files.")
    parser.add_argument("--label-file", type=Path, required=True, help="CSV file containing Patient/Responder annotations.")
    parser.add_argument("--patch-size-level0", type=int, default=512, help="Patch size (level-0 pixels) used when building graphs.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction (0 disables validation).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for PyG DataLoader instances.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension inside the GNN.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension for the readout head.")
    parser.add_argument("--num-shared-clusters", type=int, default=5, help="Number of shared archetype clusters.")
    parser.add_argument("--gnn-type", choices=["Transformer", "GAT", "GCN"], default="Transformer", help="Intra-modal GNN backbone.")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads for Transformer/GAT layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate inside the model.")
    parser.add_argument("--num-intra-layers", type=int, default=3, help="Number of intra-modal layers.")
    parser.add_argument("--num-inter-layers", type=int, default=2, help="Number of inter-modal fusion layers.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--cluster-lambda", type=float, default=0.05, help="Weight for the clustering loss term.")
    parser.add_argument("--device", default="auto", help="Torch device spec (e.g., 'cuda:0', 'cpu', or 'auto').")
    parser.add_argument("--seed", type=int, default=233, help="Random seed for data splits and initialization.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory to store checkpoints.")
    parser.add_argument("--save-prefix", default="stage2_pcr", help="Prefix for saved checkpoint files.")
    return parser.parse_args()




class ProcessedImageDataset(InMemoryDataset):
    def __init__(self, root, nucle_root, gene_root, svs_root, label_root, original_patch_size_level0=512, transform=None, pre_transform=None):
        self.nucle_raw_root = nucle_root
        self.gene_raw_root = gene_root
        self.svs_raw_root = svs_root
        self.original_patch_size_level0 = original_patch_size_level0
        self.labels = pd.read_csv(label_root)
        self.type_encoder = OneHotEncoder(sparse_output=False, categories=[['Connective', 'Neoplastic', 'Dead', 'Epithelial', 'Inflammatory']])

        print(f"DEBUG(init): Dataset root: {root}")

        super().__init__(root, transform, pre_transform)

        processed_file_path = self.processed_paths[0]
        print( f"DEBUG(init): Processed file path: {processed_file_path}" )
        if os.path.exists(processed_file_path):
            try:
                loaded_content = torch.load(processed_file_path)
                if isinstance(loaded_content, list):
                    self.data_list = loaded_content
                    if self.data_list: # Ensure self.data and self.slices are set if data_list is used
                         self.data, self.slices = self.collate(self.data_list)
                    else:
                         self.data, self.slices = self.collate([]) # Handle empty list case for collate
                else:
                    self.data, self.slices = loaded_content
                    # Reconstruct data_list if needed for __getitem__ / __len__ if they use it
                    num_graphs = 0
                    if self.slices:
                        for key in self.slices: # Determine num_graphs from slices
                            if isinstance(self.slices[key], torch.Tensor) and self.slices[key].ndim == 1:
                                num_graphs = len(self.slices[key]) -1
                                break
                    self.data_list = [super(ProcessedImageDataset, self).get(i) for i in range(num_graphs)]

                print(f"DEBUG(init): Successfully loaded data. Graphs: {len(self)}.")
            except Exception as e:
                print(f"ERROR(init): Failed to load data from {processed_file_path}: {e}")
                self.data_list = []
                self.data, self.slices = self.collate([]) # Ensure these are initialized
        else:
            self.data_list = []
            self.data, self.slices = self.collate([])
            print("DEBUG(init): Processed data file does NOT exist. process() should have run.")
        print(f"DEBUG(init): Final length of dataset: {len(self)}")


    @property
    def raw_file_names(self):
        svs_files_found = glob.glob(os.path.join(self.svs_raw_root, "*.svs"))
        sample_names = [os.path.basename(f).replace(".svs", "") for f in svs_files_found]
        # print(f"DEBUG(raw_file_names): SVS files found by glob: {svs_files_found}")
        # print(f"DEBUG(raw_file_names): Derived sample names: {sample_names}")
        return sample_names

    @property
    def processed_file_names(self):
        return ['processed_graphs_final_vis.pt'] # 更新文件名

    def process(self):
        print(f"DEBUG(process): Entering process method.")
        raw_names = self.raw_file_names
        print(f"DEBUG(process): Raw file names for processing: {raw_names}")

        if not raw_names:
            print("WARNING(process): No raw files. Saving empty data.")
            # For InMemoryDataset, save expects collated data.
            data, slices = self.collate([])
            torch.save((data, slices), self.processed_paths[0])
            return

        all_graph_data_list = []
        for sample_idx, sample_name in enumerate(tqdm(raw_names, desc="Processing Samples")):
            svs_path = os.path.join(self.svs_raw_root, sample_name + ".svs")
            nucle_path = os.path.join(self.nucle_raw_root, sample_name, "full_slide_nuclei_features_sliding_window_debug", "all_nuclei_features_full_slide_sliding_window_robust_scaled.csv")
            gene_path = os.path.join(self.gene_raw_root, sample_name + ".csv")

            if not (os.path.exists(svs_path) and os.path.exists(nucle_path) and os.path.exists(gene_path)):
                print(f"WARNING: Missing files for {sample_name}. Skipping.")
                continue
            try:
                slide = OpenSlide(svs_path)
                level = 1
                downsample_factor = slide.level_downsamples[level]
                slide.close()
                patch_size_on_level1 = self.original_patch_size_level0 / downsample_factor
                radius_threshold_for_fusion = 2.5 * patch_size_on_level1

                nucle_data = pd.read_csv(nucle_path)
                y_coords_global_cells = nucle_data["Identifier.CentoidX_Global"].values
                x_coords_global_cells = nucle_data["Identifier.CentoidY_Global"].values
                pos_cells = torch.tensor(np.vstack((x_coords_global_cells / downsample_factor, y_coords_global_cells / downsample_factor)).T, dtype=torch.float)
                if 'type' in nucle_data.columns:
                    cell_types_encoded = self.type_encoder.fit_transform(nucle_data[['type']])
                    cell_types_tensor = torch.tensor(cell_types_encoded, dtype=torch.float)
                else:
                    cell_types_tensor = torch.empty((len(nucle_data), 0), dtype=torch.float)
                columns_to_drop_cells = ["Identifier.CentoidX_Global", "Identifier.CentoidY_Global", "Global_Nuclei_ID", "type", 'Shape.HuMoments4', 'Shape.WeightedHuMoments4', "Shape.HuMoments5",
                                         'Shape.HuMoments6', 'Shape.WeightedHuMoments6', 'Shape.HuMoments7', 'Shape.WeightedHuMoments3', 'Shape.HuMoments3', 'Shape.HuMoments2', 'Shape.WeightedHuMoments2']
                cell_features_df = nucle_data.drop(columns=columns_to_drop_cells, errors='ignore')
                for col in cell_features_df.columns: cell_features_df[col] = pd.to_numeric(cell_features_df[col], errors='coerce').astype(float)
                cell_features_df = cell_features_df.fillna(0.0)

                # print( cell_features_df.values.min(), cell_features_df.values.max(), cell_features_df.columns[(cell_features_df == cell_features_df.values.max()).any()] )

                x_cells_base = torch.tensor(cell_features_df.values, dtype=torch.float)
                x_cells = torch.cat([x_cells_base, cell_types_tensor], dim=1)

                gene_data = pd.read_csv(gene_path)
                gene_coords_col_name = gene_data.columns[0]
                gene_coords_parsed = gene_data[gene_coords_col_name].str.split('_', expand=True).apply(pd.to_numeric, errors='coerce').fillna(0.0)
                y_gene_coords = gene_coords_parsed.iloc[:,0].values if gene_coords_parsed.shape[1] >=1 else np.zeros(len(gene_data))
                x_gene_coords = gene_coords_parsed.iloc[:,1].values if gene_coords_parsed.shape[1] >=2 else np.zeros(len(gene_data))
                pos_genes = torch.tensor(np.vstack((x_gene_coords, y_gene_coords)).T, dtype=torch.float)
                gene_features_df = gene_data.drop(columns=[gene_coords_col_name])
                for col in gene_features_df.columns: gene_features_df[col] = pd.to_numeric(gene_features_df[col], errors='coerce').astype(float)
                gene_features_df = gene_features_df.fillna(0.0)
                x_genes = torch.tensor(gene_features_df.values, dtype=torch.float)

                edge_index_cells, edge_attr_cells = self._create_delaunay_edges(pos_cells)
                edge_index_genes, edge_attr_genes = self._create_delaunay_edges(pos_genes)
                edge_index_c_g, edge_attr_c_g, edge_index_g_c, edge_attr_g_c = self._create_radius_edges(pos_cells, pos_genes, radius_threshold_for_fusion)

                hetero_data = HeteroData()
                if x_cells.numel() > 0 and pos_cells.shape[0] > 0 :
                    hetero_data['cell'].x = x_cells
                    hetero_data['cell'].pos = pos_cells
                if x_genes.numel() > 0 and pos_genes.shape[0] > 0 :
                    hetero_data['gene'].x = x_genes
                    hetero_data['gene'].pos = pos_genes
                if edge_index_cells.numel() > 0 : hetero_data['cell', 'c_c', 'cell'].edge_index = edge_index_cells
                if edge_attr_cells.numel() > 0 : hetero_data['cell', 'c_c', 'cell'].edge_attr = edge_attr_cells
                if edge_index_genes.numel() > 0 : hetero_data['gene', 'g_g', 'gene'].edge_index = edge_index_genes
                if edge_attr_genes.numel() > 0 : hetero_data['gene', 'g_g', 'gene'].edge_attr = edge_attr_genes
                if edge_index_c_g.numel() > 0 : hetero_data['cell', 'c_g', 'gene'].edge_index = edge_index_c_g
                if edge_attr_c_g.numel() > 0 : hetero_data['cell', 'c_g', 'gene'].edge_attr = edge_attr_c_g
                if edge_index_g_c.numel() > 0 : hetero_data['gene', 'g_c', 'cell'].edge_index = edge_index_g_c
                if edge_attr_g_c.numel() > 0 : hetero_data['gene', 'g_c', 'cell'].edge_attr = edge_attr_g_c

                if self.labels.loc[self.labels["Patient"] == sample_name, "Responder"].iloc[0] == "responder":
                    hetero_data.y = 1
                else:
                    hetero_data.y = 0
                print(sample_name, hetero_data.y)

                hetero_data.misc_info = {'radius_threshold': radius_threshold_for_fusion, 'sample_name': sample_name}

                valid_nodes_exist = any(hetero_data[nt].num_nodes > 0 for nt in hetero_data.node_types if hasattr(hetero_data[nt], 'num_nodes'))
                if not valid_nodes_exist:
                    print(f"WARNING: No valid nodes for {sample_name}. Skipping.")
                    continue
                all_graph_data_list.append(hetero_data)
            except Exception as e:
                print(f"ERROR processing {sample_name}: {e}")
                traceback.print_exc()
                continue

        if self.pre_filter is not None:
            all_graph_data_list = [data for data in all_graph_data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            all_graph_data_list = [self.pre_transform(data) for data in all_graph_data_list]

        # InMemoryDataset's save method expects the collated (data, slices) tuple.
        data, slices = self.collate(all_graph_data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"DEBUG(process): Successfully collated and saved {len(all_graph_data_list)} graphs to {self.processed_paths[0]}")
        
    def _create_delaunay_edges(self, pos_nodes):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        if len(pos_nodes) >= 3:
            try:
                tri = Delaunay(pos_nodes.numpy(), qhull_options="QJ")
                undirected_edges = set()
                for simplex in tri.simplices:
                    edges_in_simplex = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
                    for u, v in edges_in_simplex:
                        undirected_edges.add(tuple(sorted((u, v))))
                if not undirected_edges: return edge_index, edge_attr
                final_edges_list, final_distances_list = [], []
                for u, v in undirected_edges:
                    dist = torch.norm(pos_nodes[u] - pos_nodes[v], p=2)
                    final_edges_list.extend([(u, v), (v, u)])
                    final_distances_list.extend([dist.item(), dist.item()])
                edge_index = torch.tensor(final_edges_list, dtype=torch.long).T
                edge_attr_raw = torch.tensor(final_distances_list, dtype=torch.float).unsqueeze(1)
                if edge_attr_raw.numel() > 0:
                    min_dist, max_dist = edge_attr_raw.min(), edge_attr_raw.max()
                    if max_dist > min_dist: edge_attr = 1.0 - ((edge_attr_raw - min_dist) / (max_dist - min_dist))
                    else: edge_attr = torch.ones_like(edge_attr_raw) * 0.5
            except Exception as e: print(f"ERROR: Delaunay failed for shape {pos_nodes.shape}: {e}")
        return edge_index, edge_attr


    def _create_radius_edges(self, pos_cells, pos_genes, radius_threshold):
        edge_index_c_g, edge_attr_c_g, edge_index_g_c, edge_attr_g_c = torch.empty((2,0),dtype=torch.long), torch.empty((0,1),dtype=torch.float), torch.empty((2,0),dtype=torch.long), torch.empty((0,1),dtype=torch.float)
        if len(pos_cells) > 0 and len(pos_genes) > 0:
            dist_matrix = torch.cdist(pos_cells, pos_genes, p=2)
            adj = dist_matrix <= radius_threshold
            cell_indices, gene_indices = adj.nonzero(as_tuple=True)
            if cell_indices.numel() > 0:
                edge_index_c_g = torch.stack([cell_indices, gene_indices], dim=0)
                edge_attr_c_g_raw = dist_matrix[cell_indices, gene_indices].unsqueeze(1)
                if edge_attr_c_g_raw.numel() > 0:
                    min_d, max_d = edge_attr_c_g_raw.min(), edge_attr_c_g_raw.max()
                    if max_d > min_d: edge_attr_c_g = 1.0 - ((edge_attr_c_g_raw - min_d) / (max_d - min_d))
                    else: edge_attr_c_g = torch.ones_like(edge_attr_c_g_raw) * 0.5
                edge_index_g_c = torch.stack([gene_indices, cell_indices], dim=0)
                edge_attr_g_c = edge_attr_c_g
        return edge_index_c_g, edge_attr_c_g, edge_index_g_c, edge_attr_g_c


    # Standard InMemoryDataset len and get
    def len(self):
        # Correct way to get length for InMemoryDataset
        if hasattr(self, 'slices') and self.slices is not None:
            # Use a common attribute like 'y' if it exists for all graphs, or any other reliable slice
            if 'y' in self.slices:
                 return len(self.slices['y']) - 1
            # Fallback: iterate through slices to find a representative one
            for key in self.slices:
                 if isinstance(self.slices[key], torch.Tensor) and self.slices[key].ndim == 1:
                     return len(self.slices[key]) -1
        return 0 # Should not happen if data is loaded/processed correctly

    def get(self, idx):
        # Standard way to get item for InMemoryDataset
        return super().get(idx)



def infer_feature_dims(samples, default_cell=74, default_gene=215):
    cell_dim = 0
    gene_dim = 0
    for sample in samples:
        cell_x = sample['cell'].x if 'cell' in sample.node_types and hasattr(sample['cell'], 'x') else None
        gene_x = sample['gene'].x if 'gene' in sample.node_types and hasattr(sample['gene'], 'x') else None
        if cell_x is not None and cell_x.numel() > 0:
            cell_dim = cell_x.shape[1] if cell_x.dim() > 1 else 1
        if gene_x is not None and gene_x.numel() > 0:
            gene_dim = gene_x.shape[1] if gene_x.dim() > 1 else 1
        if cell_dim and gene_dim:
            break
    return (cell_dim or default_cell), (gene_dim or default_gene)


def train_epoch(model, loader, optimizer, criterion_primary, aug, lambda_clust, device):
    model.train()
    total_loss = total_primary_loss = total_cluster_loss = 0.0
    total_graphs = 0
    all_preds, all_labels = [], []
    for data in tqdm(loader, desc="Training", leave=False):
        if aug is not None:
            data = aug(data)
        data = data.to(device)
        if not hasattr(data, 'y') or data.y is None or data.y.numel() == 0:
            continue
        labels = data.y.view(-1, 1).float()
        predictions, cluster_loss = model(data)
        if predictions.shape[0] != labels.shape[0]:
            continue
        primary_loss = criterion_primary(predictions, labels)
        loss = primary_loss + lambda_clust * cluster_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        graphs = int(getattr(data, 'num_graphs', labels.size(0)))
        total_graphs += graphs
        total_loss += loss.item() * graphs
        total_primary_loss += primary_loss.item() * graphs
        total_cluster_loss += cluster_loss.item() * graphs
        all_preds.append(torch.sigmoid(predictions).detach().cpu())
        all_labels.append(labels.detach().cpu())
    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    denom = max(total_graphs, 1)
    avg_loss = total_loss / denom
    avg_primary_loss = total_primary_loss / denom
    avg_cluster_loss = total_cluster_loss / denom
    preds_np = torch.cat(all_preds).numpy().flatten()
    labels_np = torch.cat(all_labels).numpy().flatten()
    try:
        auc = roc_auc_score(labels_np, preds_np)
        acc = accuracy_score(labels_np, (preds_np > 0.5).astype(int))
    except ValueError:
        auc, acc = 0.0, 0.0
    return avg_loss, avg_primary_loss, avg_cluster_loss, acc, auc


@torch.no_grad()
def evaluate(model, loader, criterion_primary, lambda_clust, device):
    model.eval()
    total_loss = total_primary_loss = total_cluster_loss = 0.0
    total_graphs = 0
    all_preds, all_labels = [], []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = data.to(device)
        if not hasattr(data, 'y') or data.y is None or data.y.numel() == 0:
            continue
        labels = data.y.view(-1, 1).float()
        predictions, cluster_loss = model(data, compute_clustering_loss=True)
        if predictions.shape[0] != labels.shape[0]:
            continue
        primary_loss = criterion_primary(predictions, labels)
        loss = primary_loss + lambda_clust * cluster_loss
        graphs = int(getattr(data, 'num_graphs', labels.size(0)))
        total_graphs += graphs
        total_loss += loss.item() * graphs
        total_primary_loss += primary_loss.item() * graphs
        total_cluster_loss += cluster_loss.item() * graphs
        all_preds.append(torch.sigmoid(predictions).cpu())
        all_labels.append(labels.cpu())
    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    denom = max(total_graphs, 1)
    avg_loss = total_loss / denom
    avg_primary_loss = total_primary_loss / denom
    avg_cluster_loss = total_cluster_loss / denom
    preds_np = torch.cat(all_preds).numpy().flatten()
    labels_np = torch.cat(all_labels).numpy().flatten()
    try:
        auc = roc_auc_score(labels_np, preds_np)
        acc = accuracy_score(labels_np, (preds_np > 0.5).astype(int))
    except ValueError:
        auc, acc = 0.0, 0.0
    return avg_loss, avg_primary_loss, avg_cluster_loss, acc, auc


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='[%(asctime)s] %(levelname)s - %(message)s')
    _set_global_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    root_dir = args.root_dir.expanduser()
    cell_dir = args.cell_output_dir.expanduser()
    gene_dir = args.gene_dir.expanduser()
    svs_dir = args.svs_dir.expanduser()
    label_file = args.label_file.expanduser()
    checkpoint_dir = args.checkpoint_dir.expanduser()

    root_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = ProcessedImageDataset(
        root=str(root_dir),
        nucle_root=str(cell_dir),
        gene_root=str(gene_dir),
        label_root=str(label_file),
        svs_root=str(svs_dir),
        original_patch_size_level0=args.patch_size_level0,
    )
    if len(dataset) == 0:
        logging.error("Dataset is empty. Exiting.")
        return

    all_items = [dataset.get(i) for i in range(len(dataset))]
    try:
        all_targets = np.array([
            int(item['y'].item()) if isinstance(item['y'], torch.Tensor) else int(item['y'])
            for item in all_items
        ], dtype=np.int64)
    except Exception as exc:
        logging.error("Could not extract targets from dataset: %s", exc)
        return

    train_list, val_list = all_items, []
    if args.val_split > 0 and len(all_items) > 1 and len(np.unique(all_targets)) > 1:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
        train_idx, val_idx = next(sss.split(np.zeros_like(all_targets), all_targets))
        train_list = [all_items[i] for i in train_idx]
        val_list = [all_items[i] for i in val_idx]
    elif args.val_split > 0:
        logging.warning("Validation split requested but dataset is too small or single-class; using all samples for training.")

    if not train_list:
        logging.error("Training list is empty. Exiting.")
        return

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_list, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_list else None
    aug = GraphAugmentations(jitter_strength=1.0, update_edge_attr=True)

    cell_dim, gene_dim = infer_feature_dims(train_list)
    logging.info("Detected feature dims | cell: %d | gene: %d", cell_dim, gene_dim)

    model = HierarchicalMultiModalGNN(
        cell_dim,
        gene_dim,
        args.hidden_dim,
        args.embedding_dim,
        1,
        args.num_shared_clusters,
        args.gnn_type,
        args.num_heads,
        args.dropout,
        args.num_intra_layers,
        args.num_inter_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion_primary = nn.BCEWithLogitsLoss()
    best_auc = -float('inf')
    best_checkpoint = None

    logging.info("Starting training for %d epochs...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_p_loss, train_c_loss, train_acc, train_auc = train_epoch(
            model, train_loader, optimizer, criterion_primary, aug, args.cluster_lambda, device
        )
        logging.info(
            "Epoch %03d | Train Loss %.4f (P: %.4f C: %.4f) | Acc %.4f | AUC %.4f",
            epoch, train_loss, train_p_loss, train_c_loss, train_acc, train_auc,
        )

        if val_loader:
            val_loss, val_p_loss, val_c_loss, val_acc, val_auc = evaluate(
                model, val_loader, criterion_primary, args.cluster_lambda, device
            )
            logging.info(
                "Epoch %03d | Val Loss %.4f (P: %.4f C: %.4f) | Acc %.4f | AUC %.4f",
                epoch, val_loss, val_p_loss, val_c_loss, val_acc, val_auc,
            )
            if val_auc > best_auc:
                best_auc = val_auc
                best_checkpoint = checkpoint_dir / f"{args.save_prefix}_epoch{epoch:03d}_auc{val_auc:.4f}.pt"
                torch.save(model.state_dict(), best_checkpoint)
                logging.info("Saved new best model to %s", best_checkpoint)

    if not val_loader:
        best_checkpoint = checkpoint_dir / f"{args.save_prefix}_final.pt"
        torch.save(model.state_dict(), best_checkpoint)
    logging.info("Training finished. Best checkpoint: %s", best_checkpoint)


if __name__ == '__main__':
    main()

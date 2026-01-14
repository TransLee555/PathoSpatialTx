import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from MHGL_ST_model import HierarchicalMultiModalGNN
from stage2_pCR_training import (
    ProcessedImageDataset,
    _set_global_seed,
    infer_feature_dims,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the stage 2 pCR predictor.")
    parser.add_argument("--root-dir", type=Path, required=True, help="Directory containing/holding processed graphs.")
    parser.add_argument("--cell-output-dir", type=Path, required=True, help="Directory containing CellViT outputs (per sample).")
    parser.add_argument("--gene-dir", type=Path, required=True, help="Directory containing predicted gene expression CSV files.")
    parser.add_argument("--svs-dir", type=Path, required=True, help="Directory containing raw SVS files.")
    parser.add_argument("--label-file", type=Path, required=True, help="CSV file containing Patient/Responder annotations.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint (.pth).")
    parser.add_argument("--patch-size-level0", type=int, default=512, help="Patch size (level-0 pixels) used when building graphs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation DataLoader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension inside the GNN.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension passed to the readout head.")
    parser.add_argument("--num-shared-clusters", type=int, default=5, help="Number of shared archetype clusters.")
    parser.add_argument("--gnn-type", choices=["Transformer", "GAT", "GCN"], default="Transformer", help="Intra-modal GNN backbone type.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads for Transformer/GAT layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate applied inside the model.")
    parser.add_argument("--num-intra-layers", type=int, default=3, help="Number of intra-modal GNN layers.")
    parser.add_argument("--num-inter-layers", type=int, default=2, help="Number of inter-modal fusion layers.")
    parser.add_argument("--cell-feature-dim", type=int, default=None, help="Optional override for cell feature dimension.")
    parser.add_argument("--gene-feature-dim", type=int, default=None, help="Optional override for gene feature dimension.")
    parser.add_argument("--prediction-output", type=Path, default=None, help="Optional CSV path to save predictions.")
    parser.add_argument("--device", default="auto", help="Torch device spec (e.g., 'cpu', 'cuda:0', or 'auto').")
    parser.add_argument("--seed", type=int, default=233, help="Random seed.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity.")
    return parser.parse_args()


def infer_feature_dims_from_dataset(
    dataset: ProcessedImageDataset,
    fallback_cell: Optional[int] = None,
    fallback_gene: Optional[int] = None,
) -> Tuple[int, int]:
    if fallback_cell is not None and fallback_gene is not None:
        return fallback_cell, fallback_gene
    candidates: List = []
    if hasattr(dataset, 'data_list') and dataset.data_list:
        candidates = dataset.data_list
    elif len(dataset) > 0:
        candidates = [dataset.get(0)]
    cell_dim, gene_dim = infer_feature_dims(candidates, fallback_cell or 0, fallback_gene or 0)
    if cell_dim == 0 or gene_dim == 0:
        raise RuntimeError("Unable to infer feature dimensions from the dataset. Consider providing explicit overrides.")
    return cell_dim, gene_dim


@torch.no_grad()
def evaluate(model, loader, device, prediction_output: Optional[Path]):
    model.eval()
    all_preds, all_labels = [], []
    for data in tqdm(loader, desc="Predicting"):
        data = data.to(device)
        if not hasattr(data, 'y') or data.y is None or data.y.numel() == 0:
            continue
        labels = data.y.view(-1, 1).float()
        logits, _ = model(data, compute_clustering_loss=False, update_centers=False)
        preds = torch.sigmoid(logits)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    if not all_preds:
        logging.warning("No valid samples were processed.")
        return 0.0, 0.0, []

    preds_tensor = torch.cat(all_preds)
    labels_tensor = torch.cat(all_labels)
    preds_np = preds_tensor.numpy().flatten()
    labels_np = labels_tensor.numpy().flatten()

    try:
        auc = roc_auc_score(labels_np, preds_np)
    except ValueError:
        auc = 0.0

    records = [{'prediction': float(p), 'label': float(l)} for p, l in zip(preds_np, labels_np)]
    if prediction_output is not None:
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(prediction_output, index=False)
        logging.info("Saved predictions to %s", prediction_output)

    return auc, records


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
    checkpoint_path = args.checkpoint.expanduser()

    dataset = ProcessedImageDataset(
        root=str(root_dir),
        nucle_root=str(cell_dir),
        gene_root=str(gene_dir),
        label_root=str(label_file),
        svs_root=str(svs_dir),
        original_patch_size_level0=args.patch_size_level0,
    )
    if len(dataset) == 0:
        logging.error('Dataset is empty. Exiting.')
        raise SystemExit(1)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cell_dim, gene_dim = infer_feature_dims_from_dataset(
        dataset,
        args.cell_feature_dim,
        args.gene_feature_dim,
    )
    logging.info('Detected feature dimensions | cell: %d | gene: %d', cell_dim, gene_dim)

    model = HierarchicalMultiModalGNN(
        cell_dim,
        gene_dim,
        args.hidden_dim,
        args.embedding_dim,
        out_channels=1,
        num_shared_clusters=args.num_shared_clusters,
        gnn_type=args.gnn_type,
        num_attention_heads=args.num_heads,
        dropout_rate=args.dropout,
        num_intra_modal_layers=args.num_intra_layers,
        num_inter_modal_layers=args.num_inter_layers,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    logging.info('Loaded checkpoint from %s', checkpoint_path)

    prediction_output = args.prediction_output.expanduser() if args.prediction_output else None
    auc, _ = evaluate(model, loader, device, prediction_output)
    logging.info('Inference complete | AUC: %.4f', auc)


if __name__ == '__main__':
    main()

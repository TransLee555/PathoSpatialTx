import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage2_os_training import (
    _set_global_seed,
    resolve_device,
    ensure_logdata,
    ProcessedImageDataset,
    HierarchicalMultiModalGNN,
    compute_c_index,
    hazards_to_risk,
    _extract_sample_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the stage 2 overall-survival predictor."
    )
    parser.add_argument("--root-dir", type=Path, required=True, help="Directory used for processed dataset artifacts.")
    parser.add_argument("--cell-output-dir", type=Path, required=True, help="Root directory containing CellViT outputs (per-sample subdirectories).")
    parser.add_argument("--gene-dir", type=Path, required=True, help="Directory containing gene expression prediction CSV files.")
    parser.add_argument("--svs-dir", type=Path, required=True, help="Directory containing the raw SVS files.")
    parser.add_argument("--label-file", type=Path, default=None, help="Optional CSV with survival annotations (PatientID, T, E). Needed only if you want metrics like C-index.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint (.pth).")
    parser.add_argument("--patch-size-level0", type=int, default=512, help="Patch size in level-0 pixels when building graphs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference DataLoader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden channel size inside the GNN.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension passed to the readout head.")
    parser.add_argument("--num-shared-clusters", type=int, default=6, help="Number of shared archetype clusters.")
    parser.add_argument("--gnn-type", choices=["Transformer", "GAT", "GCN"], default="Transformer", help="Intra-modal GNN backbone type.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads for Transformer/GAT layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate applied inside the model.")
    parser.add_argument("--num-intra-layers", type=int, default=3, help="Number of intra-modal GNN layers.")
    parser.add_argument("--num-inter-layers", type=int, default=1, help="Number of inter-modal fusion layers.")
    parser.add_argument("--num-time-bins", type=int, default=5, help="Number of discrete time bins predicted by the model.")
    parser.add_argument("--cell-feature-dim", type=int, default=None, help="Optional override for cell feature dimension.")
    parser.add_argument("--gene-feature-dim", type=int, default=None, help="Optional override for gene feature dimension.")
    parser.add_argument("--prediction-output", type=Path, default=None, help="Optional CSV path to save per-sample risk predictions.")
    parser.add_argument("--device", default="auto", help="Torch device spec (e.g., 'cpu', 'cuda:0', or 'auto').")
    parser.add_argument("--preferred-gpu", type=int, default=0, help="Preferred CUDA index when --device=auto.")
    parser.add_argument("--seed", type=int, default=233, help="Random seed.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity.")
    return parser.parse_args()


def infer_feature_dims(dataset: ProcessedImageDataset,
                       fallback_cell: Optional[int] = None,
                       fallback_gene: Optional[int] = None) -> Tuple[int, int]:
    cell_dim = fallback_cell
    gene_dim = fallback_gene
    candidates: List = []
    if hasattr(dataset, 'data_list') and dataset.data_list:
        candidates = dataset.data_list
    elif len(dataset) > 0:
        candidates = [dataset.get(0)]

    for data in candidates:
        if cell_dim is None and 'cell' in data.node_types and hasattr(data['cell'], 'x') and data['cell'].x is not None and data['cell'].x.numel() > 0:
            cell_dim = data['cell'].x.shape[1]
        if gene_dim is None and 'gene' in data.node_types and hasattr(data['gene'], 'x') and data['gene'].x is not None and data['gene'].x.numel() > 0:
            gene_dim = data['gene'].x.shape[1]
        if cell_dim is not None and gene_dim is not None:
            break

    if cell_dim is None or gene_dim is None or cell_dim <= 0 or gene_dim <= 0:
        raise RuntimeError("Unable to infer feature dimensions from the dataset. Provide overrides via --cell-feature-dim/--gene-feature-dim.")
    return cell_dim, gene_dim


def _resolve_sample_names(data, count: int) -> List[Optional[str]]:
    names = _extract_sample_names(data) or []
    if len(names) < count:
        names = names + [None] * (count - len(names))
    elif len(names) > count:
        names = names[:count]
    return names


@torch.no_grad()
def run_inference(model: HierarchicalMultiModalGNN,
                 loader: DataLoader,
                 device: torch.device,
                 prediction_output: Optional[Path] = None) -> Tuple[float, int]:
    model.eval()
    records: List[dict] = []
    risk_tensors: List[torch.Tensor] = []
    total_samples = 0

    for data in tqdm(loader, desc="Inferencing"):
        data = ensure_logdata(data).to(device)
        logits, _ = model(data, compute_clustering_loss=False, update_centers=False)
        risks = hazards_to_risk(logits, mode="cumhaz").detach().cpu().view(-1)
        num_graphs = int(risks.numel())
        risk_tensors.append(risks)

        times = data.t.detach().cpu().view(-1) if hasattr(data, 't') and data.t is not None else None
        events = data.e.detach().cpu().view(-1) if hasattr(data, 'e') and data.e is not None else None
        names = _resolve_sample_names(data, num_graphs)

        for idx in range(num_graphs):
            rec = {
                'sample_index': total_samples + idx,
                'sample_name': names[idx],
                'risk_score': float(risks[idx].item()),
                'time_years': float(times[idx].item()) if times is not None and idx < times.numel() else None,
                'event': float(events[idx].item()) if events is not None and idx < events.numel() else None,
            }
            records.append(rec)
        total_samples += num_graphs

    if not records:
        logging.warning("No samples were processed during inference.")
        return 0.0, 0

    # Compute C-index using samples that have both time and event annotations
    valid_times, valid_events, valid_risks = [], [], []
    for rec in records:
        if rec['time_years'] is None or rec['event'] is None:
            continue
        valid_times.append(rec['time_years'])
        valid_events.append(rec['event'])
        valid_risks.append(rec['risk_score'])

    c_index = 0.0
    if len(valid_risks) >= 2:
        risk_tensor = torch.tensor(valid_risks, dtype=torch.float32)
        time_tensor = torch.tensor(valid_times, dtype=torch.float32)
        event_tensor = torch.tensor(valid_events, dtype=torch.float32)
        c_index = compute_c_index(risk_tensor, time_tensor, event_tensor)
    else:
        logging.warning("Insufficient labeled samples to compute C-index (need at least 2 with survival info).")

    if prediction_output is not None:
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(prediction_output, index=False)
        logging.info("Saved predictions to %s", prediction_output)

    return c_index, len(records)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='[%(asctime)s] %(levelname)s - %(message)s')
    _set_global_seed(args.seed)
    device = resolve_device(args.device, args.preferred_gpu)
    logging.info("Using device: %s", device)

    root_dir = args.root_dir.expanduser()
    cell_dir = args.cell_output_dir.expanduser()
    gene_dir = args.gene_dir.expanduser()
    svs_dir = args.svs_dir.expanduser()
    label_file = args.label_file.expanduser() if args.label_file else None
    checkpoint_path = args.checkpoint.expanduser()

    if label_file is None:
        logging.info('No label file provided; survival metrics will be skipped and predictions only will be exported.')

    dataset = ProcessedImageDataset(
        root=str(root_dir),
        nucle_root=str(cell_dir),
        gene_root=str(gene_dir),
        svs_root=str(svs_dir),
        label_root=str(label_file) if label_file else None,
        original_patch_size_level0=args.patch_size_level0,
    )
    if len(dataset) == 0:
        logging.error('Dataset is empty. Exiting.')
        raise SystemExit(1)

    dataset.set_cg_sampling_mode('eval')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cell_dim, gene_dim = infer_feature_dims(dataset, args.cell_feature_dim, args.gene_feature_dim)
    logging.info('Detected feature dimensions | cell: %d | gene: %d', cell_dim, gene_dim)

    model = HierarchicalMultiModalGNN(
        cell_dim,
        gene_dim,
        args.hidden_dim,
        args.embedding_dim,
        args.num_time_bins,
        args.num_shared_clusters,
        gnn_type=args.gnn_type,
        num_attention_heads=args.num_heads,
        dropout_rate=args.dropout,
        num_intra_modal_layers=args.num_intra_layers,
        num_inter_modal_layers=args.num_inter_layers,
        num_time_bins=args.num_time_bins,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    logging.info('Loaded checkpoint from %s', checkpoint_path)

    prediction_output = args.prediction_output.expanduser() if args.prediction_output else None
    c_index, num_samples = run_inference(model, loader, device, prediction_output)
    logging.info('Inference complete on %d samples | C-index: %.4f', num_samples, c_index)


if __name__ == '__main__':
    main()

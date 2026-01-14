#!/usr/bin/env python3
"""Gene expression prediction using stage-1 patches and a CONCH backbone."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional

import warnings

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn.pool import knn_graph as pyg_knn_graph
from torch_geometric.nn.pool import radius_graph as pyg_radius_graph
from torch_geometric.utils import to_undirected

from stage1_IRSVGs_predictor_training import (
    GeneGraphRegressor,
    load_tile_encoder,
    resolve_device,
    setup_logging,
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None
    TQDM_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict gene expression from histology patches using a pre-trained CONCH encoder and a GNN head.",
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        required=True,
        help="Directory that contains the stage-1 'patches' (and optional 'mask') subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-sample CSV predictions will be written.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to the trained regression head checkpoint (.pth).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory for cached patch embeddings (defaults to PATCH_ROOT/cache_patches_embedding).",
    )
    parser.add_argument(
        "--num-genes",
        type=int,
        default=215,
        help="Number of regression targets produced by the model.",
    )
    parser.add_argument(
        "--graph-type",
        choices=["radius", "knn"],
        default="radius",
        help="Spatial graph construction strategy.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.1,
        help="Radius (in normalized coordinate units) when using radius graphs.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="Number of neighbors when using KNN graphs.",
    )
    parser.add_argument(
        "--use-edge-attr",
        action="store_true",
        help="Attach relative positional features to each graph edge.",
    )
    parser.add_argument(
        "--edge-attr-dim",
        type=int,
        default=2,
        help="Number of features emitted per edge attribute (relative positions are padded or truncated to match).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples processed per evaluation batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (keep at zero when the dataset needs GPU access).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--conch-model-id",
        default="conch_ViT-B-16",
        help="Model identifier for create_model_from_pretrained.",
    )
    parser.add_argument(
        "--conch-checkpoint",
        default="hf_hub:MahmoodLab/conch",
        help="Checkpoint reference passed to create_model_from_pretrained.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token required for private checkpoints.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=512,
        help="Feature dimension after projecting CONCH embeddings before the GNN layers.",
    )
    parser.add_argument(
        "--position-dim",
        type=int,
        default=20,
        help="Dimensionality of the sinusoidal positional encoding for each spot.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability applied inside the regression head.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--gene-columns",
        type=Path,
        default=None,
        help="Optional file that lists gene names for the prediction columns (first column per line is used).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a tqdm progress bar (requires tqdm to be installed).",
    )
    return parser.parse_args()

def load_gene_columns_file(path: Path) -> List[str]:
    columns: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = re.split(r"[\s,]+", stripped)
            if parts:
                columns.append(parts[0])
    return columns
class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        tile_encoder: torch.nn.Module,
        preprocess,
        device: torch.device,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.root_dir = root_dir
        self.patch_dir = self.root_dir / "patches"
        if not self.patch_dir.exists():
            raise FileNotFoundError(f"Patch directory {self.patch_dir} does not exist.")
        self.tile_encoder = tile_encoder
        self.preprocess = preprocess
        self.device = device
        self.cache_dir = cache_dir or (self.root_dir / "cache_patches_embedding")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        visual = getattr(self.tile_encoder, "visual", None)
        self.backbone_dim = getattr(visual, "output_dim", 512)
        self.records = self._index_patches()
        self.sample_names = sorted(self.records["sample_name"].unique().tolist()) if not self.records.empty else []

    def _index_patches(self) -> pd.DataFrame:
        rows = []
        for sample_dir in sorted(self.patch_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            for patch_path in sorted(sample_dir.iterdir()):
                if not patch_path.is_file() or patch_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                coords = self._extract_coords(patch_path.stem)
                if coords is None:
                    logging.debug("Skipping patch with unexpected name: %s", patch_path.name)
                    continue
                rows.append(
                    {
                        "sample_name": sample_dir.name,
                        "patch_path": patch_path,
                        "x": coords[0],
                        "y": coords[1],
                    }
                )
        if not rows:
            logging.warning("No patch files found under %s", self.patch_dir)
            return pd.DataFrame(columns=["sample_name", "patch_path", "x", "y"])
        return pd.DataFrame(rows)

    @staticmethod
    def _extract_coords(stem: str) -> Optional[tuple[int, int]]:
        parts = stem.split("_")
        if len(parts) < 3:
            return None
        try:
            return int(parts[-2]), int(parts[-1])
        except ValueError:
            return None

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> dict:
        sample_name = self.sample_names[idx]
        sample_df = self.records[self.records["sample_name"] == sample_name].reset_index(drop=True)
        raw_coords = torch.tensor(sample_df[["x", "y"]].values, dtype=torch.float32)
        cache_path = self.cache_dir / f"{sample_name}_patches.pt"

        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu")
            embeddings = cache.get("embeddings", torch.empty(0, self.backbone_dim))
            valid_indices = cache.get("indices", torch.empty(0, dtype=torch.long))
        else:
            embeddings, valid_indices = self._encode_sample(sample_df, sample_name)
            torch.save({"embeddings": embeddings, "indices": valid_indices}, cache_path)
            logging.info("Cached %d embeddings for %s", embeddings.shape[0], sample_name)

        if embeddings.numel() == 0 or valid_indices.numel() == 0:
            return {
                "sample_name": sample_name,
                "patches": torch.empty(0, self.backbone_dim),
                "coords": torch.empty(0, 2),
                "raw_coords": torch.empty(0, 2),
            }

        coord_filtered = raw_coords[valid_indices]
        coord_normalized = self._normalize_coords(coord_filtered)
        return {
            "sample_name": sample_name,
            "patches": embeddings,
            "coords": coord_normalized,
            "raw_coords": coord_filtered,
        }

    def _encode_sample(self, sample_df: pd.DataFrame, sample_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        valid_indices: List[int] = []
        for row_idx, patch_path in enumerate(sample_df["patch_path"]):
            try:
                with Image.open(patch_path) as image:
                    image = image.convert("RGB")
                    tensor = self.preprocess(image) if self.preprocess else image
            except FileNotFoundError:
                logging.warning("Missing patch file %s", patch_path)
                continue
            except Exception as exc:
                logging.error("Failed to load %s: %s", patch_path, exc)
                continue

            if not isinstance(tensor, torch.Tensor):
                logging.warning("Preprocess pipeline did not return a tensor for %s", patch_path)
                continue

            with torch.no_grad():
                embedding = self.tile_encoder.encode_image(
                    tensor.unsqueeze(0).to(self.device),
                    proj_contrast=True,
                    normalize=True,
                )
            embeddings.append(embedding.squeeze(0).cpu())
            valid_indices.append(row_idx)

        if embeddings:
            return torch.stack(embeddings, dim=0), torch.tensor(valid_indices, dtype=torch.long)
        return torch.empty(0, self.backbone_dim), torch.empty(0, dtype=torch.long)

    @staticmethod
    def _normalize_coords(coords: torch.Tensor) -> torch.Tensor:
        if coords.numel() == 0:
            return coords
        normalized = coords.clone()
        min_x, _ = torch.min(normalized[:, 0], dim=0)
        max_x, _ = torch.max(normalized[:, 0], dim=0)
        min_y, _ = torch.min(normalized[:, 1], dim=0)
        max_y, _ = torch.max(normalized[:, 1], dim=0)
        normalized[:, 0] = 0.5 if max_x == min_x else (normalized[:, 0] - min_x) / (max_x - min_x)
        normalized[:, 1] = 0.5 if max_y == min_y else (normalized[:, 1] - min_y) / (max_y - min_y)
        return normalized


class GraphCollator:
    def __init__(
        self,
        device: torch.device,
        graph_type: str,
        radius: float,
        knn_k: int,
        use_edge_attr: bool,
        edge_attr_dim: int,
    ) -> None:
        self.device = device
        self.graph_type = graph_type
        self.radius = radius
        self.knn_k = knn_k
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim if use_edge_attr else 0

    def __call__(self, batch: List[dict]) -> Optional[dict]:
        sample_names: List[str] = []
        sample_lengths: List[int] = []
        raw_coords: List[torch.Tensor] = []
        patch_batches: List[torch.Tensor] = []
        coord_batches: List[torch.Tensor] = []
        edge_indices: List[torch.Tensor] = []
        edge_attrs: List[torch.Tensor] = []
        offset = 0

        for sample in batch:
            sample_names.append(sample["sample_name"])
            coords = sample["coords"].to(torch.float32)
            patches = sample["patches"].to(torch.float32)
            raw_coord = sample["raw_coords"].to(torch.float32)
            raw_coords.append(raw_coord)
            length = coords.shape[0]
            sample_lengths.append(length)

            if length == 0:
                continue

            patch_batches.append(patches)
            coord_batches.append(coords)

            edge_index, edge_attr = self._build_graph(coords)
            if edge_index is not None and edge_index.numel() > 0:
                edge_indices.append(edge_index + offset)
                if edge_attr is not None:
                    edge_attrs.append(edge_attr)
            offset += length

        if not patch_batches:
            return None

        patches_tensor = torch.cat(patch_batches, dim=0).to(self.device)
        coords_tensor = torch.cat(coord_batches, dim=0).to(self.device)

        if edge_indices:
            edge_index_tensor = torch.cat(edge_indices, dim=1).to(self.device)
            if self.use_edge_attr and edge_attrs:
                edge_attr_tensor = torch.cat(edge_attrs, dim=0).to(self.device)
            else:
                edge_attr_tensor = None
        else:
            edge_index_tensor = torch.empty(2, 0, dtype=torch.long, device=self.device)
            edge_attr_tensor = (
                torch.empty(0, self.edge_attr_dim, dtype=torch.float32, device=self.device)
                if self.use_edge_attr
                else None
            )

        return {
            "sample_names": sample_names,
            "sample_lengths": sample_lengths,
            "raw_coords": raw_coords,
            "patches": patches_tensor,
            "coords": coords_tensor,
            "edge_index": edge_index_tensor,
            "edge_attr": edge_attr_tensor,
        }

    def _build_graph(
        self, coords: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if coords.shape[0] <= 1:
            return None, None
        coords_cpu = coords.cpu()
        if self.graph_type == "radius":
            edge_index = pyg_radius_graph(coords_cpu, r=self.radius, loop=False)
        else:
            edge_index = pyg_knn_graph(coords_cpu, k=self.knn_k, loop=False)
        if edge_index.numel() == 0:
            return None, None
        edge_index = to_undirected(edge_index, num_nodes=coords.shape[0])
        if not self.use_edge_attr:
            return edge_index, None
        row, col = edge_index
        edge_attr = coords_cpu[col] - coords_cpu[row]
        if edge_attr.shape[1] < self.edge_attr_dim:
            pad = torch.zeros(edge_attr.shape[0], self.edge_attr_dim - edge_attr.shape[1])
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        elif edge_attr.shape[1] > self.edge_attr_dim:
            edge_attr = edge_attr[:, : self.edge_attr_dim]
        return edge_index, edge_attr


def save_predictions(
    predictions: torch.Tensor,
    raw_coords: torch.Tensor,
    sample_name: str,
    output_dir: Path,
    gene_columns: Optional[List[str]] = None,
) -> None:
    coords_np = raw_coords.cpu().numpy()
    preds_np = predictions.cpu().numpy()
    index_labels = [f"{int(x)}_{int(y)}" for x, y in coords_np]
    if gene_columns and len(gene_columns) == preds_np.shape[1]:
        df = pd.DataFrame(preds_np, index=index_labels, columns=gene_columns)
    else:
        if gene_columns and len(gene_columns) != preds_np.shape[1]:
            logging.warning(
                "Gene column count (%d) does not match prediction dimension (%d); writing unnamed columns.",
                len(gene_columns),
                preds_np.shape[1],
            )
        df = pd.DataFrame(preds_np, index=index_labels)
    out_path = output_dir / f"{sample_name}.csv"
    df.to_csv(out_path)
    logging.info("Saved predictions for %s to %s", sample_name, out_path)


def run_inference(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        logging.info("Using CUDA device %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logging.info("Using device %s", device)

    gene_columns: Optional[List[str]] = None
    if args.gene_columns:
        gene_path = args.gene_columns.expanduser()
        if not gene_path.exists():
            logging.error("Gene columns file %s does not exist; columns will remain unnamed.", gene_path)
        else:
            gene_columns = load_gene_columns_file(gene_path)
            if gene_columns and len(gene_columns) != args.num_genes:
                logging.warning(
                    "Gene column count (%d) does not match --num-genes (%d); ignoring provided columns.",
                    len(gene_columns),
                    args.num_genes,
                )
                gene_columns = None

    tile_encoder, preprocess = load_tile_encoder(
        args.conch_model_id,
        args.conch_checkpoint,
        args.hf_token,
        device,
    )

    dataset = ImageDataset(
        root_dir=args.patch_root.expanduser(),
        tile_encoder=tile_encoder,
        preprocess=preprocess,
        device=device,
        cache_dir=args.cache_dir.expanduser() if args.cache_dir else None,
    )

    if len(dataset) == 0:
        logging.error("No samples found under %s", args.patch_root)
        return

    collate_fn = GraphCollator(
        device=device,
        graph_type=args.graph_type,
        radius=args.radius,
        knn_k=args.knn_k,
        use_edge_attr=args.use_edge_attr,
        edge_attr_dim=args.edge_attr_dim,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = GeneGraphRegressor(
        backbone_feature_dim=dataset.backbone_dim,
        projected_feature_dim=args.projection_dim,
        position_dim=args.position_dim,
        output_dim=args.num_genes,
        dropout_rate=args.dropout,
        edge_attr_dim=args.edge_attr_dim if args.use_edge_attr else 0,
    ).to(device)

    logging.info("Loading model checkpoint from %s", args.model_checkpoint)
    state_dict = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = dataloader
    if args.progress and TQDM_AVAILABLE:
        iterator = tqdm(dataloader, desc="Predicting", unit="batch")
    elif args.progress and not TQDM_AVAILABLE:
        logging.warning("tqdm is not installed; progress bar disabled.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if batch is None:
                logging.warning("Skipping empty batch at index %d", batch_idx)
                continue
            if batch["patches"].shape[0] == 0:
                logging.warning("Batch %d does not contain valid patches.", batch_idx)
                continue

            predictions, _ = model(
                batch["patches"],
                batch["coords"],
                batch["edge_index"],
                batch["edge_attr"],
            )

            start = 0
            for sample_name, length, raw_coord in zip(
                batch["sample_names"], batch["sample_lengths"], batch["raw_coords"]
            ):
                if length == 0:
                    logging.warning("Sample %s has no valid patches; skipping output.", sample_name)
                    continue
                end = start + length
                save_predictions(
                    predictions[start:end],
                    raw_coord,
                    sample_name,
                    output_dir,
                    gene_columns,
                )
                start = end


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

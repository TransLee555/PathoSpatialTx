#!/usr/bin/env python3
"""Train the stage-1 gene expression predictor with configurable CLI arguments."""
from __future__ import annotations

import argparse
import gzip
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from CONCH_main.conch.open_clip_custom import create_model_from_pretrained
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import knn_graph as pyg_knn_graph
from torch_geometric.nn.pool import radius_graph as pyg_radius_graph
from torch_geometric.utils import to_undirected

DEFAULT_GTF_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz"
DEFAULT_ANNOTATIONS_DIR = Path("data") / "annotations"
DEFAULT_GENE_MAPPING_FILE = "gene_name_to_ensg.tsv"
DEFAULT_GENE_LENGTH_FILE = "gene_length.txt"

# ---------------------------------------------------------------------------
# Argument parsing and logging
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the graph-based gene expression regressor using CONCH features.",
    )
    parser.add_argument("--patch-root", type=Path, required=True, help="Directory containing patch images.")
    parser.add_argument("--train-csv", type=Path, required=True, help="CSV file with training samples.")
    parser.add_argument("--val-csv", type=Path, default=None, help="Optional CSV file for validation samples.")
    parser.add_argument("--test-csv", type=Path, default=None, help="Optional CSV file for test evaluation.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio when --val-csv is not provided.",
    )
    parser.add_argument(
        "--gene-mapping",
        type=Path,
        default=None,
        help=(
            "Optional TSV/CSV mapping from gene names to ENSG IDs. "
            "If omitted, the script will download GENCODE annotations and build it automatically."
        ),
    )
    parser.add_argument(
        "--gene-lengths",
        type=Path,
        default=None,
        help=(
            "Optional TSV/CSV containing ENSG IDs and their lengths. "
            "If omitted, lengths are derived from the downloaded GENCODE GTF."
        ),
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help=(
            "Directory used to cache downloaded GTF files when --gene-mapping/--gene-lengths are not provided."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for cached patch embeddings (defaults to PATCH_ROOT/cache_training_embeddings).",
    )
    parser.add_argument(
        "--patch-extension",
        default=".jpg",
        help="Patch image extension (e.g., .jpg or .png).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers (keep 0 if dataset holds GPU models).",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device string.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for CONCH weights.")
    parser.add_argument("--conch-model-id", default="conch_ViT-B-16", help="Model id for CONCH encoder.")
    parser.add_argument(
        "--conch-checkpoint",
        default="hf_hub:MahmoodLab/conch",
        help="Checkpoint reference for CONCH encoder.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=512,
        help="Projected feature dimension before the TransformerConv layers.",
    )
    parser.add_argument(
        "--position-dim",
        type=int,
        default=20,
        help="Sinusoidal positional encoding dimension (must be even).",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate inside the model head.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=1.0,
        help="Weight applied to the contrastive InfoNCE loss.",
    )
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature.")
    parser.add_argument("--batch-size-train", type=int, default=2, help="Training batch size (samples).")
    parser.add_argument("--batch-size-eval", type=int, default=1, help="Validation/test batch size.")
    parser.add_argument(
        "--graph-type",
        choices=["radius", "knn"],
        default="radius",
        help="Spatial graph builder type.",
    )
    parser.add_argument("--radius", type=float, default=0.1, help="Radius used when graph-type=radius.")
    parser.add_argument("--knn-k", type=int, default=10, help="Neighbors count when graph-type=knn.")
    parser.add_argument(
        "--use-edge-attr",
        action="store_true",
        help="Include relative positional edge attributes.",
    )
    parser.add_argument("--edge-attr-dim", type=int, default=2, help="Edge attribute dimension.")
    parser.add_argument(
        "--gene-sim-threshold",
        type=float,
        default=0.5,
        help="Threshold for cosine/pearson gene similarity when building positives.",
    )
    parser.add_argument(
        "--gene-sim-metric",
        choices=["cosine", "pearson"],
        default="cosine",
        help="Metric used to compute gene similarity.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where checkpoints will be stored.",
    )
    parser.add_argument(
        "--save-prefix",
        default="stage1_regressor",
        help="Prefix used when saving checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not found; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def load_tile_encoder(
    model_id: str,
    checkpoint: str,
    hf_token: Optional[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[Callable]]:
    tile_encoder, preprocess = create_model_from_pretrained(model_id, checkpoint, hf_auth_token=hf_token)
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    for param in tile_encoder.parameters():
        param.requires_grad = False
    logging.info("Loaded CONCH encoder '%s' from %s", model_id, checkpoint)
    return tile_encoder, preprocess


# ---------------------------------------------------------------------------
# Gene annotation utilities
# ---------------------------------------------------------------------------


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logging.info("Using cached %s", destination)
        return
    logging.info("Downloading %s -> %s", url, destination)
    urlretrieve(url, destination)


def _decompress_gzip(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    logging.info("Decompressing %s -> %s", src, dst)
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def _parse_gtf_attributes(attr_str: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for field in attr_str.strip().split(";"):
        field = field.strip()
        if not field:
            continue
        if " " not in field:
            continue
        key, value = field.split(" ", 1)
        attributes[key] = value.strip().strip('"')
    return attributes


def _extract_gene_annotations(gtf_path: Path, mapping_out: Path, lengths_out: Path) -> None:
    logging.info("Extracting gene annotations from %s", gtf_path)
    name_to_gene: Dict[str, str] = {}
    gene_lengths = defaultdict(int)
    with gtf_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            attrs = _parse_gtf_attributes(parts[8])
            gene_id = attrs.get("gene_id")
            if not gene_id:
                continue
            gene_id = gene_id.split(".")[0]
            gene_name = attrs.get("gene_name")
            if gene_name and gene_name not in name_to_gene:
                name_to_gene[gene_name] = gene_id
            if feature != "exon":
                continue
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                continue
            length = max(0, end - start + 1)
            gene_lengths[gene_id] += length
    if not name_to_gene:
        raise RuntimeError("No gene annotations could be parsed from the GTF file.")
    with mapping_out.open("w", encoding="utf-8") as handle:
        for gene_name in sorted(name_to_gene):
            handle.write(f"{gene_name}\t{name_to_gene[gene_name]}\n")
    with lengths_out.open("w", encoding="utf-8") as handle:
        handle.write("gene_id\tlength\n")
        for gene_id in sorted(gene_lengths):
            handle.write(f"{gene_id}\t{gene_lengths[gene_id]}\n")
    logging.info(
        "Saved %d gene mappings to %s and %d gene lengths to %s",
        len(name_to_gene),
        mapping_out,
        len(gene_lengths),
        lengths_out,
    )


def _prepare_gene_annotations(
    gene_mapping: Optional[Path],
    gene_lengths: Optional[Path],
    annotations_dir: Path,
    gtf_url: str = DEFAULT_GTF_URL,
) -> Tuple[Path, Path]:
    mapping_path = gene_mapping.expanduser() if gene_mapping else None
    lengths_path = gene_lengths.expanduser() if gene_lengths else None
    if mapping_path and lengths_path:
        return mapping_path, lengths_path

    annotations_dir = annotations_dir.expanduser()
    annotations_dir.mkdir(parents=True, exist_ok=True)
    gtf_filename = Path(gtf_url.split("/")[-1])
    gtf_gz_path = annotations_dir / gtf_filename
    gtf_path = annotations_dir / Path(gtf_filename.stem)
    _download_file(gtf_url, gtf_gz_path)
    _decompress_gzip(gtf_gz_path, gtf_path)

    mapping_out = annotations_dir / DEFAULT_GENE_MAPPING_FILE
    lengths_out = annotations_dir / DEFAULT_GENE_LENGTH_FILE
    if not mapping_out.exists() or not lengths_out.exists():
        _extract_gene_annotations(gtf_path, mapping_out, lengths_out)
    else:
        logging.info("Using cached gene annotations in %s", annotations_dir)
    return mapping_out, lengths_out


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def generate_2d_sinusoidal_position_encoding(coords: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if feature_dim % 2 != 0:
        raise ValueError("position_dim must be even")
    num_spots = coords.shape[0]
    half_dim = feature_dim // 2
    div_term = torch.exp(torch.arange(0, half_dim, 2, dtype=torch.float32, device=coords.device) * (-math.log(10000.0) / half_dim))
    pe_x = torch.zeros(num_spots, half_dim, dtype=torch.float32, device=coords.device)
    pe_y = torch.zeros_like(pe_x)
    pe_x[:, 0::2] = torch.sin(coords[:, 0].unsqueeze(1) * div_term)
    pe_x[:, 1::2] = torch.cos(coords[:, 0].unsqueeze(1) * div_term)
    pe_y[:, 0::2] = torch.sin(coords[:, 1].unsqueeze(1) * div_term)
    pe_y[:, 1::2] = torch.cos(coords[:, 1].unsqueeze(1) * div_term)
    return torch.cat([pe_x, pe_y], dim=1)


class GeneGraphRegressor(nn.Module):
    def __init__(
        self,
        backbone_feature_dim: int,
        projected_feature_dim: int,
        position_dim: int,
        output_dim: int,
        dropout_rate: float,
        edge_attr_dim: int,
    ) -> None:
        super().__init__()
        self.position_dim = position_dim
        self.initial_dim = projected_feature_dim + position_dim
        self.edge_attr_dim = edge_attr_dim
        edge_dim = edge_attr_dim if edge_attr_dim > 0 else None

        self.tile_proj = nn.Sequential(
            nn.Linear(backbone_feature_dim, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, projected_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(projected_feature_dim),
            nn.Dropout(dropout_rate),
        )
        self.gnn_layer1 = TransformerConv(self.initial_dim, self.initial_dim // 4, heads=4, edge_dim=edge_dim)
        self.gnn_layer1_norm = nn.LayerNorm(self.initial_dim)
        self.gnn_layer1_drop = nn.Dropout(dropout_rate)
        self.gnn_layer2 = TransformerConv(self.initial_dim, self.initial_dim // 4, heads=4, edge_dim=edge_dim)
        self.gnn_layer2_norm = nn.LayerNorm(self.initial_dim)
        self.gnn_layer2_drop = nn.Dropout(dropout_rate)
        self.predictor = nn.Sequential(
            nn.Linear(self.initial_dim * 2, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
        )

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_features = self.tile_proj(patch_embeddings)
        position_encoding = generate_2d_sinusoidal_position_encoding(coordinates, self.position_dim)
        initial_features = torch.cat([patch_features, position_encoding], dim=1)

        x1 = self.gnn_layer1(initial_features, edge_index, edge_attr=edge_attr)
        x1 = self.gnn_layer1_drop(self.gnn_layer1_norm(F.relu(x1)))
        context = initial_features + x1

        x2 = self.gnn_layer2(context, edge_index, edge_attr=edge_attr)
        x2 = self.gnn_layer2_drop(self.gnn_layer2_norm(F.relu(x2)))
        context = context + x2

        final_features = torch.cat([initial_features, context], dim=1)
        regression_output = self.predictor(final_features)
        return regression_output, context

    @staticmethod
    def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _read_mapping_file(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Mapping file {path} must contain at least two columns.")
    name_col = df.columns[0]
    ensg_col = df.columns[1]
    df[ensg_col] = df[ensg_col].astype(str).str.split(".").str[0]
    return dict(zip(df[name_col], df[ensg_col]))


def _read_gene_lengths(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path, sep=None, engine="python")
    lower_cols = {c.lower(): c for c in df.columns}
    gene_col = lower_cols.get("gene") or next(iter(df.columns))
    length_col = lower_cols.get("length") or lower_cols.get("merged") or df.columns[-1]
    df[gene_col] = df[gene_col].astype(str).str.split(".").str[0]
    return dict(zip(df[gene_col], df[length_col]))


class ExpressionDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        patch_root: Path,
        preprocess: Optional[Callable],
        tile_encoder: torch.nn.Module,
        device: torch.device,
        cache_dir: Path,
        patch_extension: str,
        gene_mapping_path: Optional[Path] = None,
        gene_lengths_path: Optional[Path] = None,
    ) -> None:
        self.csv_path = csv_path
        self.patch_root = patch_root
        self.preprocess = preprocess
        self.tile_encoder = tile_encoder
        self.device = device
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.patch_extension = patch_extension if patch_extension.startswith(".") else f".{patch_extension}"
        self.data = pd.read_csv(csv_path, index_col=0)
        self._augment_index()
        self.gene_cols = [c for c in self.data.columns if c not in {"Sample_Name", "Coord_X", "Coord_Y"}]
        self.num_genes = len(self.gene_cols)
        if self.num_genes == 0:
            raise ValueError(f"No gene expression columns found in {csv_path}.")
        visual = getattr(self.tile_encoder, "visual", None)
        self.backbone_dim = getattr(visual, "output_dim", 512)
        self.sample_names = sorted(self.data["Sample_Name"].dropna().unique().tolist())
        self.gene_lengths = self._build_gene_lengths(gene_mapping_path, gene_lengths_path)

    def _augment_index(self) -> None:
        samples = []
        xs = []
        ys = []
        split_index = self.data.index.to_series().str.split("_")
        for idx, tokens in split_index.items():
            if tokens is None or len(tokens) < 3:
                samples.append(None)
                xs.append(np.nan)
                ys.append(np.nan)
                logging.warning("Index %s does not contain sample and coordinates; skipping.", idx)
                continue
            samples.append("_".join(tokens[:-2]))
            xs.append(tokens[-2])
            ys.append(tokens[-1])
        self.data["Sample_Name"] = samples
        self.data["Coord_X"] = pd.to_numeric(xs, errors="coerce")
        self.data["Coord_Y"] = pd.to_numeric(ys, errors="coerce")

    def _build_gene_lengths(
        self,
        gene_mapping_path: Optional[Path],
        gene_lengths_path: Optional[Path],
    ) -> Optional[torch.Tensor]:
        if gene_mapping_path is None or gene_lengths_path is None:
            logging.warning("Gene lengths will not be used (missing mapping or lengths file).")
            return None
        if not gene_mapping_path.exists() or not gene_lengths_path.exists():
            logging.warning("Gene mapping (%s) or lengths (%s) not found; skipping TPM normalization.", gene_mapping_path, gene_lengths_path)
            return None
        try:
            name_to_ensg = _read_mapping_file(gene_mapping_path)
            ensg_to_length = _read_gene_lengths(gene_lengths_path)
        except Exception as exc:
            logging.error("Failed to load gene mapping/lengths: %s", exc)
            return None
        lengths: List[float] = []
        matched = 0
        for gene in self.gene_cols:
            ensg = name_to_ensg.get(gene)
            length = ensg_to_length.get(ensg) if ensg else None
            if length and length > 0:
                matched += 1
                lengths.append(float(length))
            else:
                lengths.append(0.0)
        if matched == 0:
            logging.warning("No gene lengths matched; TPM normalization disabled.")
            return None
        tensor = torch.tensor(lengths, dtype=torch.float32)
        tensor[tensor <= 0] = 1e-6
        logging.info("Matched %d / %d genes with length annotations.", matched, len(self.gene_cols))
        return tensor

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_name = self.sample_names[idx]
        sample_df = self.data[self.data["Sample_Name"] == sample_name].copy()
        if sample_df.empty:
            logging.warning("Sample %s is empty in %s.", sample_name, self.csv_path)
            return {
                "sample_name": sample_name,
                "patches": torch.empty(0, self.backbone_dim),
                "coords": torch.empty(0, 2),
                "genes": torch.empty(0, self.num_genes),
            }
        gene_tensor = torch.tensor(sample_df[self.gene_cols].values, dtype=torch.float32)
        coord_tensor = torch.tensor(sample_df[["Coord_X", "Coord_Y"]].values, dtype=torch.float32)
        cache_path = self.cache_dir / f"{sample_name}_patches.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu")
            embeddings = cache.get("embeddings", torch.empty(0, self.backbone_dim))
            valid_indices = cache.get("indices", torch.empty(0, dtype=torch.long))
        else:
            embeddings, valid_indices = self._encode_sample(sample_df)
            torch.save({"embeddings": embeddings, "indices": valid_indices}, cache_path)
            logging.info("Cached %d embeddings for %s", embeddings.shape[0], sample_name)
        if embeddings.numel() == 0 or valid_indices.numel() == 0:
            return {
                "sample_name": sample_name,
                "patches": torch.empty(0, self.backbone_dim),
                "coords": torch.empty(0, 2),
                "genes": torch.empty(0, self.num_genes),
            }
        coords_filtered = coord_tensor[valid_indices]
        genes_filtered = gene_tensor[valid_indices]
        coords_normalized = self._normalize_coords(coords_filtered)
        genes_processed = self._apply_tpm_log(genes_filtered)
        return {
            "sample_name": sample_name,
            "patches": embeddings,
            "coords": coords_normalized,
            "genes": genes_processed,
        }

    def _encode_sample(self, sample_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        valid_indices: List[int] = []
        for row_idx, index_label in enumerate(sample_df.index):
            img_path = self.patch_root / f"{index_label}{self.patch_extension}"
            if not img_path.exists():
                logging.debug("Patch %s not found for %s", img_path, index_label)
                continue
            try:
                with Image.open(img_path) as image:
                    image = image.convert("RGB")
                    tensor = self.preprocess(image) if self.preprocess else image
            except Exception as exc:
                logging.warning("Failed to load %s: %s", img_path, exc)
                continue
            if not isinstance(tensor, torch.Tensor):
                logging.warning("Preprocess pipeline did not return a tensor for %s", img_path)
                continue
            with torch.no_grad():
                embedding = self.tile_encoder.encode_image(tensor.unsqueeze(0).to(self.device), proj_contrast=True, normalize=True)
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
        return normalized.to(torch.float32)

    def _apply_tpm_log(self, genes: torch.Tensor) -> torch.Tensor:
        if self.gene_lengths is None:
            return genes
        gene_lengths = self.gene_lengths.unsqueeze(0)
        rpk = torch.where(gene_lengths > 0, genes / (gene_lengths / 1000.0), torch.zeros_like(genes))
        total_rpk = torch.sum(rpk, dim=1, keepdim=True)
        tpm = torch.where(total_rpk > 0, rpk / (total_rpk / 1e6), torch.zeros_like(rpk))
        return torch.log1p(tpm) / math.log(2.0)


# ---------------------------------------------------------------------------
# Collate and loss helpers
# ---------------------------------------------------------------------------


class GraphContrastiveCollator:
    def __init__(
        self,
        device: torch.device,
        graph_type: str,
        radius: float,
        knn_k: int,
        use_edge_attr: bool,
        edge_attr_dim: int,
        gene_similarity_threshold: float,
        gene_similarity_metric: str,
    ) -> None:
        self.device = device
        self.graph_type = graph_type
        self.radius = radius
        self.knn_k = knn_k
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim if use_edge_attr else 0
        self.gene_similarity_threshold = gene_similarity_threshold
        self.gene_similarity_metric = gene_similarity_metric

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
        sample_lengths: List[int] = []
        sample_names: List[str] = []
        patches_list: List[torch.Tensor] = []
        coords_list: List[torch.Tensor] = []
        genes_list: List[torch.Tensor] = []
        edge_indices: List[torch.Tensor] = []
        edge_attrs: List[torch.Tensor] = []
        offset = 0
        for sample in batch:
            sample_name = sample.get("sample_name", "unknown")
            sample_names.append(sample_name)
            patches = sample["patches"]
            coords = sample["coords"]
            genes = sample["genes"]
            length = coords.shape[0]
            sample_lengths.append(length)
            if length == 0:
                continue
            patches_list.append(patches)
            coords_list.append(coords)
            genes_list.append(genes)
            edge_index, edge_attr = self._build_graph(coords)
            if edge_index is not None and edge_index.numel() > 0:
                edge_indices.append(edge_index + offset)
                if edge_attr is not None:
                    edge_attrs.append(edge_attr)
            offset += length
        if not patches_list:
            logging.warning("All samples in batch were empty; skipping batch.")
            return None
        patches_tensor = torch.cat(patches_list, dim=0).to(self.device)
        coords_tensor = torch.cat(coords_list, dim=0).to(self.device)
        genes_tensor = torch.cat(genes_list, dim=0).to(self.device)
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
        positive_mask = self._build_positive_mask(
            genes_tensor,
            sample_lengths,
            edge_index_tensor,
        )
        return {
            "sample_names": sample_names,
            "sample_lengths": sample_lengths,
            "patches": patches_tensor,
            "coords": coords_tensor,
            "genes": genes_tensor,
            "edge_index": edge_index_tensor,
            "edge_attr": edge_attr_tensor,
            "positive_mask": positive_mask,
        }

    def _build_graph(self, coords: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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

    def _gene_similarity(self, genes: torch.Tensor) -> torch.Tensor:
        if self.gene_similarity_metric == "cosine":
            normed = F.normalize(genes, dim=1)
            return normed @ normed.T
        genes_centered = genes - genes.mean(dim=1, keepdim=True)
        normed = F.normalize(genes_centered, dim=1)
        return normed @ normed.T

    def _build_positive_mask(
        self,
        genes: torch.Tensor,
        sample_lengths: Sequence[int],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        total = genes.shape[0]
        positive_mask = torch.zeros(total, total, dtype=torch.bool, device=self.device)
        gene_sim = self._gene_similarity(genes)
        boundaries = torch.cumsum(torch.tensor([0] + list(sample_lengths), dtype=torch.long), dim=0)
        for i in range(len(sample_lengths)):
            start = boundaries[i].item()
            end = boundaries[i + 1].item()
            if start == end:
                continue
            sample_size = end - start
            adjacency = torch.zeros(sample_size, sample_size, dtype=torch.bool, device=self.device)
            if edge_index.numel() > 0:
                mask = (
                    (edge_index[0] >= start)
                    & (edge_index[0] < end)
                    & (edge_index[1] >= start)
                    & (edge_index[1] < end)
                )
                local_edges = edge_index[:, mask] - start
                if local_edges.numel() > 0:
                    adjacency[local_edges[0], local_edges[1]] = True
            sim_block = gene_sim[start:end, start:end] > self.gene_similarity_threshold
            sample_mask = adjacency & sim_block
            positive_mask[start:end, start:end] = sample_mask
        return positive_mask


def info_nce_loss(representations: torch.Tensor, positive_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    num_samples = representations.size(0)
    if positive_mask is None or positive_mask.sum() == 0:
        return torch.tensor(0.0, device=representations.device)
    reps = F.normalize(representations, dim=1)
    sim = torch.matmul(reps, reps.T) / temperature
    eye = torch.eye(num_samples, dtype=torch.bool, device=representations.device)
    sim = sim.masked_fill(eye, float("-inf"))
    row_max = sim.max(dim=1, keepdim=True).values
    stabilized = sim - row_max
    log_den = row_max + torch.logsumexp(stabilized, dim=1)
    pos_mask = positive_mask.clone()
    pos_mask[eye] = False
    anchors, positives = torch.where(pos_mask)
    if anchors.numel() == 0:
        return torch.tensor(0.0, device=representations.device)
    pos_log_prob = sim[anchors, positives] - log_den[anchors]
    counts = torch.zeros(num_samples, device=representations.device).scatter_add_(0, anchors, torch.ones_like(pos_log_prob))
    sums = torch.zeros(num_samples, device=representations.device).scatter_add_(0, anchors, pos_log_prob)
    valid = counts > 0
    return -(sums[valid] / counts[valid]).mean()


def compute_gene_correlations(preds: np.ndarray, targets: np.ndarray) -> float:
    correlations: List[float] = []
    num_genes = preds.shape[1]
    for gene_idx in range(num_genes):
        pred_gene = preds[:, gene_idx]
        true_gene = targets[:, gene_idx]
        if np.std(pred_gene) < 1e-6 or np.std(true_gene) < 1e-6:
            continue
        corr = np.corrcoef(pred_gene, true_gene)[0, 1]
        correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 0.0


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: GeneGraphRegressor,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    contrastive_weight: float,
    temperature: float,
) -> Dict[str, float]:
    model.train()
    total_reg = 0.0
    total_contrast = 0.0
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        if batch is None:
            continue
        patches = batch["patches"]
        coords = batch["coords"]
        genes = batch["genes"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        positive_mask = batch["positive_mask"]
        optimizer.zero_grad(set_to_none=True)
        preds, contrastive_features = model(patches, coords, edge_index, edge_attr)
        reg_loss = model.compute_loss(preds, genes)
        contrast_loss = info_nce_loss(contrastive_features, positive_mask, temperature)
        loss = reg_loss + contrastive_weight * contrast_loss
        loss.backward()
        optimizer.step()
        total_reg += reg_loss.item()
        total_contrast += contrast_loss.item()
        total_loss += loss.item()
        count += 1
    if count == 0:
        return {"loss": 0.0, "reg_loss": 0.0, "contrast_loss": 0.0}
    return {
        "loss": total_loss / count,
        "reg_loss": total_reg / count,
        "contrast_loss": total_contrast / count,
    }


def evaluate(
    model: GeneGraphRegressor,
    dataloader: Optional[DataLoader],
    contrastive_weight: float,
    temperature: float,
) -> Dict[str, float]:
    if dataloader is None:
        return {"loss": 0.0, "reg_loss": 0.0, "contrast_loss": 0.0, "correlation": 0.0}
    model.eval()
    total_reg = 0.0
    total_contrast = 0.0
    total_loss = 0.0
    count = 0
    preds_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            patches = batch["patches"]
            coords = batch["coords"]
            genes = batch["genes"]
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            positive_mask = batch["positive_mask"]
            preds, contrastive_features = model(patches, coords, edge_index, edge_attr)
            reg_loss = model.compute_loss(preds, genes)
            contrast_loss = info_nce_loss(contrastive_features, positive_mask, temperature)
            loss = reg_loss + contrastive_weight * contrast_loss
            total_reg += reg_loss.item()
            total_contrast += contrast_loss.item()
            total_loss += loss.item()
            count += 1
            preds_all.append(preds.cpu())
            targets_all.append(genes.cpu())
    if count == 0:
        return {"loss": 0.0, "reg_loss": 0.0, "contrast_loss": 0.0, "correlation": 0.0}
    preds_np = torch.cat(preds_all, dim=0).numpy() if preds_all else np.empty((0, model.predictor[-1].out_features))
    targets_np = torch.cat(targets_all, dim=0).numpy() if targets_all else np.empty_like(preds_np)
    correlation = compute_gene_correlations(preds_np, targets_np) if preds_np.size else 0.0
    return {
        "loss": total_loss / count,
        "reg_loss": total_reg / count,
        "contrast_loss": total_contrast / count,
        "correlation": correlation,
    }


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    collator = GraphContrastiveCollator(
        device=device,
        graph_type=args.graph_type,
        radius=args.radius,
        knn_k=args.knn_k,
        use_edge_attr=args.use_edge_attr,
        edge_attr_dim=args.edge_attr_dim,
        gene_similarity_threshold=args.gene_sim_threshold,
        gene_similarity_metric=args.gene_sim_metric,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collator,
    )


def prepare_datasets(
    args: argparse.Namespace,
    preprocess: Optional[Callable],
    tile_encoder: torch.nn.Module,
    device: torch.device,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset], int, int]:
    cache_root = args.cache_dir or (args.patch_root / "cache_training_embeddings")
    base_train_dataset = ExpressionDataset(
        csv_path=args.train_csv,
        patch_root=args.patch_root,
        preprocess=preprocess,
        tile_encoder=tile_encoder,
        device=device,
        cache_dir=cache_root / "train",
        patch_extension=args.patch_extension,
        gene_mapping_path=args.gene_mapping,
        gene_lengths_path=args.gene_lengths,
    )
    if args.val_csv:
        train_dataset: Dataset = base_train_dataset
        val_dataset: Optional[Dataset] = ExpressionDataset(
            csv_path=args.val_csv,
            patch_root=args.patch_root,
            preprocess=preprocess,
            tile_encoder=tile_encoder,
            device=device,
            cache_dir=cache_root / "val",
            patch_extension=args.patch_extension,
            gene_mapping_path=args.gene_mapping,
            gene_lengths_path=args.gene_lengths,
        )
    else:
        total_len = len(base_train_dataset)
        val_len = max(1, int(total_len * args.val_split))
        train_len = total_len - val_len
        if train_len <= 0:
            raise ValueError("Validation split too large; no training samples remain.")
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(base_train_dataset, [train_len, val_len], generator=generator)
        logging.info("Split training set into %d train / %d val samples.", train_len, val_len)
    test_dataset: Optional[Dataset] = None
    if args.test_csv:
        test_dataset = ExpressionDataset(
            csv_path=args.test_csv,
            patch_root=args.patch_root,
            preprocess=preprocess,
            tile_encoder=tile_encoder,
            device=device,
            cache_dir=cache_root / "test",
            patch_extension=args.patch_extension,
            gene_mapping_path=args.gene_mapping,
            gene_lengths_path=args.gene_lengths,
        )
    num_genes = base_train_dataset.num_genes
    backbone_dim = base_train_dataset.backbone_dim
    return train_dataset, val_dataset, test_dataset, num_genes, backbone_dim


def save_checkpoint(model: GeneGraphRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info("Saved checkpoint to %s", path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    args.patch_root = args.patch_root.expanduser()
    args.train_csv = args.train_csv.expanduser()
    if args.val_csv:
        args.val_csv = args.val_csv.expanduser()
    if args.test_csv:
        args.test_csv = args.test_csv.expanduser()
    if args.cache_dir:
        args.cache_dir = args.cache_dir.expanduser()
    args.annotations_dir = args.annotations_dir.expanduser()
    args.checkpoint_dir = args.checkpoint_dir.expanduser()
    gene_mapping_path, gene_lengths_path = _prepare_gene_annotations(
        args.gene_mapping.expanduser() if args.gene_mapping else None,
        args.gene_lengths.expanduser() if args.gene_lengths else None,
        args.annotations_dir,
    )
    args.gene_mapping = gene_mapping_path
    args.gene_lengths = gene_lengths_path
    device = resolve_device(args.device)
    tile_encoder, preprocess = load_tile_encoder(
        args.conch_model_id,
        args.conch_checkpoint,
        args.hf_token,
        device,
    )
    train_dataset, val_dataset, test_dataset, num_genes, backbone_dim = prepare_datasets(
        args,
        preprocess,
        tile_encoder,
        device,
    )
    train_loader = build_dataloader(train_dataset, args.batch_size_train, True, args, device)
    val_loader = build_dataloader(val_dataset, args.batch_size_eval, False, args, device) if val_dataset else None
    test_loader = build_dataloader(test_dataset, args.batch_size_eval, False, args, device) if test_dataset else None
    model = GeneGraphRegressor(
        backbone_feature_dim=backbone_dim,
        projected_feature_dim=args.projection_dim,
        position_dim=args.position_dim,
        output_dim=num_genes,
        dropout_rate=args.dropout,
        edge_attr_dim=args.edge_attr_dim if args.use_edge_attr else 0,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_corr = -float("inf")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Starting training for %d epochs.", args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, args.contrastive_weight, args.temperature)
        val_metrics = evaluate(model, val_loader, args.contrastive_weight, args.temperature)
        test_metrics = evaluate(model, test_loader, args.contrastive_weight, args.temperature) if test_loader else None
        logging.info(
            "Epoch %03d | train %.4f (reg %.4f / con %.4f) | val %.4f (reg %.4f / con %.4f, corr %.4f)",
            epoch,
            train_metrics["loss"],
            train_metrics["reg_loss"],
            train_metrics["contrast_loss"],
            val_metrics["loss"],
            val_metrics["reg_loss"],
            val_metrics["contrast_loss"],
            val_metrics["correlation"],
        )
        if test_metrics:
            logging.info(
                "Test %.4f (reg %.4f / con %.4f, corr %.4f)",
                test_metrics["loss"],
                test_metrics["reg_loss"],
                test_metrics["contrast_loss"],
                test_metrics["correlation"],
            )
        if val_metrics["correlation"] > best_val_corr:
            best_val_corr = val_metrics["correlation"]
            ckpt_name = f"{args.save_prefix}_epoch{epoch:03d}_corr{best_val_corr:.4f}.pth"
            save_checkpoint(model, args.checkpoint_dir / ckpt_name)
    logging.info("Training complete. Best validation correlation: %.4f", best_val_corr)


if __name__ == "__main__":
    main()

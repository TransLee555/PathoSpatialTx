"""MHGL-ST Stage3 OS archetype analysis (phases 1-2 pipeline)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from archetype_analysis.cohort_config import load_cohort_specs
from archetype_analysis.os_data_loader import (
    build_sample_dict,
    extract_all_data_for_analysis,
    get_sample,
    initialize_model,
    load_all_cohorts,
)
from archetype_analysis.prototype_analysis import assign_prototypes_with_conf_dual
from archetype_analysis.visualization import plot_umap_advanced_panels
from archetype_analysis.spatial_metrics import assemble_spatial_metrics
from archetype_analysis.wsi_overlay import (
    overlay_wsi_core_edge_for_sample,
    overlay_wsi_gate_for_sample,
    overlay_wsi_prototypes_for_sample,
    report_wsi_asset_status,
    _resolve_cohort_path,
)
from stage2_os_training import compute_c_index, hazards_to_risk

try:  # Optional dependencies used in DE reporting
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:  # pragma: no cover
    multipletests = None

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
except ImportError:  # pragma: no cover
    KaplanMeierFitter = CoxPHFitter = logrank_test = None


OUTPUT_LAYOUT = {
    "raw": "00_raw_embeddings",
    "processed": "01_processed_entities",
    "umap": "02_umap_visualizations",
    "spatial": "03_spatial_metrics",
    "survival": "04_survival",
    "ecology": "05_ecology_stats",
    "de": "06_interface_gene_de",
    "wsi": "07_wsi_overlays",
    "clinical": "08_clinical_validation",
}


def _canonical_gene_symbol(gene: str) -> str:
    s = str(gene).strip()
    return re.sub(r"[^A-Za-z0-9_.-]", "", s.upper())


def get_cell_feature_names(template_csv: Path) -> List[str]:
    cols_to_drop = {
        "Identifier.CentoidX_Global", "Identifier.CentoidY_Global",
        "Global_Nuclei_ID", "type",
        "Shape.HuMoments2", "Shape.HuMoments3", "Shape.HuMoments4",
        "Shape.HuMoments5", "Shape.HuMoments6", "Shape.HuMoments7",
        "Shape.WeightedHuMoments2", "Shape.WeightedHuMoments3",
        "Shape.WeightedHuMoments4", "Shape.WeightedHuMoments6",
    }
    one_hot_cats = ['Connective', 'Neoplastic', 'Dead', 'Epithelial', 'Inflammatory']
    all_cols = pd.read_csv(template_csv, nrows=0).columns.tolist()
    base_feats = [c for c in all_cols if c not in cols_to_drop]
    one_hot_feats = [f"Type_{cat}" for cat in one_hot_cats]
    return base_feats + one_hot_feats


def clean_gene_list(raw_genes: List[str]) -> List[str]:
    cleaned = [_canonical_gene_symbol(g) for g in raw_genes]
    dedup = []
    seen = set()
    for g in cleaned:
        if g and g not in seen:
            dedup.append(g)
            seen.add(g)
    return dedup


def _infer_input_dims(sample_index: Dict[str, Tuple[object, int]]) -> Tuple[int, int]:
    for ds, idx in sample_index.values():
        try:
            data = ds.get(idx)
            cell_dim = data['cell'].x.shape[1] if 'cell' in data.node_types else 0
            gene_dim = data['gene'].x.shape[1] if 'gene' in data.node_types else 0
            if cell_dim > 0 and gene_dim > 0:
                return cell_dim, gene_dim
        except Exception:
            continue
    raise RuntimeError("Unable to infer input dimensions from dataset.")


def _get_any_centers_shared(model, device):
    candidates = [
        "shared_cluster_centers", "shared_centers", "cluster_centers",
        "centers_shared", "centers", "prototypes", "prototype_centers",
        "W_shared", "W", "proto_weight",
    ]
    for name in candidates:
        if hasattr(model, name):
            tensor = getattr(model, name)
            if tensor is not None and torch.is_tensor(tensor) and tensor.ndim == 2 and tensor.numel() > 0:
                return tensor.detach().float().to(device)
    emb_dim = getattr(model, "embedding_dim", None)
    for name, param in model.named_parameters():
        if param.ndim != 2 or param.numel() == 0:
            continue
        if any(tok in name.lower() for tok in ("center", "proto", "cluster")):
            return param.detach().float().to(device)
        if emb_dim and param.size(1) == emb_dim:
            return param.detach().float().to(device)
    return None


def compute_auc_metrics(events: np.ndarray, risks: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    labels = events.astype(int)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        metrics["roc_auc"] = float('nan')
        metrics["pr_auc"] = float('nan')
        return metrics
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, risks))
    except Exception:
        metrics["roc_auc"] = float('nan')
    try:
        metrics["pr_auc"] = float(average_precision_score(labels, risks))
    except Exception:
        metrics["pr_auc"] = float('nan')
    return metrics


def _resolve_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        try:
            dev = torch.device(device_override)
            if dev.type.startswith('cuda') and not torch.cuda.is_available():
                logging.warning("Requested CUDA %s but CUDA unavailable; using CPU.", device_override)
                return torch.device('cpu')
            return dev
        except (RuntimeError, ValueError):
            logging.warning("Invalid device '%s'; falling back to auto selection.", device_override)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _prepare_output_dirs(base: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for key, rel in OUTPUT_LAYOUT.items():
        path = base / rel
        path.mkdir(parents=True, exist_ok=True)
        out[key] = path
    return out


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)


def _safe_to_parquet(df: pd.DataFrame, path: Path, label: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logging.info("Saved %s to %s (rows=%d, cols=%d)", label, path, len(df), len(df.columns))
    except Exception as exc:
        logging.warning("Failed to write %s to %s: %s", label, path, exc)


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        yield list(seq)
        return
    for start in range(0, len(seq), size):
        yield list(seq[start:start + size])


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token or "sample"


def _parse_pairs(raw: Optional[str]) -> Tuple[Tuple[int, int], ...]:
    if not raw:
        return tuple()
    pairs: List[Tuple[int, int]] = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            a, b = token.split('-')
            pairs.append((int(a), int(b)))
        except Exception:
            logging.warning("Could not parse spatial pair token '%s' (expected A-B).", token)
    return tuple(pairs)


def _parse_radii(raw: Optional[str]) -> Tuple[int, ...]:
    if not raw:
        return (100, 150, 200)
    vals: List[int] = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            logging.warning("Invalid radius token '%s' – skipping.", tok)
    return tuple(vals) if vals else (100, 150, 200)


def _subsample_dataframe(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _parse_int_list(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    vals = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            logging.warning("Could not parse integer token '%s'", tok)
    return vals


def _ensure_umap_coordinates(
    df: pd.DataFrame,
    entity: str,
    max_points: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> Tuple[pd.DataFrame, bool]:
    if df.empty:
        return df, False
    embed_cols = [c for c in df.columns if str(c).startswith('embed_')]
    if not embed_cols:
        logging.info("Skip UMAP for %s – no embed_* columns present.", entity)
        return df, False
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        logging.warning("UMAP dependency missing (%s); skipping %s visualization step.", exc, entity)
        return df, False

    subset = df.dropna(subset=embed_cols)
    if subset.empty:
        logging.info("Skip UMAP for %s – embeddings contain only NaNs.", entity)
        return df, False
    if max_points > 0 and len(subset) > max_points:
        subset = subset.sample(n=max_points, random_state=random_state)
    X = subset[embed_cols].to_numpy(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    reducer = umap.UMAP(
        n_neighbors=max(2, int(n_neighbors)),
        min_dist=max(0.001, float(min_dist)),
        n_components=2,
        random_state=random_state,
        densmap=False,
    )
    coords = reducer.fit_transform(X)
    df = df.copy()
    df['umap_x'] = np.nan
    df['umap_y'] = np.nan
    df.loc[subset.index, 'umap_x'] = coords[:, 0]
    df.loc[subset.index, 'umap_y'] = coords[:, 1]
    logging.info("Computed 2D UMAP for %s on %d rows.", entity, len(subset))
    return df, True


def _ensure_celltype_column(cell_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    if cell_df.empty:
        return cell_df
    out = cell_df.copy()
    type_cols = [c for c in feature_names if c.startswith('Type_') and c in out.columns]
    if type_cols:
        out['cell_type'] = out[type_cols].astype(float).idxmax(axis=1).str.replace('Type_', '', regex=False)
    elif 'cell_type' not in out.columns:
        out['cell_type'] = 'Unknown'
    out['cell_type'] = (
        out['cell_type']
        .astype(str)
        .str.strip()
        .replace({'': 'Unknown', 'nan': 'Unknown', 'None': 'Unknown', 'NA': 'Unknown'})
        .fillna('Unknown')
    )
    return out


def compute_local_ecology_kNN(cell_df: pd.DataFrame, k: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    need = {'pos_x', 'pos_y', 'prototype_id'}
    if not need.issubset(cell_df.columns):
        empty = pd.DataFrame(index=cell_df.index)
        logging.warning("Local ecology skipped – missing columns %s", need - set(cell_df.columns))
        return empty, empty
    df = cell_df.dropna(subset=['pos_x', 'pos_y', 'prototype_id']).copy()
    if len(df) < max(3, k + 1):
        empty = pd.DataFrame(index=cell_df.index)
        logging.warning("Local ecology skipped – insufficient cells with coordinates (%d).", len(df))
        return empty, empty
    try:
        from sklearn.neighbors import KDTree
    except ImportError as exc:
        logging.warning("scikit-learn missing (%s); cannot compute local ecology.", exc)
        empty = pd.DataFrame(index=cell_df.index)
        return empty, empty

    coords = df[['pos_x', 'pos_y']].to_numpy(np.float32)
    neigh = KDTree(coords).query(coords, k=min(len(df), k + 1), return_distance=False)[:, 1:]

    proto = pd.to_numeric(df['prototype_id'], errors='coerce').astype(int).to_numpy()
    uniq_proto = np.unique(proto)
    proto_map = {p: i for i, p in enumerate(uniq_proto)}
    proto_mix = np.zeros((len(df), len(uniq_proto)), dtype=np.float32)
    for row, idx in enumerate(neigh):
        np.add.at(proto_mix[row], [proto_map[p] for p in proto[idx]], 1)
    proto_mix /= np.maximum(proto_mix.sum(axis=1, keepdims=True), 1e-6)
    neigh_proto_prop = pd.DataFrame(proto_mix, index=df.index, columns=[f'P{int(p)}' for p in uniq_proto])

    cell_types = df.get('cell_type', pd.Series('Unknown', index=df.index)).astype(str).fillna('Unknown').to_numpy()
    uniq_types = np.unique(cell_types)
    type_map = {t: i for i, t in enumerate(uniq_types)}
    type_mix = np.zeros((len(df), len(uniq_types)), dtype=np.float32)
    for row, idx in enumerate(neigh):
        np.add.at(type_mix[row], [type_map[t] for t in cell_types[idx]], 1)
    type_mix /= np.maximum(type_mix.sum(axis=1, keepdims=True), 1e-6)
    neigh_type_prop = pd.DataFrame(type_mix, index=df.index, columns=[f'CT:{t}' for t in uniq_types])

    neigh_proto_prop = neigh_proto_prop.reindex(index=cell_df.index).fillna(0.0)
    neigh_type_prop = neigh_type_prop.reindex(index=cell_df.index).fillna(0.0)
    return neigh_proto_prop, neigh_type_prop


def aggregate_ecology_by_anchor(
    cell_df: pd.DataFrame,
    neigh_proto_prop: pd.DataFrame,
    neigh_type_prop: pd.DataFrame,
    min_cells: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cell_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    proto_mix: Dict[int, pd.Series] = {}
    type_mix: Dict[int, pd.Series] = {}
    for pid, sdf in cell_df.groupby('prototype_id'):
        if len(sdf) < max(1, int(min_cells)):
            continue
        idx = sdf.index
        if not neigh_proto_prop.empty:
            proto_mix[int(pid)] = neigh_proto_prop.loc[idx].mean(axis=0)
        if not neigh_type_prop.empty:
            type_mix[int(pid)] = neigh_type_prop.loc[idx].mean(axis=0)
    proto_df = pd.DataFrame.from_dict(proto_mix, orient='index').sort_index()
    type_df = pd.DataFrame.from_dict(type_mix, orient='index').sort_index()
    return proto_df, type_df


def add_interface_strength_columns(
    cell_df: pd.DataFrame,
    thr_exist: float = 0.05,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    df = cell_df.copy()
    pcols = [c for c in df.columns if c.startswith('P') and c[1:].isdigit()]
    if not pcols:
        logging.warning("Interface column builder skipped – no P* columns found.")
        return df, []
    proto_ids = sorted({int(c[1:]) for c in pcols})
    pairs: List[Tuple[int, int]] = []
    group_key = 'sample_name' if 'sample_name' in df.columns else None
    for idx_i, i in enumerate(proto_ids):
        col_i = f'P{i}'
        for j in proto_ids[idx_i + 1:]:
            col_j = f'P{j}'
            if col_j not in df.columns:
                continue
            exists = False
            if group_key:
                for _, sdf in df.groupby(group_key):
                    mean_i = float(pd.to_numeric(sdf[col_i], errors='coerce').fillna(0.0).mean())
                    mean_j = float(pd.to_numeric(sdf[col_j], errors='coerce').fillna(0.0).mean())
                    if mean_i >= thr_exist and mean_j >= thr_exist:
                        exists = True
                        break
            else:
                exists = True
            if not exists:
                continue
            pairs.append((i, j))
            df[f'IF_P{i}_P{j}'] = np.minimum(
                pd.to_numeric(df[col_i], errors='coerce').fillna(0.0),
                pd.to_numeric(df[col_j], errors='coerce').fillna(0.0),
            ).astype(np.float32)
    return df, pairs


def summarize_interface_pairs(
    cell_df: pd.DataFrame,
    pairs: Sequence[Tuple[int, int]],
    thr_if: float,
) -> pd.DataFrame:
    if not pairs or 'sample_name' not in cell_df.columns:
        return pd.DataFrame()
    rows = []
    for sample_name, sdf in cell_df.groupby('sample_name'):
        rec = {'sample_name': sample_name}
        for i, j in pairs:
            col = f'IF_P{i}_P{j}'
            if col not in sdf.columns:
                continue
            vals = pd.to_numeric(sdf[col], errors='coerce').fillna(0.0)
            rec[f'mean_IF_P{i}_P{j}'] = float(vals.mean())
            rec[f'n_cells_ge_thr_P{i}_P{j}'] = int((vals >= float(thr_if)).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def load_gene_sets(json_path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if not json_path:
        return {}
    path = Path(json_path)
    if not path.exists():
        logging.warning("Gene set file not found: %s", path)
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        logging.warning("Failed to load gene set json %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning("Gene set json %s malformed (expect dict).", path)
        return {}
    gene_sets = {}
    for name, cfg in data.items():
        genes = cfg.get('genes') if isinstance(cfg, dict) else None
        if not isinstance(genes, (list, tuple)):
            continue
        gene_sets[name] = {
            'genes': [str(g).strip() for g in genes if str(g).strip()],
            'direction': str(cfg.get('direction', 'two-sided')) if isinstance(cfg, dict) else 'two-sided',
        }
    return gene_sets


def compute_center_similarity(centers: torch.Tensor) -> np.ndarray:
    if centers is None or centers.numel() == 0:
        return np.zeros((0, 0), dtype=np.float32)
    array = centers.detach().float().cpu().numpy()
    array /= np.linalg.norm(array, axis=1, keepdims=True) + 1e-9
    sim = array @ array.T
    sim = np.clip(sim, -1.0, 1.0)
    return sim.astype(np.float32)


def assign_proto_hubs(sim: np.ndarray, thr: float, min_size: int) -> Dict[int, int]:
    K = sim.shape[0]
    if K == 0:
        return {}
    visited = set()
    hub_map: Dict[int, int] = {}
    hub_id = 0
    for pid in range(K):
        if pid in visited:
            continue
        stack = [pid]
        component = set()
        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            for nbr in range(K):
                if nbr == cur or sim[cur, nbr] < thr:
                    continue
                if nbr not in component:
                    stack.append(nbr)
        visited |= component
        if len(component) < max(1, min_size):
            for node in component:
                hub_map[node] = hub_id
                hub_id += 1
        else:
            for node in sorted(component):
                hub_map[node] = hub_id
            hub_id += 1
    return hub_map


def run_hub_analysis(
    centers_tensor: Optional[torch.Tensor],
    args: argparse.Namespace,
    out_dirs: Dict[str, Path],
    cell_df: pd.DataFrame,
) -> Dict[str, object]:
    outputs: Dict[str, object] = {}
    if centers_tensor is None or centers_tensor.numel() == 0:
        return outputs
    sim = compute_center_similarity(centers_tensor)
    if sim.size == 0:
        return outputs
    proto_ids = [f'P{i}' for i in range(sim.shape[0])]
    sim_path = out_dirs['ecology'] / "hub_similarity.csv"
    pd.DataFrame(sim, index=proto_ids, columns=proto_ids).to_csv(sim_path)
    hub_map = assign_proto_hubs(sim, float(getattr(args, 'hub_sim_thr', 0.7)), int(getattr(args, 'hub_min_size', 1)))
    proto2hub = pd.DataFrame(
        {
            'prototype_id': list(hub_map.keys()),
            'hub_id': [hub_map[k] for k in hub_map],
        }
    ).sort_values(['hub_id', 'prototype_id'])
    proto2hub_path = out_dirs['ecology'] / "proto_to_hub.csv"
    proto2hub.to_csv(proto2hub_path, index=False)
    hub_stats = {}
    if not cell_df.empty and 'prototype_id' in cell_df.columns:
        proto_counts = pd.to_numeric(cell_df['prototype_id'], errors='coerce').dropna().astype(int)
        for pid, hub in hub_map.items():
            hub_stats.setdefault(hub, 0)
            hub_stats[hub] += int((proto_counts == pid).sum())
    hub_stats_path = out_dirs['ecology'] / "hub_counts.csv"
    pd.DataFrame([
        {'hub_id': hub, 'cell_count': count}
        for hub, count in sorted(hub_stats.items())
    ]).to_csv(hub_stats_path, index=False)

    max_components = max(1, int(getattr(args, 'hub_max_components', 3)))
    try:
        deg = np.diag(sim.sum(axis=1))
        lap = deg - sim
        eigvals, eigvecs = np.linalg.eigh(lap)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        coords = eigvecs[:, 1:1 + max_components]
        coord_cols = [f'd{i+1}' for i in range(coords.shape[1])]
        coords_df = pd.DataFrame(coords, columns=coord_cols)
        coords_df.insert(0, 'prototype_id', proto_ids)
        hub_coord_path = out_dirs['ecology'] / "hub_diffusion_coords.csv"
        coords_df.to_csv(hub_coord_path, index=False)
        outputs['hub_coords'] = str(hub_coord_path)
    except Exception as exc:
        logging.warning("Hub diffusion computation failed: %s", exc)

    outputs.update({
        'similarity_csv': str(sim_path),
        'proto_to_hub_csv': str(proto2hub_path),
        'hub_counts_csv': str(hub_stats_path),
    })
    return outputs


def export_anchor_focus_stats(
    cell_df: pd.DataFrame,
    anchor_pids: Sequence[int],
    prop_thr: float,
    out_dir: Path,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if not anchor_pids:
        return outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_rows = []
    type_rows = []
    for pid in anchor_pids:
        col = f'P{pid}'
        if col not in cell_df.columns:
            continue
        subset = cell_df[pd.to_numeric(cell_df[col], errors='coerce').fillna(0.0) >= float(prop_thr)]
        if subset.empty:
            continue
        if 'prototype_id' in subset.columns:
            proto_counts = (
                pd.to_numeric(subset['prototype_id'], errors='coerce')
                .dropna()
                .astype(int)
                .value_counts(normalize=True)
            )
            for proto_id, frac in proto_counts.items():
                proto_rows.append({'anchor_pid': pid, 'prototype_id': int(proto_id), 'fraction': float(frac)})
        if 'cell_type' in subset.columns:
            type_counts = subset['cell_type'].astype(str).value_counts(normalize=True)
            for ctype, frac in type_counts.items():
                type_rows.append({'anchor_pid': pid, 'cell_type': ctype, 'fraction': float(frac)})
    if proto_rows:
        proto_path = out_dir / "anchor_proto_focus.csv"
        pd.DataFrame(proto_rows).to_csv(proto_path, index=False)
        outputs['anchor_proto_focus'] = str(proto_path)
    if type_rows:
        type_path = out_dir / "anchor_celltype_focus.csv"
        pd.DataFrame(type_rows).to_csv(type_path, index=False)
        outputs['anchor_celltype_focus'] = str(type_path)
    return outputs


def prepare_full_resolution_cells(
    cell_df: pd.DataFrame,
    args: argparse.Namespace,
    out_dirs: Dict[str, Path],
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    cache_enabled = bool(args.spatial_cache_fullres or args.spatial_use_cached_cells)
    if not cache_enabled:
        return None, None, None
    cache_dir = Path(args.spatial_cache_dir or (out_dirs['ecology'] / "fullres_cache"))
    cache_path = cache_dir / str(args.spatial_cache_fname)
    per_sample_manifest = None
    if args.spatial_use_cached_cells and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logging.info("Loaded cached full-resolution cells from %s (rows=%d)", cache_path, len(df))
            return df, str(cache_path), None
        except Exception as exc:
            logging.warning("Failed to load cached cells from %s: %s", cache_path, exc)
    if not args.spatial_cache_fullres:
        return None, None, None
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_to_cache = cell_df.copy()
    max_cells = max(0, int(args.spatial_max_cells or 0))
    if max_cells and len(df_to_cache) > max_cells:
        df_to_cache = _subsample_dataframe(df_to_cache, max_cells, seed=int(args.spatial_rng_seed))
        logging.info("Subsampled cells to %d rows for cache.", len(df_to_cache))
    _safe_to_parquet(df_to_cache, cache_path, "full_cell_cache")
    if args.spatial_cache_per_sample and 'sample_name' in df_to_cache.columns:
        per_sample_dir = cache_dir / "per_sample"
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for sample, sdf in df_to_cache.groupby('sample_name'):
            safe_name = _sanitize_token(sample)
            sample_path = per_sample_dir / f"{safe_name}.parquet"
            try:
                sdf.to_parquet(sample_path)
                manifest_rows.append({
                    'sample_name': sample,
                    'path': str(sample_path),
                    'rows': int(len(sdf)),
                })
            except Exception as exc:
                logging.warning("Failed to write per-sample cache for %s: %s", sample, exc)
        if manifest_rows:
            per_sample_manifest = str(per_sample_dir / "index.json")
            _write_json(Path(per_sample_manifest), {'samples': manifest_rows})
    return (df_to_cache if args.spatial_use_cached_cells else None), str(cache_path), per_sample_manifest


def _compute_spatial_metrics_core(
    df: pd.DataFrame,
    pairs: Tuple[Tuple[int, int], ...],
    radii: Tuple[int, ...],
    args: argparse.Namespace,
):
    try:
        metrics = assemble_spatial_metrics(
            df,
            k=args.spatial_k,
            radii=radii,
            focus_proto=args.focus_proto,
            pairs=pairs or ((2, 0), (2, 4)),
            microns_per_pixel=args.microns_per_pixel,
            core_thr=args.core_thr,
            excl_thr=args.excl_thr,
            interface_thr=args.interface_thr,
        )
        return metrics
    except Exception as exc:
        logging.warning("Spatial metrics computation failed: %s", exc)
        return None


def compute_spatial_metrics_batched(
    cell_df: pd.DataFrame,
    args: argparse.Namespace,
    pairs: Tuple[Tuple[int, int], ...],
    radii: Tuple[int, ...],
    out_dirs: Dict[str, Path],
) -> Optional[Path]:
    if args.skip_spatial_analysis:
        logging.info("Spatial analysis skipped via --skip-spatial-analysis flag.")
        return None
    if cell_df.empty:
        logging.warning("Spatial analysis skipped – no cell rows available.")
        return None
    df = _subsample_dataframe(cell_df, max(0, int(args.spatial_max_cells or 0)), seed=int(args.spatial_rng_seed))
    metrics_frames = []
    batch_size = int(args.spatial_batch_size or 0)
    if batch_size > 0 and 'sample_name' in df.columns:
        samples = sorted(df['sample_name'].dropna().astype(str).unique().tolist())
        for chunk in _chunked(samples, batch_size):
            chunk_df = df[df['sample_name'].isin(chunk)]
            if chunk_df.empty:
                continue
            metrics = _compute_spatial_metrics_core(chunk_df, pairs, radii, args)
            if metrics is not None and not metrics.empty:
                metrics_frames.append(metrics)
    else:
        metrics = _compute_spatial_metrics_core(df, pairs, radii, args)
        if metrics is not None and not metrics.empty:
            metrics_frames.append(metrics)
    if not metrics_frames:
        logging.warning("Spatial metrics results are empty; no CSV emitted.")
        return None
    combined = pd.concat(metrics_frames, axis=0)
    combined = combined[~combined.index.duplicated(keep='last')]
    metrics_path = out_dirs['spatial'] / "spatial_metrics_summary.csv"
    combined.to_csv(metrics_path)
    logging.info("Spatial metrics saved to %s", metrics_path)
    return metrics_path


def run_wsi_overlays(
    cell_df: pd.DataFrame,
    args: argparse.Namespace,
    out_dirs: Dict[str, Path],
    cohort_svs: Dict[str, str],
    cohort_geojson: Dict[str, str],
    allowlist: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    outputs: Dict[str, object] = {}
    if not getattr(args, 'enable_wsi_overlays', False):
        return outputs
    if not cohort_svs or not cohort_geojson:
        logging.warning("WSI overlay skipped – cohort SVS/GeoJSON mappings not provided.")
        return outputs
    if 'sample_name' not in cell_df.columns:
        logging.warning("WSI overlay skipped – sample_name column missing.")
        return outputs
    out_dir = out_dirs['wsi']
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        asset_csv = out_dir / "wsi_asset_status.csv"
        report_wsi_asset_status(
            cell_df,
            cohort_svs,
            cohort_geojson,
            out_csv=str(asset_csv),
            allow_cohorts=allowlist,
        )
        outputs['asset_csv'] = str(asset_csv)
    except Exception as exc:
        logging.warning("WSI asset status failed: %s", exc)

    max_samples = max(0, int(getattr(args, 'wsi_max_samples', 0)))
    sample_counts = (
        cell_df.groupby('sample_name')['prototype_id']
        .count()
        .sort_values(ascending=False)
    )
    sample_names = sample_counts.index.tolist()
    if max_samples and len(sample_names) > max_samples:
        sample_names = sample_names[:max_samples]
    prototype_outputs = []
    core_edge_outputs = []
    gate_outputs = []
    interface_outputs: List[str] = []
    interface_pairs = _parse_pairs(getattr(args, 'wsi_interface_pairs', None))
    for sample in sample_names:
        sdf = cell_df[cell_df['sample_name'] == sample]
        if sdf.empty:
            continue
        cohort_val = sdf['cohort'].iat[0] if 'cohort' in sdf.columns else None
        svs_root = _resolve_cohort_path(cohort_svs, cohort_val)
        gj_root = _resolve_cohort_path(cohort_geojson, cohort_val)
        if not svs_root or not gj_root:
            continue
        safe_name = _sanitize_token(sample)
        base_dir = out_dir / safe_name
        base_dir.mkdir(parents=True, exist_ok=True)
        proto_png = base_dir / f"{safe_name}_prototypes.png"
        try:
            overlay_wsi_prototypes_for_sample(
                sample,
                cell_df,
                svs_root,
                gj_root,
                out_png=str(proto_png),
                use_merged=True,
            )
            prototype_outputs.append(str(proto_png))
        except Exception as exc:
            logging.warning("WSI prototype overlay failed for %s: %s", sample, exc)
        anchor_pid = int(getattr(args, 'wsi_anchor_proto', 2))
        tag = f'P{anchor_pid}'
        if tag in cell_df.columns:
            core_png = base_dir / f"{safe_name}_core_edge.png"
            try:
                overlay_wsi_core_edge_for_sample(
                    sample,
                    cell_df,
                    svs_root,
                    gj_root,
                    out_png=str(core_png),
                    anchor_pid=anchor_pid,
                    prop_thr=float(getattr(args, 'wsi_core_thr', 0.7)),
                )
                core_edge_outputs.append(str(core_png))
            except Exception as exc:
                logging.warning("WSI core-edge overlay failed for %s: %s", sample, exc)
        gate_col = getattr(args, 'wsi_gate_column', None)
        if gate_col and gate_col in cell_df.columns:
            gate_png = base_dir / f"{safe_name}_gate.png"
            try:
                overlay_wsi_gate_for_sample(
                    sample,
                    cell_df,
                    svs_root,
                    gj_root,
                    out_png=str(gate_png),
                    scalar_col=gate_col,
                )
                gate_outputs.append(str(gate_png))
            except Exception as exc:
                logging.warning("WSI gate overlay failed for %s: %s", sample, exc)
        if interface_pairs:
            for pair in interface_pairs:
                col = f'IF_P{pair[0]}_P{pair[1]}'
                if col not in cell_df.columns:
                    continue
                iface_png = base_dir / f"{safe_name}_IF_P{pair[0]}_P{pair[1]}.png"
                try:
                    overlay_wsi_gate_for_sample(
                        sample,
                        cell_df,
                        svs_root,
                        gj_root,
                        out_png=str(iface_png),
                        scalar_col=col,
                    )
                    interface_outputs.append(str(iface_png))
                except Exception as exc:
                    logging.warning("WSI interface overlay failed for %s pair %s: %s", sample, pair, exc)
    outputs['prototype_overlays'] = prototype_outputs
    outputs['core_edge_overlays'] = core_edge_outputs
    outputs['gate_overlays'] = gate_outputs
    outputs['interface_overlays'] = interface_outputs
    return outputs


def run_clinical_survival_checks(surv_df: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    outputs: Dict[str, object] = {}
    if surv_df is None or surv_df.empty:
        return outputs
    df = surv_df.copy()
    df = df[['duration', 'event', 'risk_score']].dropna()
    if df.empty:
        return outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    df['event'] = df['event'].astype(int)
    median_risk = float(df['risk_score'].median())
    df['risk_group'] = np.where(df['risk_score'] >= median_risk, 'High', 'Low')
    risk_summary = df.groupby('risk_group').agg(
        n_samples=('risk_score', 'size'),
        mean_risk=('risk_score', 'mean'),
        mean_duration=('duration', 'mean'),
        event_rate=('event', 'mean'),
    ).reset_index()
    risk_summary_path = out_dir / "risk_group_summary.csv"
    risk_summary.to_csv(risk_summary_path, index=False)
    outputs['risk_group_summary'] = str(risk_summary_path)

    if CoxPHFitter is not None:
        try:
            cph = CoxPHFitter()
            cph.fit(df[['duration', 'event', 'risk_score']], duration_col='duration', event_col='event')
            cox_path = out_dir / "cox_summary.csv"
            cph.summary.to_csv(cox_path)
            outputs['cox_summary'] = str(cox_path)
        except Exception as exc:
            logging.warning("CoxPH fitting failed: %s", exc)
    else:
        logging.info("lifelines CoxPHFitter unavailable; skipping Cox summary.")

    km_outputs = []
    if KaplanMeierFitter is not None:
        try:
            for group, sdf in df.groupby('risk_group'):
                if len(sdf) < 3:
                    continue
                km = KaplanMeierFitter()
                km.fit(sdf['duration'], event_observed=sdf['event'], label=str(group))
                km_path = out_dir / f"km_{group}.csv"
                km.survival_function_.to_csv(km_path)
                km_outputs.append(str(km_path))
            if km_outputs and logrank_test is not None:
                try:
                    high = df[df['risk_group'] == 'High']
                    low = df[df['risk_group'] == 'Low']
                    if len(high) >= 3 and len(low) >= 3:
                        lr = logrank_test(high['duration'], low['duration'], high['event'], low['event'])
                        lr_path = out_dir / "logrank_test.json"
                        _write_json(lr_path, {'p_value': float(lr.p_value)})
                        outputs['logrank'] = str(lr_path)
                except Exception as exc:
                    logging.warning("Log-rank test failed: %s", exc)
        except Exception as exc:
            logging.warning("KM analysis failed: %s", exc)
    else:
        logging.info("lifelines KaplanMeierFitter unavailable; skipping KM curves.")
    outputs['km_curves'] = km_outputs
    return outputs


def _winsor_clip(arr: np.ndarray, quantile: float) -> np.ndarray:
    if quantile <= 0 or not np.isfinite(arr).any():
        return arr
    low = np.nanquantile(arr, quantile, axis=0)
    high = np.nanquantile(arr, 1 - quantile, axis=0)
    return np.clip(arr, low, high)


def _aggregate_group_expression(
    X: np.ndarray,
    agg_method: str = 'median',
    winsor_q: float = 0.0,
) -> Optional[np.ndarray]:
    if X is None or X.size == 0:
        return None
    X_proc = _winsor_clip(X, winsor_q) if winsor_q > 0 else X
    if agg_method == 'mean':
        return np.nanmean(X_proc, axis=0)
    return np.nanmedian(X_proc, axis=0)


def _cg_weighted_cell_gene_matrix(
    sdf: pd.DataFrame,
    gdf: pd.DataFrame,
    gene_cols: List[str],
    data_sample,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    try:
        edge_index = data_sample['cell', 'c_g', 'gene'].edge_index.detach().cpu().numpy()
    except Exception:
        return None, None, 0
    rows_all = edge_index[0].astype(int)
    cols_all = edge_index[1].astype(int)

    if 'node_idx' in sdf.columns:
        cell_ids = pd.to_numeric(sdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        cell_id2row = {cid: i for i, cid in enumerate(cell_ids)}
        n_cells = len(cell_ids)
    else:
        n_cells = len(sdf)
        cell_id2row = {i: i for i in range(n_cells)}

    if 'node_idx' in gdf.columns:
        gene_ids = pd.to_numeric(gdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        gene_id2col = {gid: i for i, gid in enumerate(gene_ids)}
    else:
        gene_id2col = {i: i for i in range(len(gdf))}

    try:
        edge_attr = getattr(data_sample['cell', 'c_g', 'gene'], 'edge_attr', None)
        weights_all = (
            edge_attr.detach().cpu().numpy().reshape(-1)
            if edge_attr is not None and edge_attr.numel() > 0
            else np.ones(len(rows_all), dtype=np.float32)
        )
    except Exception:
        weights_all = np.ones(len(rows_all), dtype=np.float32)

    keep = np.array([(r in cell_id2row) and (c in gene_id2col) for r, c in zip(rows_all, cols_all)], dtype=bool)
    if not np.any(keep):
        return None, None, 0
    rows = rows_all[keep]
    cols = cols_all[keep]
    weights = weights_all[keep]

    gene_matrix = gdf[gene_cols].to_numpy(np.float32, copy=True)
    X_hat = np.zeros((n_cells, len(gene_cols)), dtype=np.float32)
    row_weights = np.zeros((n_cells,), dtype=np.float32)
    for r, c, w in zip(rows, cols, weights):
        rr = cell_id2row.get(int(r))
        cc = gene_id2col.get(int(c))
        if rr is None or cc is None:
            continue
        X_hat[rr] += w * gene_matrix[cc]
        row_weights[rr] += w
    valid = row_weights > 0
    X_hat[valid] = X_hat[valid] / row_weights[valid][:, None]
    return X_hat, valid, len(rows)


def _resolve_gene_names(gene_cols: List[str], gene_list: List[str]) -> List[str]:
    resolved = []
    for col in gene_cols:
        name = col
        if col.startswith('gene_expr_'):
            try:
                idx = int(col.split('_')[-1])
                if 0 <= idx < len(gene_list):
                    name = gene_list[idx]
            except Exception:
                name = col
        resolved.append(str(name))
    return resolved


def run_interface_gene_de(
    cell_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    gene_list: List[str],
    sample_dict: Dict[str, torch.Tensor],
    pairs: Sequence[Tuple[int, int]],
    out_dir: Path,
    t_if: float,
    t_core: float,
    min_cells: int,
    agg_method: str,
    winsor_q: float,
    log2fc_thr: float,
    p_thr: float,
    test_method: str,
    gene_sets: Optional[Dict[str, Dict[str, object]]] = None,
    debug: bool = False,
) -> List[Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_cols = [c for c in gene_df.columns if c.startswith('gene_expr_')]
    if not gene_cols:
        logging.warning("Interface DE skipped – gene_df lacks gene_expr_* columns.")
        return []
    gene_names = _resolve_gene_names(gene_cols, gene_list)
    eps = 1e-6
    reports: List[Dict[str, object]] = []
    gene_sets_idx: Dict[str, Dict[str, object]] = {}
    if gene_sets:
        gene_upper = {name.upper(): idx for idx, name in enumerate(gene_names)}
        for set_name, cfg in gene_sets.items():
            idxs = []
            for g in cfg.get('genes', []):
                idx = gene_upper.get(str(g).upper())
                if idx is not None:
                    idxs.append(idx)
            if idxs:
                gene_sets_idx[set_name] = {
                    'indices': idxs,
                    'direction': cfg.get('direction', 'two-sided'),
                }

    for i, j in pairs:
        if_col = f'IF_P{i}_P{j}'
        Pi_col = f'P{i}'
        Pj_col = f'P{j}'
        if if_col not in cell_df.columns or Pi_col not in cell_df.columns or Pj_col not in cell_df.columns:
            continue
        per_if: List[np.ndarray] = []
        per_core: List[np.ndarray] = []
        for sample_name, sdf in cell_df.groupby('sample_name'):
            sdf_gene = gene_df[gene_df['sample_name'] == sample_name]
            if sdf_gene.empty:
                continue
            sample_graph = get_sample(sample_dict, sample_name)
            if sample_graph is None:
                continue
            Pi = pd.to_numeric(sdf[Pi_col], errors='coerce').fillna(0.0).to_numpy()
            Pj = pd.to_numeric(sdf[Pj_col], errors='coerce').fillna(0.0).to_numpy()
            if_vals = pd.to_numeric(sdf[if_col], errors='coerce').fillna(0.0).to_numpy()
            interface_mask = if_vals >= float(t_if)
            core_mask = ((Pi >= float(t_core)) & (Pj <= 1.0 - float(t_if))) | (
                (Pj >= float(t_core)) & (Pi <= 1.0 - float(t_if))
            )
            if interface_mask.sum() < min_cells or core_mask.sum() < min_cells:
                continue
            X_hat, valid_mask, used_edges = _cg_weighted_cell_gene_matrix(sdf, sdf_gene, gene_cols, sample_graph)
            if X_hat is None or used_edges == 0:
                continue
            valid = valid_mask.astype(bool) if valid_mask is not None else np.ones(len(X_hat), dtype=bool)
            if_profiles = _aggregate_group_expression(
                X_hat[interface_mask & valid], agg_method=agg_method, winsor_q=winsor_q
            )
            core_profiles = _aggregate_group_expression(
                X_hat[core_mask & valid], agg_method=agg_method, winsor_q=winsor_q
            )
            if if_profiles is None or core_profiles is None:
                continue
            per_if.append(if_profiles)
            per_core.append(core_profiles)
        if len(per_if) < 2:
            logging.info("[IF-DE] Skip pair (%d,%d) – insufficient samples.", i, j)
            continue
        if_arr = np.vstack(per_if)
        core_arr = np.vstack(per_core)
        mean_if = np.nanmean(if_arr, axis=0)
        mean_core = np.nanmean(core_arr, axis=0)
        log2fc = np.log2(mean_if + eps) - np.log2(mean_core + eps)
        p_values = np.full(len(gene_cols), np.nan, dtype=np.float32)
        for idx in range(len(gene_cols)):
            mask = np.isfinite(if_arr[:, idx]) & np.isfinite(core_arr[:, idx])
            if mask.sum() < 2:
                continue
            x = if_arr[mask, idx]
            y = core_arr[mask, idx]
            if stats is not None and test_method == 'wilcoxon':
                try:
                    _, pval = stats.wilcoxon(x, y)
                except ValueError:
                    pval = np.nan
            elif stats is not None and test_method == 'ttest':
                _, pval = stats.ttest_rel(x, y, nan_policy='omit')
            else:
                diff = x - y
                var = np.nanvar(diff)
                if var <= 1e-9:
                    pval = np.nan
                else:
                    z = np.nanmean(diff) / np.sqrt(var / len(diff))
                    pval = float(math.erfc(abs(z) / math.sqrt(2.0)))
            p_values[idx] = pval
        q_values = np.full_like(p_values, np.nan)
        if multipletests is not None:
            try:
                mask = np.isfinite(p_values)
                if mask.any():
                    _, qvals, _, _ = multipletests(p_values[mask], method='fdr_bh')
                    q_values[mask] = qvals
            except Exception:
                pass
        result_df = pd.DataFrame(
            {
                'gene': gene_names,
                'log2fc': log2fc,
                'mean_interface': mean_if,
                'mean_core': mean_core,
                'p_value': p_values,
                'q_value': q_values,
            }
        )
        full_path = out_dir / f"P{i}_P{j}_interface_de_full.csv"
        result_df.to_csv(full_path, index=False)
        sig_mask = (
            np.abs(result_df['log2fc']) >= float(log2fc_thr)
        ) & (result_df['p_value'] <= float(p_thr))
        sig_path = None
        gene_set_path = None
        if sig_mask.any():
            sig_path = out_dir / f"P{i}_P{j}_interface_DEGs.csv"
            result_df[sig_mask].sort_values('p_value').to_csv(sig_path, index=False)
        if gene_sets_idx:
            gs_rows = []
            for gs_name, cfg in gene_sets_idx.items():
                idxs = cfg['indices']
                if_arr_set = np.nanmean(if_arr[:, idxs], axis=1)
                core_arr_set = np.nanmean(core_arr[:, idxs], axis=1)
                mask = np.isfinite(if_arr_set) & np.isfinite(core_arr_set)
                if mask.sum() < 2:
                    continue
                set_if_mean = float(np.nanmean(if_arr_set))
                set_core_mean = float(np.nanmean(core_arr_set))
                set_log2fc = float(np.log2(set_if_mean + eps) - np.log2(set_core_mean + eps))
                diff = if_arr_set[mask] - core_arr_set[mask]
                if stats is not None and test_method == 'wilcoxon':
                    try:
                        _, set_pval = stats.wilcoxon(diff)
                    except ValueError:
                        set_pval = np.nan
                elif stats is not None and test_method == 'ttest':
                    _, set_pval = stats.ttest_rel(if_arr_set[mask], core_arr_set[mask], nan_policy='omit')
                else:
                    var = np.nanvar(diff)
                    if var <= 1e-9:
                        set_pval = np.nan
                    else:
                        z = np.nanmean(diff) / np.sqrt(var / len(diff))
                        set_pval = float(math.erfc(abs(z) / math.sqrt(2.0)))
                gs_rows.append({
                    'gene_set': gs_name,
                    'direction': cfg.get('direction'),
                    'log2fc': set_log2fc,
                    'mean_interface': set_if_mean,
                    'mean_core': set_core_mean,
                    'p_value': float(set_pval) if set_pval is not None else np.nan,
                    'n_samples': int(mask.sum()),
                })
            if gs_rows:
                gene_set_path = out_dir / f"P{i}_P{j}_gene_sets.csv"
                pd.DataFrame(gs_rows).to_csv(gene_set_path, index=False)
        reports.append(
            {
                'pair': [int(i), int(j)],
                'n_samples': len(per_if),
                'full_csv': str(full_path),
                'sig_csv': str(sig_path) if sig_path else None,
                'significant_genes': int(sig_mask.sum()),
                'gene_set_csv': str(gene_set_path) if gene_set_path else None,
            }
        )
        if debug:
            logging.info("[IF-DE][debug] pair (%d,%d) processed with %d samples", i, j, len(per_if))
    return reports


@torch.no_grad()
def compute_survival_summary(model, loader, device, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    out_records = []
    for data in tqdm(loader, desc="Hazard inference"):
        data = data.to(device)
        logits, _ = model(data, compute_clustering_loss=False, update_centers=False)
        risks = hazards_to_risk(logits, mode="cumhaz").view(-1).detach().cpu().numpy()
        times = getattr(data, 't', torch.full((len(risks),), float('nan'), device=device)).view(-1).detach().cpu().numpy()
        events = getattr(data, 'e', torch.full((len(risks),), float('nan'), device=device)).view(-1).detach().cpu().numpy()
        names = data.patient_id if isinstance(data.patient_id, list) else [data.patient_id]
        if len(names) != len(risks):
            names = [names[0]] * len(risks)
        for name, risk, t, e in zip(names, risks, times, events):
            out_records.append({
                'sample_name': str(name),
                'risk_score': float(risk),
                'duration': float(t),
                'event': float(e),
            })
    if not out_records:
        logging.warning("Survival inference produced no records; skipping summary CSV.")
        return pd.DataFrame()
    df = pd.DataFrame(out_records)
    out_path = out_dir / "survival_summary.csv"
    df.to_csv(out_path, index=False)
    try:
        c_index_tensor = compute_c_index(
            torch.tensor(df['risk_score'].values, dtype=torch.float32),
            torch.tensor(df['duration'].values, dtype=torch.float32),
            torch.tensor(df['event'].values, dtype=torch.float32),
        )
        logging.info("[SURV] C-index=%.4f", float(c_index_tensor))
    except Exception as exc:
        logging.warning("C-index computation failed: %s", exc)
    metrics = compute_auc_metrics(df['event'].values.astype(float), df['risk_score'].values.astype(float))
    logging.info("[SURV] ROC-AUC=%s PR-AUC=%s", metrics.get('roc_auc'), metrics.get('pr_auc'))
    return df


def _summarize_cohort_counts(cell_df: pd.DataFrame) -> Dict[str, int]:
    if cell_df.empty or 'cohort' not in cell_df.columns:
        return {}
    group_key = 'sample_id' if 'sample_id' in cell_df.columns else 'sample_name'
    if group_key not in cell_df.columns:
        return {}
    dedup = cell_df.dropna(subset=[group_key]).drop_duplicates(subset=['cohort', group_key])
    if dedup.empty:
        return {}
    counts = dedup.groupby('cohort')[group_key].nunique().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MHGL-ST Stage3 OS archetype analysis")
    parser.add_argument("--cohort-config", type=Path, required=True,
                        help="JSON file describing OS cohorts (paths, labels, etc.)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Override trained MHGL-ST OS checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to store analysis outputs")
    parser.add_argument("--cohort-allowlist", type=str, default=None,
                        help="Comma-separated cohort names to analyze")
    parser.add_argument("--cell-feature-template", type=Path, default=None,
                        help="Representative CSV for reconstructing cell feature names")
    parser.add_argument("--gene-id-path", type=Path, default=None,
                        help="Gene ID list corresponding to gene_expr_* columns")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device override (e.g. cuda:0, cpu)")
    parser.add_argument("--tau-cell", type=float, default=0.30,
                        help="Temperature for cell prototype softmax")
    parser.add_argument("--tau-gene", type=float, default=0.40,
                        help="Temperature for gene prototype softmax")
    parser.add_argument("--fast-skip-umap", action='store_true', dest="skip_umap",
                        help="Skip UMAP/visualization to save time")
    parser.add_argument("--umap-max-cells", type=int, default=250_000,
                        help="Max cells for UMAP reducer (<=0 means all)")
    parser.add_argument("--umap-max-genes", type=int, default=150_000,
                        help="Max gene entries for UMAP reducer (<=0 means all)")
    parser.add_argument("--umap-nn-cell", type=int, default=20,
                        help="n_neighbors for cell UMAP")
    parser.add_argument("--umap-nn-gene", type=int, default=15,
                        help="n_neighbors for gene UMAP")
    parser.add_argument("--umap-min-dist-cell", type=float, default=0.05,
                        help="min_dist for cell UMAP")
    parser.add_argument("--umap-min-dist-gene", type=float, default=0.10,
                        help="min_dist for gene UMAP")
    parser.add_argument("--umap-random-state", type=int, default=42,
                        help="Random seed for UMAP sampling")
    parser.add_argument("--core-thr", type=float, default=0.4,
                        help="Core threshold for spatial metrics")
    parser.add_argument("--excl-thr", type=float, default=0.12,
                        help="Exclusivity margin for spatial metrics")
    parser.add_argument("--interface-thr", type=float, default=0.4,
                        help="Interface threshold for spatial metrics")
    parser.add_argument("--spatial-pairs", type=str, default="2-0,2-4",
                        help="Prototype pairs (e.g. 2-0,2-4) for interface metrics")
    parser.add_argument("--spatial-radii", type=str, default="100,150,200",
                        help="Comma-separated radii (microns) for occupancy metrics")
    parser.add_argument("--spatial-k", type=int, default=15,
                        help="k for spatial kNN interface density")
    parser.add_argument("--focus-proto", type=int, default=2,
                        help="Prototype used as anchor for occupancy metrics")
    parser.add_argument("--microns-per-pixel", type=float, default=1.0,
                        help="Microns per pixel for occupancy radii conversion")
    parser.add_argument("--ecology-k", type=int, default=15,
                        help="k for local ecology prototype/type proportions")
    parser.add_argument("--interface-pair-thr", type=float, default=0.05,
                        help="Mean occupancy threshold to keep prototype pair for DE")
    parser.add_argument("--de-interface-thr", type=float, default=0.45,
                        help="Threshold on IF_Pi_Pj to define interface cells")
    parser.add_argument("--de-core-thr", type=float, default=0.8,
                        help="Threshold on Pi/Pj to define core cells")
    parser.add_argument("--de-min-cells", type=int, default=60,
                        help="Minimum cells per group per sample for DE")
    parser.add_argument("--de-log2fc-thr", type=float, default=0.0,
                        help="Absolute log2 fold-change cutoff for reporting DE genes")
    parser.add_argument("--de-p-thr", type=float, default=0.05,
                        help="P-value cutoff for reporting DE genes")
    parser.add_argument("--de-agg-method", type=str, default='median', choices=['median', 'mean'],
                        help="Pseudo-bulk aggregation method per sample")
    parser.add_argument("--de-winsor-quantile", type=float, default=0.0,
                        help="Winsor clipping quantile for pseudo-bulk aggregation")
    parser.add_argument("--de-test-method", type=str, default='wilcoxon', choices=['wilcoxon', 'ttest'],
                        help="Paired statistical test for interface vs core pseudo-bulk")
    parser.add_argument("--de-debug", action='store_true', help="Verbose logging for interface DE")
    parser.add_argument("--de-gene-set-json", type=Path, default=None,
                        help="Optional JSON describing gene sets for interface DE summary")
    parser.add_argument("--enable-wsi-overlays", action='store_true', help="Generate WSI overlays when SVS/GeoJSON assets exist")
    parser.add_argument("--wsi-max-samples", type=int, default=8, help="Max samples to visualize for WSI overlays (0=all)")
    parser.add_argument("--wsi-anchor-proto", type=int, default=2, help="Prototype ID used for core-edge overlays")
    parser.add_argument("--wsi-core-thr", type=float, default=0.7, help="Threshold on Pk occupancy for WSI core highlighting")
    parser.add_argument("--wsi-gate-column", type=str, default='gate_strength_norm', help="Cell dataframe column used for gate overlays")
    parser.add_argument("--wsi-interface-pairs", type=str, default=None,
                        help="Comma-separated interface pairs (e.g. 2-4,0-2) for WSI overlays")
    parser.add_argument("--spatial-max-cells", type=int, default=0,
                        help="Optional subsampling cap for cells when caching or running batched spatial metrics")
    parser.add_argument("--spatial-batch-size", type=int, default=0,
                        help="Number of samples per spatial metrics batch (0 = all samples at once)")
    parser.add_argument("--spatial-cache-fullres", action='store_true',
                        help="Write a reusable full-resolution cell cache after prototype assignment")
    parser.add_argument("--spatial-use-cached-cells", action='store_true',
                        help="Load cell data from cache for downstream spatial analysis instead of in-memory df")
    parser.add_argument("--spatial-cache-dir", type=Path, default=None,
                        help="Directory for full-resolution cell cache (default: 05_ecology_stats/fullres_cache)")
    parser.add_argument("--spatial-cache-fname", type=str, default="full_cell_df.parquet",
                        help="Filename for cached cell dataframe")
    parser.add_argument("--spatial-cache-per-sample", action='store_true',
                        help="Write an additional per-sample parquet alongside the main cache")
    parser.add_argument("--spatial-rng-seed", type=int, default=1337,
                        help="RNG seed for spatial subsampling/caching")
    parser.add_argument("--skip-spatial-analysis", action='store_true',
                        help="Skip spatial metrics assembly entirely")
    parser.add_argument("--hub-sim-thr", type=float, default=0.75, help="Cosine similarity threshold for grouping prototypes into hubs")
    parser.add_argument("--hub-min-size", type=int, default=2, help="Minimum prototypes per hub before splitting")
    parser.add_argument("--hub-max-components", type=int, default=3, help="Number of eigen components to export for hub diffusion coords")
    parser.add_argument("--ecology-anchor-pids", type=str, default=None,
                        help="Comma-separated prototype IDs for anchor ecology focus summaries")
    parser.add_argument("--ecology-anchor-thr", type=float, default=0.7,
                        help="Threshold on anchor Pk to include cells in anchor focus summaries")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cohort_specs, cfg_model_path, cfg_output_dir, cfg_allowlist = load_cohort_specs(args.cohort_config)
    model_path = Path(args.model_path or cfg_model_path or '')
    output_dir = Path(args.output_dir or cfg_output_dir or '')
    if not model_path:
        raise ValueError("Model path must be provided via CLI or config.")
    if not output_dir:
        raise ValueError("Output directory must be provided via CLI or config.")
    model_path = model_path.expanduser()
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowlist = None
    if args.cohort_allowlist:
        allowlist = {c.strip() for c in args.cohort_allowlist.split(',') if c.strip()}
    elif cfg_allowlist:
        allowlist = set(cfg_allowlist)

    out_dirs = _prepare_output_dirs(output_dir)
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cli": vars(args),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "cohort_allowlist": sorted(allowlist) if allowlist else None,
        "output_layout": {k: str(v) for k, v in out_dirs.items()},
    }
    _write_json(output_dir / "stage_manifest.json", manifest)

    spatial_pairs_cfg = _parse_pairs(args.spatial_pairs)
    spatial_radii_cfg = _parse_radii(args.spatial_radii)

    logging.info("Loading cohorts ...")
    loader, sample_index, cohort_svs, cohort_geojson, cohort_pos = load_all_cohorts(
        cohort_specs=cohort_specs,
        allowlist=allowlist,
    )
    cell_in, gene_in = _infer_input_dims(sample_index)
    logging.info("Input dims: cell=%d gene=%d", cell_in, gene_in)

    device = _resolve_device(args.device)
    logging.info("Using device: %s", device)
    model = initialize_model(
        cell_in_channels=cell_in,
        gene_in_channels=gene_in,
        hidden_channels=64,
        embedding_dim=64,
        out_channels=1,
        num_shared_clusters=5,
        gnn_type='Transformer',
        num_attention_heads=4,
        dropout_rate=0.5,
        num_intra_modal_layers=3,
        num_inter_modal_layers=1,
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    logging.info("Extracting cell/gene embeddings ...")
    cell_df_raw, gene_df_raw = extract_all_data_for_analysis(model, loader, device)
    _safe_to_parquet(cell_df_raw, out_dirs['raw'] / "cell_df_raw.parquet", "cell_df_raw")
    _safe_to_parquet(gene_df_raw, out_dirs['raw'] / "gene_df_raw.parquet", "gene_df_raw")

    sample_dict = build_sample_dict(loader)
    torch.save(sample_dict, out_dirs['raw'] / "sample_cache.pt")

    cell_features: List[str] = []
    if args.cell_feature_template:
        try:
            cell_features = get_cell_feature_names(Path(args.cell_feature_template).expanduser())
        except Exception as exc:
            logging.warning("get_cell_feature_names failed: %s", exc)
    if not cell_features:
        nf = len([c for c in cell_df_raw.columns if c.startswith('raw_feat_')])
        cell_features = [f"raw_feat_{i}" for i in range(nf)]
    _write_json(out_dirs['raw'] / "cell_feature_names.json", {"cell_features": cell_features})

    gene_list: List[str] = []
    if args.gene_id_path:
        try:
            gene_tokens = pd.read_csv(Path(args.gene_id_path).expanduser(), header=None)[0].tolist()
            gene_list = clean_gene_list(gene_tokens)
        except Exception as exc:
            logging.warning("load gene list failed: %s", exc)
    if not gene_list:
        gene_cols = [c for c in gene_df_raw.columns if c.startswith('gene_expr_')]
        gene_list = [f"G{i}" for i in range(len(gene_cols))]
    _write_json(out_dirs['raw'] / "gene_list.json", {"genes": gene_list})

    gene_sets_cfg = load_gene_sets(args.de_gene_set_json)

    logging.info("Running survival inference ...")
    surv_df = compute_survival_summary(model, loader, device, out_dirs['survival'])
    clinical_outputs = run_clinical_survival_checks(surv_df, out_dirs['clinical'])

    centers_tensor = _get_any_centers_shared(model, device)
    if centers_tensor is None:
        logging.warning("Could not locate shared centers; prototype-dependent analysis will be limited.")
    else:
        torch.save(centers_tensor.detach().cpu(), out_dirs['processed'] / "prototype_centers.pt")

    cell_df = cell_df_raw.copy()
    gene_df = gene_df_raw.copy()
    if centers_tensor is not None:
        cell_df = assign_prototypes_with_conf_dual(cell_df, centers_tensor, tau=args.tau_cell, store_soft=False)
        gene_df = assign_prototypes_with_conf_dual(gene_df, centers_tensor, tau=args.tau_gene, store_soft=False)
    else:
        logging.info("Skipping prototype assignment since centers are unavailable.")

    did_cell_umap = False
    did_gene_umap = False
    if not args.skip_umap:
        cell_df, did_cell_umap = _ensure_umap_coordinates(
            cell_df,
            entity='cell',
            max_points=args.umap_max_cells,
            n_neighbors=args.umap_nn_cell,
            min_dist=args.umap_min_dist_cell,
            random_state=args.umap_random_state,
        )
        gene_df, did_gene_umap = _ensure_umap_coordinates(
            gene_df,
            entity='gene',
            max_points=args.umap_max_genes,
            n_neighbors=args.umap_nn_gene,
            min_dist=args.umap_min_dist_gene,
            random_state=args.umap_random_state,
        )
        if did_cell_umap or did_gene_umap:
            try:
                plot_umap_advanced_panels(cell_df, gene_df, out_dir=str(out_dirs['umap']))
            except Exception as exc:
                logging.warning("UMAP plotting failed: %s", exc)
    else:
        logging.info("UMAP computation skipped via --fast-skip-umap flag.")

    # ===== Phase 2: Ecology + Interface DE =====
    cell_df = _ensure_celltype_column(cell_df, cell_features)
    ecology_outputs: Dict[str, str] = {}
    gene_de_reports: List[Dict[str, object]] = []

    try:
        neigh_proto_prop, neigh_type_prop = compute_local_ecology_kNN(cell_df, k=args.ecology_k)
    except Exception as exc:
        logging.warning("Local ecology failed: %s", exc)
        neigh_proto_prop = pd.DataFrame(index=cell_df.index)
        neigh_type_prop = pd.DataFrame(index=cell_df.index)

    if not neigh_proto_prop.empty or not neigh_type_prop.empty:
        cell_df = pd.concat([cell_df, neigh_proto_prop, neigh_type_prop], axis=1)
        if not neigh_proto_prop.empty:
            proto_path = out_dirs['ecology'] / "cell_knn_proto_props.parquet"
            _safe_to_parquet(neigh_proto_prop, proto_path, "cell_knn_proto_props")
            ecology_outputs['cell_knn_proto_props'] = str(proto_path)
        if not neigh_type_prop.empty:
            type_path = out_dirs['ecology'] / "cell_knn_type_props.parquet"
            _safe_to_parquet(neigh_type_prop, type_path, "cell_knn_type_props")
            ecology_outputs['cell_knn_type_props'] = str(type_path)
        anchor_proto_mix, anchor_type_mix = aggregate_ecology_by_anchor(
            cell_df,
            neigh_proto_prop,
            neigh_type_prop,
            min_cells=max(50, args.de_min_cells),
        )
        if not anchor_proto_mix.empty:
            anchor_proto_path = out_dirs['ecology'] / "anchor_proto_mix.csv"
            anchor_proto_mix.to_csv(anchor_proto_path)
            ecology_outputs['anchor_proto_mix'] = str(anchor_proto_path)
        if not anchor_type_mix.empty:
            anchor_type_path = out_dirs['ecology'] / "anchor_celltype_mix.csv"
            anchor_type_mix.to_csv(anchor_type_path)
            ecology_outputs['anchor_celltype_mix'] = str(anchor_type_path)

    cell_df, interface_pairs = add_interface_strength_columns(
        cell_df,
        thr_exist=args.interface_pair_thr,
    )
    if interface_pairs:
        summary_df = summarize_interface_pairs(cell_df, interface_pairs, args.de_interface_thr)
        if not summary_df.empty:
            summary_path = out_dirs['ecology'] / "interface_pair_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            ecology_outputs['interface_pair_summary'] = str(summary_path)
        gene_de_reports = run_interface_gene_de(
            cell_df=cell_df,
            gene_df=gene_df,
            gene_list=gene_list,
            sample_dict=sample_dict,
            pairs=interface_pairs,
            out_dir=out_dirs['de'],
            t_if=args.de_interface_thr,
            t_core=args.de_core_thr,
            min_cells=max(10, args.de_min_cells),
            agg_method=args.de_agg_method,
            winsor_q=max(0.0, args.de_winsor_quantile),
            log2fc_thr=args.de_log2fc_thr,
            p_thr=args.de_p_thr,
            test_method=args.de_test_method,
            gene_sets=gene_sets_cfg,
            debug=args.de_debug,
        )
    else:
        logging.info("Interface DE skipped – no prototype pairs satisfied --interface-pair-thr.")

    processed_cell_path = out_dirs['processed'] / "cell_df_processed.parquet"
    processed_gene_path = out_dirs['processed'] / "gene_df_processed.parquet"
    _safe_to_parquet(cell_df, processed_cell_path, "cell_df_processed")
    _safe_to_parquet(gene_df, processed_gene_path, "gene_df_processed")

    cached_cells_df = None
    fullres_cache_path = None
    per_sample_manifest = None
    if args.spatial_cache_fullres or args.spatial_use_cached_cells:
        cached_cells_df, fullres_cache_path, per_sample_manifest = prepare_full_resolution_cells(
            cell_df,
            args,
            out_dirs,
        )
    spatial_source_df = cached_cells_df if cached_cells_df is not None else cell_df
    metrics_path = compute_spatial_metrics_batched(
        spatial_source_df,
        args,
        spatial_pairs_cfg,
        spatial_radii_cfg,
        out_dirs,
    )
    hub_outputs = run_hub_analysis(centers_tensor, args, out_dirs, cell_df)
    anchor_focus_outputs = export_anchor_focus_stats(
        cell_df,
        _parse_int_list(getattr(args, 'ecology_anchor_pids', None)),
        float(getattr(args, 'ecology_anchor_thr', 0.7)),
        out_dirs['ecology'] / "anchor_focus",
    )
    wsi_outputs = run_wsi_overlays(
        cell_df,
        args,
        out_dirs,
        cohort_svs,
        cohort_geojson,
        allowlist,
    )

    summary = {
        "n_samples": len(sample_dict),
        "cell_rows": int(len(cell_df)),
        "gene_rows": int(len(gene_df)),
        "cell_columns": len(cell_df.columns),
        "gene_columns": len(gene_df.columns),
        "umap": {"cell": bool(did_cell_umap), "gene": bool(did_gene_umap)},
        "prototype_centers": int(centers_tensor.shape[0]) if centers_tensor is not None else None,
        "spatial_metrics_csv": str(metrics_path) if metrics_path else None,
        "survival_csv": str(out_dirs['survival'] / "survival_summary.csv") if not surv_df.empty else None,
        "cohort_sample_counts": _summarize_cohort_counts(cell_df),
        "raw_tables": {
            "cell": str(out_dirs['raw'] / "cell_df_raw.parquet"),
            "gene": str(out_dirs['raw'] / "gene_df_raw.parquet"),
        },
        "processed_tables": {
            "cell": str(processed_cell_path),
            "gene": str(processed_gene_path),
        },
        "ecology_tables": ecology_outputs,
        "interface_pairs": [list(p) for p in interface_pairs],
        "gene_de_reports": gene_de_reports,
        "fullres_cache": {
            "cache_path": fullres_cache_path,
            "per_sample_manifest": per_sample_manifest,
        },
        "wsi_outputs": wsi_outputs,
        "hub_outputs": hub_outputs,
        "anchor_focus": anchor_focus_outputs,
        "clinical_outputs": clinical_outputs,
    }
    _write_json(output_dir / "stage_summary.json", summary)
    logging.info("Stage3 OS archetype analysis completed. Outputs at %s", output_dir)


if __name__ == "__main__":
    sys.exit(main())

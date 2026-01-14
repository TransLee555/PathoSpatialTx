# analyze_archetypes.py (V10.10-fix1)  Functional-Spatial Integration Edition (final)
# 
#  - NEW: assign_prototypes_with_conf_dual(centers_tensor=...)
#  - CHANGE:  model.shared_cluster_centers 
#  - CHANGE: gene -bulk(mean of gene.x) +  pCR 
#  - /DataLoader  torch_geometric  DataLoader

import os, gc, re, types, json, warnings, shutil, argparse, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random, sys, platform, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib
from collections import defaultdict

try:
    _ = os.environ["DISPLAY"]
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from statannotations.Annotator import Annotator
import umap

# 
try:
    from sklearn.neighbors import KDTree
except Exception:
    KDTree = None
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None
try:
    import openslide
    from openslide import OpenSlide
except Exception:
    OpenSlide = None
try:
    from shapely.geometry import shape, Polygon, MultiPolygon
    _HAS_SHAPELY = True
except Exception:
    shape = Polygon = MultiPolygon = None
    _HAS_SHAPELY = False
try:
    from skimage.draw import polygon as SKIMAGE_POLYGON
except Exception:
    SKIMAGE_POLYGON = None

# ---- Shared MHGL-ST backbone/datasets ----
from MHGL_ST_model import HierarchicalMultiModalGNN

from archetype_analysis.cohort_config import (
    CohortSpec,
    LabelSpec,
    canon_id,
    load_cohort_specs,
    load_labels_for_cohort,
)
from archetype_analysis.data_loader import (
    build_sample_dict as build_sample_dict_mod,
    extract_all_data_for_analysis as extract_all_data_for_analysis_mod,
    get_sample as get_sample_mod,
    initialize_model,
    load_all_cohorts as load_all_cohorts_mod,
)
from archetype_analysis.prototype_analysis import (
    apply_merge_to_cell_df as apply_merge_to_cell_df_mod,
    assign_prototypes_with_conf_dual as assign_prototypes_with_conf_dual_mod,
    build_merge_map_multi_signal as build_merge_map_multi_signal_mod,
)
from archetype_analysis.spatial_metrics import (
    assemble_spatial_metrics as assemble_spatial_metrics_mod,
)
from archetype_analysis.visualization import (
    overlay_cell_prototypes_per_sample as overlay_cell_prototypes_per_sample_mod,
    plot_cell_prototypes_umap as plot_cell_prototypes_umap_mod,
    plot_umap_advanced_panels as plot_umap_advanced_panels_mod,
)
from archetype_analysis.wsi_overlay import (
    collect_cell_alignment_scores as collect_cell_alignment_scores_mod,
    overlay_mask_heatmap_on_wsi as overlay_mask_heatmap_on_wsi_mod,
    overlay_wsi_core_edge_for_sample as overlay_wsi_core_edge_for_sample_mod,
    overlay_wsi_gate_for_sample as overlay_wsi_gate_for_sample_mod,
    overlay_wsi_prototypes_for_sample as overlay_wsi_prototypes_for_sample_mod,
    report_wsi_asset_status as report_wsi_asset_status_mod,
    run_alignment_wsi_overlay_for_batch as run_alignment_wsi_overlay_for_batch_mod,
)

from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset


# =========================
# Reproducibility helpers
# =========================
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def dump_manifest(outdir: str, model_path: str):
    m = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "NA"),
        "numpy": np.__version__,
        "umap": getattr(umap, "__version__", "NA"),
        "seaborn": getattr(sns, "__version__", "NA"),
        "script_sha1": hashlib.sha1(open(__file__, "rb").read()).hexdigest(),
        "model_path": model_path,
    }
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)

# =========================================================================
# 1) CONFIG
# =========================================================================
# Prefer GPU:1 on HEST, but fall back gracefully if unavailable
if torch.cuda.is_available():
    preferred_idx = 0
    device_idx = preferred_idx if torch.cuda.device_count() > preferred_idx else 0
    DEVICE = torch.device(f'cuda:{device_idx}')
else:
    DEVICE = torch.device('cpu')

_device_override = os.environ.get("ANALYZE_DEVICE")
if _device_override:
    try:
        DEVICE = torch.device(_device_override)
    except (ValueError, RuntimeError):
        print(f"[WARN] Invalid ANALYZE_V5_DEVICE override '{_device_override}'; using {DEVICE}.")

MODEL_PATH: Optional[str] = os.environ.get("ANALYZE_MODEL_PATH")
ANALYSIS_OUTPUT_DIR = os.environ.get("ANALYZE_OUTPUT_DIR", "./archetype_analysis_results")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

SCRIPT_SHA1 = hashlib.sha1(open(__file__, "rb").read()).hexdigest()
CACHE_ENABLED = True
CACHE_DIR = Path(ANALYSIS_OUTPUT_DIR) / "cache"


def _ensure_cache_dir():
    if CACHE_ENABLED:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_meta_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_meta.json"


def _cache_write_meta(name: str, meta: dict):
    if not CACHE_ENABLED:
        return
    _ensure_cache_dir()
    base = {
        "script_sha1": SCRIPT_SHA1,
        "model_path": MODEL_PATH,
        "allowlist": sorted(list(COHORT_ALLOWLIST)) if COHORT_ALLOWLIST else None,
    }
    base.update(meta or {})
    with open(_cache_meta_path(name), "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)


def _cache_read_meta(name: str):
    if not CACHE_ENABLED:
        return None
    path = _cache_meta_path(name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_meta_matches(name: str, required: dict):
    meta = _cache_read_meta(name)
    if not meta:
        return None
    for k, v in (required or {}).items():
        if meta.get(k) != v:
            return None
    return meta


def _tensor_hash(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().float().cpu().contiguous().numpy()
    else:
        arr = np.asarray(tensor)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _json_hash(obj):
    try:
        payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        payload = repr(obj).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

# 
CELL_IN_CHANNELS, GENE_IN_CHANNELS = 74, 215
HIDDEN_CHANNELS, EMBEDDING_DIM, OUT_CHANNELS = 64, 32, 1
NUM_SHARED_CLUSTERS, GNN_TYPE, NUM_ATTENTION_HEADS = 5, 'Transformer', 4
DROPOUT_RATE, NUM_INTRA_MODAL_LAYERS, NUM_INTER_MODAL_LAYERS = 0.5, 3, 1
MAX_CELLS_FOR_UMAP = 150000

# Cohort metadata placeholders (overwritten by CLI config if provided)
COHORT_ALLOWLIST: Optional[set] = None
COHORT_SPECS: List[CohortSpec] = []
COHORT_SVS: Dict[str, str] = {}
COHORT_CELLGJSON: Dict[str, str] = {}
COHORT_POS_LEVEL: Dict[str, int] = {}

print(f"--- MHGL-ST Stage 3 Archetype Analysis ---")
print(f"Device: {DEVICE}")
print(f"Outputs: {ANALYSIS_OUTPUT_DIR}")

# =========================================================================
# 2) COHORT CONFIGURATION
# =========================================================================

# =========================================================================
# 3) HELPERS
# =========================================================================


def clean_gene_list(gene_list):
    cleaned = []
    for g in gene_list:
        m = re.match(r'^[A-Z0-9\.\-]+', str(g))
        cleaned.append(m.group(0) if m else str(g))
    if len(cleaned) != len(set(cleaned)):
        print(f"[WARN] Gene symbols duplicated after cleaning ({len(cleaned)} -> {len(set(cleaned))}).")
    return cleaned


def _normalize_cohort_key(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    return re.sub(r"\s+", "", s)

def get_cell_feature_names(representative_csv_path: str):
    print("\nReconstructing cell feature names from a representative sample...")
    cols_to_drop = {
        "Identifier.CentoidX_Global", "Identifier.CentoidY_Global",
        "Global_Nuclei_ID", "type",
        "Shape.HuMoments2", "Shape.HuMoments3", "Shape.HuMoments4",
        "Shape.HuMoments5", "Shape.HuMoments6", "Shape.HuMoments7",
        "Shape.WeightedHuMoments2", "Shape.WeightedHuMoments3",
        "Shape.WeightedHuMoments4", "Shape.WeightedHuMoments6",
    }
    one_hot_cats = ['Connective', 'Neoplastic', 'Dead', 'Epithelial', 'Inflammatory']
    try:
        all_cols = pd.read_csv(representative_csv_path, nrows=0).columns.tolist()
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find representative CSV: {representative_csv_path}")
    base_feats = [c for c in all_cols if c not in cols_to_drop]
    one_hot_feats = [f"Type_{cat}" for cat in one_hot_cats]
    feats = base_feats + one_hot_feats
    print(f"  -> {len(feats)} features collected (including {len(one_hot_feats)} Type_*).")
    return feats

# ----  forward gate----
def analysis_forward(self, data, soften_gate: bool = False, soften_temp: float = None):
    device = next(self.parameters()).device; h_intra = {}
    cell_in_size_tuple = self.intra_modal_convs[0].convs[('cell', 'c_c', 'cell')].in_channels
    cell_in = cell_in_size_tuple[0] if isinstance(cell_in_size_tuple, tuple) else cell_in_size_tuple
    gene_in_size_tuple = self.intra_modal_convs[0].convs[('gene', 'g_g', 'gene')].in_channels
    gene_in = gene_in_size_tuple[0] if isinstance(gene_in_size_tuple, tuple) else gene_in_size_tuple
    h_intra['cell'] = getattr(data['cell'], 'x', torch.empty((0, cell_in), device=device))
    h_intra['gene'] = getattr(data['gene'], 'x', torch.empty((0, gene_in), device=device))
    for i_layer, conv_layer in enumerate(self.intra_modal_convs):
        attrs = {et: data[et].edge_attr for et in data.edge_types if hasattr(data[et], 'edge_attr') and et in conv_layer.convs}
        h_out = conv_layer(h_intra, data.edge_index_dict, edge_attr_dict=attrs if attrs else None)
        norms = self.intra_modal_norms[i_layer]
        for ntype, h in h_out.items():
            if h.numel() > 0: h_intra[ntype] = F.relu(norms[ntype](h))
    h_inter = h_intra
    for i_layer, conv_layer in enumerate(self.inter_modal_convs):
        attrs = {et: data[et].edge_attr for et in data.edge_types if hasattr(data[et], 'edge_attr') and et in conv_layer.convs}
        h_update = conv_layer(h_inter, data.edge_index_dict, edge_attr_dict=attrs if attrs else None)
        norms = self.inter_modal_norms[i_layer]
        for ntype, h in h_update.items():
            if h.numel() > 0: h_inter[ntype] = F.relu(norms[ntype](h_inter[ntype] + h))
    return h_inter.get('cell', None), h_inter.get('gene', None)

HierarchicalMultiModalGNN.analysis_forward_with_gate = analysis_forward

# ----  ----
# moved to archetype_analysis.prototype_analysis

# ============== Advanced 2D UMAP panels (moved to archetype_analysis.visualization) ==============

# =========================================================================
# --- 4.A  &  cell  ---
# =========================================================================
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def proto_center_similarity_cell(model, use_ema=True):
    if use_ema and hasattr(model, "centers_cell_ema"):
        C = model.centers_cell_ema.detach().cpu().numpy()
    elif hasattr(model, "centers_cell"):
        C = model.centers_cell.detach().cpu().numpy()
    else:
        raise AttributeError("Model missing centers_cell(_ema).")
    S = cosine_similarity(C)
    return S, C

# def merge_cell_prototypes(
#     model,
#     strategy: str = "threshold",     # "threshold"  "nclusters"
#     cos_thr: float = 0.9,          # 
#     n_clusters: int = None,          # 
#     use_ema: bool = True,
#     out_dir: str = "./analysis_out",
#     method: str = "complete",        # linkage "complete"/"average"/"single"/"ward"complete
#     compact_ids: bool = False        # True ID 0..M-1 False
# ):
#     """
#     : dict  {old_id -> merged_id}
#     
#       - 
#       -  old_id
#       -  compact_ids=True ID 0..M-1
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     # 1) 
#     S, C = proto_center_similarity_cell(model, use_ema=use_ema)  # S: KxK cosine, C: KxD
#     K = int(S.shape[0])

#     # 
#     plt.figure(figsize=(5,4))
#     sns.heatmap(S, vmin=-1, vmax=1, cmap="coolwarm", annot=False)
#     plt.title("Cell prototype center cosine")
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, "cell_proto_center_cosine.png"), dpi=200)
#     plt.close()

#     # 2) 
#     D = 1.0 - np.clip(S, -1, 1)  # 1 - cos in [-2, 2]   [-0, 2]cos  [-1,1]
#     tri = D[np.triu_indices_from(D, k=1)]
#     Z = linkage(tri, method=method)
#     plt.figure(figsize=(6,3))
#     dendrogram(Z)
#     plt.title(f"Cell prototype linkage ({method}, 1-cos)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, "cell_proto_dendrogram.png"), dpi=220)
#     plt.close()

#     # 3) 
#     if strategy == "nclusters" and n_clusters is not None:
#         lab = fcluster(Z, t=int(n_clusters), criterion="maxclust")
#     else:
#         # threshold:  1 - cos_thr cos_thr 
#         t = float(max(0.0, 1.0 - cos_thr))
#         lab = fcluster(Z, t=t, criterion="distance")

#     # 4) 
#     groups = defaultdict(list)
#     for old_id, lab_id in enumerate(lab):
#         groups[int(lab_id)].append(int(old_id))

#     # A  
#     if all(len(v) == 1 for v in groups.values()):
#         mapping = {old_id: old_id for old_id in range(K)}
#         merges_happened = False
#     else:
#         # B   old_id
#         mapping = {}
#         for members in groups.values():
#             new_id = int(min(members))
#             for m in members:
#                 mapping[int(m)] = new_id
#         merges_happened = True

#     # ID 0..M-1
#     if compact_ids:
#         uniq_new = {v: i for i, v in enumerate(sorted(set(mapping.values())))}
#         mapping = {k: uniq_new[v] for k, v in mapping.items()}

#     # 5) 
#     map_path = os.path.join(out_dir, "cell_proto_merge_map.txt")
#     with open(map_path, "w") as f:
#         f.write("# old_id -> merged_id\n")
#         for k in sorted(mapping):
#             f.write(f"{k}\t{mapping[k]}\n")

#     clusters_path = os.path.join(out_dir, "cell_proto_clusters.txt")
#     with open(clusters_path, "w") as f:
#         for members in sorted(groups.values(), key=lambda x: (len(x), x), reverse=True):
#             f.write(f"cluster(size={len(members)}): {sorted(members)}\n")

#     # 
#     merged_pairs = sorted({(mapping[k], k) for k in mapping if mapping[k] != k})
#     if merges_happened:
#         print(f"[MERGE] Done. merges: {merged_pairs}")
#     else:
#         print("[MERGE] No actual merges; returned identity mapping.")
#     print(f"[MERGE] mapping saved: {map_path}")
#     print(f"[MERGE] clusters saved: {clusters_path}")

#     return mapping

def apply_merge_to_cell_df(cell_df, merge_map, overwrite_prototype_id: bool = False):
    """
     cell_df  'prototype_id_merged' merge_map 
     overwrite_prototype_id=True merged  'prototype_id' 'prototype_id_raw'

    
    ----
    cell_df : pd.DataFrame
         'prototype_id' 
    merge_map : dict[int,int]
        old_id -> new_id 
    overwrite_prototype_id : bool
        True  'prototype_id' 'prototype_id_raw' 

    
    ----
    pd.DataFrame
         'prototype_id_merged' 'prototype_id'
    """
    import pandas as pd
    import numpy as np

    if cell_df.empty or "prototype_id" not in cell_df.columns:
        return cell_df

    out = cell_df.copy()

    #  prototype_id  NA
    pid_series = pd.to_numeric(out["prototype_id"], errors="coerce").astype("Int64")

    #  Series
    m = pd.Series(merge_map, name="merged", dtype="Int64")

    #  mapmap NaN ID
    merged = pid_series.map(m)
    need_fill = merged.isna() & pid_series.notna()
    merged.loc[need_fill] = pid_series.loc[need_fill]

    out["prototype_id_merged"] = merged.astype("Int64")

    if overwrite_prototype_id:
        # 
        if "prototype_id_raw" not in out.columns:
            out["prototype_id_raw"] = out["prototype_id"]
        out["prototype_id"] = out["prototype_id_merged"]

    return out


# ======================= 4.E WSI overlays moved to archetype_analysis.wsi_overlay =======================

# =========================================================================
# --- 4.B visualization moved to archetype_analysis.visualization ---
# =========================================================================

# =========================================================================
# 4.C c-g 
# =========================================================================
# =========================================================================
# 4.C c-g 
# =========================================================================
def _cg_weighted_cell_gene_matrix(sdf, gdf, gene_cols, data_sample):
    """
    : X_hat (cells_in_subset  G), valid_mask( subset ), used_edges
     cells_in_subset  sdf  node_idx.max()+1
    """
    try:
        ei = data_sample['cell','c_g','gene'].edge_index.detach().cpu().numpy()
    except Exception:
        return None, None, 0
    rows_all = ei[0].astype(int)
    cols_all = ei[1].astype(int)

    # ===  node_idx   ===
    if 'node_idx' in sdf.columns:
        cell_ids = pd.to_numeric(sdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        cell_id2row = {cid:i for i, cid in enumerate(cell_ids)}
        Nc = len(cell_ids)
    else:
        #  node_idx 
        Nc = len(sdf)
        cell_id2row = {i:i for i in range(Nc)}

    # ===  node_idx  ===
    if 'node_idx' in gdf.columns:
        gene_ids = pd.to_numeric(gdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        gene_id2col = {gid:i for i, gid in enumerate(gene_ids)}
        Ng = len(gene_ids)
    else:
        Ng = len(gdf)
        gene_id2col = {i:i for i in range(Ng)}

    # 
    try:
        eattr = getattr(data_sample['cell','c_g','gene'],'edge_attr',None)
        w_all = eattr.detach().cpu().numpy().reshape(-1) if (eattr is not None and eattr.numel()>0) \
                else np.ones(len(rows_all), dtype=np.float32)
    except Exception:
        w_all = np.ones(len(rows_all), dtype=np.float32)

    #   cell  gene
    keep = [(r in cell_id2row) and (c in gene_id2col) for r, c in zip(rows_all, cols_all)]
    if not np.any(keep):
        return None, None, 0
    rows = rows_all[keep]
    cols = cols_all[keep]
    w    = w_all[keep]

    # gene  gene 
    G = len(gene_cols)
    Xg_full = gdf[gene_cols].to_numpy(np.float32, copy=True)    #  ~ (len(gdf), G)
    #  gdf  0..Ng-1 / gene_id2col  node_idx  
    #  gene_id  gene_id2col[c]  gdf 

    X_hat = np.zeros((Nc, G), dtype=np.float32)
    ws    = np.zeros((Nc,), dtype=np.float32)

    # 
    for r, c, ww in zip(rows, cols, w):
        rr = cell_id2row.get(int(r), None)
        cc = gene_id2col.get(int(c), None)
        if rr is None or cc is None:
            continue
        X_hat[rr] += ww * Xg_full[cc]
        ws[rr]    += ww

    valid = ws > 0
    X_hat[valid] = X_hat[valid] / ws[valid][:, None]
    return X_hat, valid, len(rows)


def _cg_map_with_strength(sdf, gdf, gene_cols, data_sample,
                          weighting="blend", lambda_blend=0.5,
                          gate_mode="softmax", gate_temp=2.0,
                          winsor_q=(0.05, 0.95)):
    X_hat, valid, used = _cg_weighted_cell_gene_matrix(sdf, gdf, gene_cols, data_sample)
    if X_hat is None or used==0:
        return None, None, None, None

    ei = data_sample['cell','c_g','gene'].edge_index.cpu().numpy()
    rows = ei[0]
    try:
        eattr = getattr(data_sample['cell','c_g','gene'],'edge_attr',None)
        w = eattr.detach().cpu().numpy().reshape(-1) if (eattr is not None and eattr.numel()>0) else np.ones(len(rows), dtype=np.float32)
    except Exception:
        w = np.ones(len(rows), dtype=np.float32)

    n_cell = int(sdf['node_idx'].max())+1 if 'node_idx' in sdf.columns else sdf.shape[0]
    row_sum = np.bincount(rows, weights=w, minlength=n_cell).astype(np.float32)

    if (row_sum>0).any():
        low, high = np.quantile(row_sum[row_sum>0], winsor_q)
    else:
        low, high = 0.0, 1.0
    rs = np.clip(row_sum, low, high)
    rs = (rs - rs.min())/(rs.max()-rs.min() + 1e-9)
    w_cell = rs.astype(np.float32)
    return X_hat, (valid.astype(bool) if hasattr(valid,'dtype') else valid), row_sum, w_cell

# =========================================================================
# 4.C c-g 
# =========================================================================
def _cg_weighted_cell_gene_matrix(sdf, gdf, gene_cols, data_sample):
    """
    : X_hat (cells_in_subset  G), valid_mask( subset ), used_edges
     cells_in_subset  sdf  node_idx.max()+1
    """
    try:
        ei = data_sample['cell','c_g','gene'].edge_index.detach().cpu().numpy()
    except Exception:
        return None, None, 0
    rows_all = ei[0].astype(int)
    cols_all = ei[1].astype(int)

    # ===  node_idx   ===
    if 'node_idx' in sdf.columns:
        cell_ids = pd.to_numeric(sdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        cell_id2row = {cid:i for i, cid in enumerate(cell_ids)}
        Nc = len(cell_ids)
    else:
        #  node_idx 
        Nc = len(sdf)
        cell_id2row = {i:i for i in range(Nc)}

    # ===  node_idx  ===
    if 'node_idx' in gdf.columns:
        gene_ids = pd.to_numeric(gdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        gene_id2col = {gid:i for i, gid in enumerate(gene_ids)}
        Ng = len(gene_ids)
    else:
        Ng = len(gdf)
        gene_id2col = {i:i for i in range(Ng)}

    # 
    try:
        eattr = getattr(data_sample['cell','c_g','gene'],'edge_attr',None)
        w_all = eattr.detach().cpu().numpy().reshape(-1) if (eattr is not None and eattr.numel()>0) \
                else np.ones(len(rows_all), dtype=np.float32)
    except Exception:
        w_all = np.ones(len(rows_all), dtype=np.float32)

    #   cell  gene
    keep = [(r in cell_id2row) and (c in gene_id2col) for r, c in zip(rows_all, cols_all)]
    if not np.any(keep):
        return None, None, 0
    rows = rows_all[keep]
    cols = cols_all[keep]
    w    = w_all[keep]

    # gene  gene 
    G = len(gene_cols)
    Xg_full = gdf[gene_cols].to_numpy(np.float32, copy=True)    #  ~ (len(gdf), G)
    #  gdf  0..Ng-1 / gene_id2col  node_idx  
    #  gene_id  gene_id2col[c]  gdf 

    X_hat = np.zeros((Nc, G), dtype=np.float32)
    ws    = np.zeros((Nc,), dtype=np.float32)

    # 
    for r, c, ww in zip(rows, cols, w):
        rr = cell_id2row.get(int(r), None)
        cc = gene_id2col.get(int(c), None)
        if rr is None or cc is None:
            continue
        X_hat[rr] += ww * Xg_full[cc]
        ws[rr]    += ww

    valid = ws > 0
    X_hat[valid] = X_hat[valid] / ws[valid][:, None]
    return X_hat, valid, len(rows)


def _cg_map_with_strength(sdf, gdf, gene_cols, data_sample,
                          weighting="blend", lambda_blend=0.5,
                          gate_mode="softmax", gate_temp=2.0,
                          winsor_q=(0.05, 0.95)):
    X_hat, valid, used = _cg_weighted_cell_gene_matrix(sdf, gdf, gene_cols, data_sample)
    if X_hat is None or used==0:
        return None, None, None, None

    ei = data_sample['cell','c_g','gene'].edge_index.cpu().numpy()
    rows = ei[0]
    try:
        eattr = getattr(data_sample['cell','c_g','gene'],'edge_attr',None)
        w = eattr.detach().cpu().numpy().reshape(-1) if (eattr is not None and eattr.numel()>0) else np.ones(len(rows), dtype=np.float32)
    except Exception:
        w = np.ones(len(rows), dtype=np.float32)

    n_cell = int(sdf['node_idx'].max())+1 if 'node_idx' in sdf.columns else sdf.shape[0]
    row_sum = np.bincount(rows, weights=w, minlength=n_cell).astype(np.float32)

    if (row_sum>0).any():
        low, high = np.quantile(row_sum[row_sum>0], winsor_q)
    else:
        low, high = 0.0, 1.0
    rs = np.clip(row_sum, low, high)
    rs = (rs - rs.min())/(rs.max()-rs.min() + 1e-9)
    w_cell = rs.astype(np.float32)
    return X_hat, (valid.astype(bool) if hasattr(valid,'dtype') else valid), row_sum, w_cell

# ======================= 4.E WSI overlays moved to archetype_analysis.wsi_overlay =======================

# =========================================================================
# --- 4.B UMAP / 3D  /  ---
# =========================================================================
def plot_cell_prototypes_umap(cell_df, out_dir="./analysis_out", use_merged=False, max_cells=200_000):
    os.makedirs(out_dir, exist_ok=True)
    pid_col = "prototype_id_merged" if (use_merged and "prototype_id_merged" in cell_df.columns) else "prototype_id"
    if cell_df.empty or pid_col not in cell_df.columns:
        print("[VIS] no cell prototypes; skip."); return

    if {'umap_x','umap_y'}.issubset(cell_df.columns):
        df2d = cell_df.dropna(subset=['umap_x','umap_y'])
        if not df2d.empty:
            if len(df2d) > max_cells: df2d = df2d.sample(n=max_cells, random_state=42)
            plt.figure(figsize=(10,8))
            sns.kdeplot(data=df2d, x='umap_x', y='umap_y', hue=pid_col,
                        fill=True, alpha=0.65, levels=70, thresh=0.05, common_norm=False)
            plt.title(f"Cell prototypes  2D density UMAP ({'merged' if pid_col!='prototype_id' else 'raw'})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"cell_proto_umap2d_density_{pid_col}.png"), dpi=260)
            plt.close()
    else:
        print("[VIS] Missing 2D UMAP columns on cell_df; skip density panel.")

    if {'umap_3d_x','umap_3d_y','umap_3d_z'}.issubset(cell_df.columns):
        df3d = cell_df.dropna(subset=['umap_3d_x','umap_3d_y','umap_3d_z'])
        if not df3d.empty:
            if len(df3d) > max_cells: df3d = df3d.sample(n=max_cells, random_state=42)
            fig = plt.figure(figsize=(12,11)); ax = fig.add_subplot(111, projection='3d')
            for k in sorted(pd.to_numeric(df3d[pid_col], errors='coerce').dropna().unique()):
                sub = df3d[df3d[pid_col]==k]
                ax.scatter(sub['umap_3d_x'], sub['umap_3d_y'], sub['umap_3d_z'], s=7, alpha=0.6, label=f"P{k}")
            ax.set_title(f"Cell prototypes  3D UMAP ({'merged' if pid_col!='prototype_id' else 'raw'})")
            ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
            ax.legend(markerscale=2.0)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"cell_proto_umap3d_scatter_{pid_col}.png"), dpi=280)
            plt.close()
    else:
        print("[VIS] Missing 3D UMAP columns on cell_df; skip 3D panel.")

def overlay_cell_prototypes_per_sample(cell_df, sample_dict, svs_root_map,
                                       out_dir="./analysis_out/overlays",
                                       use_merged=False, min_cells=50):
    os.makedirs(out_dir, exist_ok=True)
    pid_col = "prototype_id_merged" if (use_merged and "prototype_id_merged" in cell_df.columns) else "prototype_id"
    need = {'pos_x','pos_y', pid_col, 'sample_name'}
    if not need.issubset(cell_df.columns):
        print("[OVERLAY] missing columns; skip overlays."); return
    uniq = sorted(pd.to_numeric(cell_df[pid_col], errors='coerce').dropna().unique().tolist())
    pal = sns.color_palette("tab10", n_colors=max(10, len(uniq)))
    cmap = {int(k): pal[i % len(pal)] for i,k in enumerate(uniq)}

    for sample, sdf in cell_df.groupby('sample_name'):
        if len(sdf) < min_cells: continue
        cohort = str(sdf['cohort'].iat[0]) if 'cohort' in sdf.columns else "NA"
        plt.figure(figsize=(8,7))
        for k, sub in sdf.groupby(pid_col):
            plt.scatter(sub['pos_x'], sub['pos_y'], s=1, alpha=0.6, color=cmap.get(int(k),(0.5,0.5,0.5)), label=f"P{k}")
        plt.gca().invert_yaxis()
        plt.axis("equal"); plt.legend(markerscale=4, fontsize=8, loc="upper right")
        plt.title(f"{cohort}  {sample}  cell prototypes ({pid_col})")
        spath = os.path.join(out_dir, f"{cohort}__{sample}__{pid_col}.png")
        plt.tight_layout(); plt.savefig(spath, dpi=260); plt.close()

# =========================================================================
# 4.C c-g 
# =========================================================================
def _cg_weighted_cell_gene_matrix(sdf, gdf, gene_cols, data_sample):
    """
    : X_hat (cells_in_subset  G), valid_mask( subset ), used_edges
     cells_in_subset  sdf  node_idx.max()+1
    """
    try:
        ei = data_sample['cell','c_g','gene'].edge_index.detach().cpu().numpy()
    except Exception:
        return None, None, 0
    rows_all = ei[0].astype(int)
    cols_all = ei[1].astype(int)

    # ===  node_idx   ===
    if 'node_idx' in sdf.columns:
        cell_ids = pd.to_numeric(sdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        cell_id2row = {cid:i for i, cid in enumerate(cell_ids)}
        Nc = len(cell_ids)
    else:
        #  node_idx 
        Nc = len(sdf)
        cell_id2row = {i:i for i in range(Nc)}

    # ===  node_idx  ===
    if 'node_idx' in gdf.columns:
        gene_ids = pd.to_numeric(gdf['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        gene_id2col = {gid:i for i, gid in enumerate(gene_ids)}
        Ng = len(gene_ids)
    else:
        Ng = len(gdf)
        gene_id2col = {i:i for i in range(Ng)}

    # 
    try:
        eattr = getattr(data_sample['cell','c_g','gene'],'edge_attr',None)
        w_all = eattr.detach().cpu().numpy().reshape(-1) if (eattr is not None and eattr.numel()>0) \
                else np.ones(len(rows_all), dtype=np.float32)
    except Exception:
        w_all = np.ones(len(rows_all), dtype=np.float32)

    #   cell  gene
    keep = [(r in cell_id2row) and (c in gene_id2col) for r, c in zip(rows_all, cols_all)]
    if not np.any(keep):
        return None, None, 0
    rows = rows_all[keep]
    cols = cols_all[keep]
    w    = w_all[keep]

    # gene  gene 
    G = len(gene_cols)
    Xg_full = gdf[gene_cols].to_numpy(np.float32, copy=True)    #  ~ (len(gdf), G)
    #  gdf  0..Ng-1 / gene_id2col  node_idx  
    #  gene_id  gene_id2col[c]  gdf 

    X_hat = np.zeros((Nc, G), dtype=np.float32)
    ws    = np.zeros((Nc,), dtype=np.float32)

    # 
    for r, c, ww in zip(rows, cols, w):
        rr = cell_id2row.get(int(r), None)
        cc = gene_id2col.get(int(c), None)
        if rr is None or cc is None:
            continue
        X_hat[rr] += ww * Xg_full[cc]
        ws[rr]    += ww

    valid = ws > 0
    X_hat[valid] = X_hat[valid] / ws[valid][:, None]
    return X_hat, valid, len(rows)


def _cg_map_with_strength(sdf, gdf, gene_cols, data_sample,
                          weighting="blend", lambda_blend=0.5,
                          gate_mode="softmax", gate_temp=2.0,
                          winsor_q=(0.05, 0.95)):
    X_hat, valid, used = _cg_weighted_cell_gene_matrix(sdf, gdf, gene_cols, data_sample)
    if X_hat is None or used==0:
        return None, None, None, None

    ei = data_sample['cell','c_g','gene'].edge_index.cpu().numpy()
    rows = ei[0]
    try:
        eattr = getattr(data_sample['cell','c_g','gene'],'edge_attr',None)
        w = eattr.detach().cpu().numpy().reshape(-1) if (eattr is not None and eattr.numel()>0) else np.ones(len(rows), dtype=np.float32)
    except Exception:
        w = np.ones(len(rows), dtype=np.float32)

    n_cell = int(sdf['node_idx'].max())+1 if 'node_idx' in sdf.columns else sdf.shape[0]
    row_sum = np.bincount(rows, weights=w, minlength=n_cell).astype(np.float32)

    if (row_sum>0).any():
        low, high = np.quantile(row_sum[row_sum>0], winsor_q)
    else:
        low, high = 0.0, 1.0
    rs = np.clip(row_sum, low, high)
    rs = (rs - rs.min())/(rs.max()-rs.min() + 1e-9)
    w_cell = rs.astype(np.float32)
    return X_hat, (valid.astype(bool) if hasattr(valid,'dtype') else valid), row_sum, w_cell

# =========================================================================
# 3) LOADING
# =========================================================================

# =========================================================================
# 4) CORE ANALYSIS
# =========================================================================
def analyze_archetypes_and_umap(df_full, centers_tensor, entity_type='cell', tau=0.4):
    print(f"\n[ANALYSIS 1/5] {entity_type.title()} prototypes & UMAP...")
    if df_full.empty: return pd.DataFrame()
    df = assign_prototypes_with_conf_dual_mod(df_full, centers_tensor=centers_tensor, tau=tau, store_soft=False)

    embed_cols = [c for c in df.columns if c.startswith('embed_')]
    n_plot = min(len(df), MAX_CELLS_FOR_UMAP if entity_type=='cell' else len(df))
    samp = df.sample(n=n_plot, random_state=42) if len(df) > n_plot else df
    X = samp[embed_cols].to_numpy(np.float32)
    if entity_type == 'cell':
        reducer_2d = umap.UMAP(n_neighbors=20 if entity_type=='cell' else 15,
                       min_dist=0.05 if entity_type=='cell' else 0.1,
                       n_components=2, random_state=42, densmap=True)
        XY = reducer_2d.fit_transform(X)
        df['umap_x'], df['umap_y'] = np.nan, np.nan
        df.loc[samp.index, ['umap_x','umap_y']] = XY

    reducer_3d = umap.UMAP(n_neighbors=20 if entity_type=='cell' else 15,
                       min_dist=0.05 if entity_type=='cell' else 0.1,
                       n_components=3, random_state=42)
    XYZ = reducer_3d.fit_transform(X)
    df['umap_3d_x'], df['umap_3d_y'], df['umap_3d_z'] = np.nan, np.nan, np.nan
    df.loc[samp.index, ['umap_3d_x','umap_3d_y','umap_3d_z']] = XYZ

    print(f"  -> done prototypes & UMAP for {entity_type}.")
    return df

def plot_all_umaps(cell_df, gene_df, output_dir):
    print("\n[VISUALIZATION] UMAP plots...")
    viz_dir = os.path.join(output_dir, "1_umap_visualizations"); os.makedirs(viz_dir, exist_ok=True)

    if {'umap_x', 'umap_y'}.issubset(cell_df.columns):
        df_2d = cell_df.dropna(subset=['umap_x', 'umap_y'])
        if not df_2d.empty:
            plt.figure(figsize=(12, 10))
            sns.kdeplot(data=df_2d, x='umap_x', y='umap_y', hue='prototype_id', fill=True, alpha=0.7)
            plt.title('2D Density UMAP of Cell Prototypes'); plt.legend(title='Prototype'); plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "umap_cell_density_2D.png"), dpi=300); plt.close()
            print("  Saved 1/4: 2D Cell Density UMAP.")
        else:
            print("  [UMAP] No finite 2D coordinates; skip density plot.")
    else:
        print("  [UMAP] Missing cell 2D coordinates; skip 2D plots.")

    def _scatter3d(df, x,y,z,c_col,title,fname):
        needed = {x, y, z, c_col}
        if not needed.issubset(df.columns):
            print(f"  [UMAP] Missing columns for {title}; skip.")
            return False
        sub = df.dropna(subset=[x,y,z,c_col])
        if sub.empty:
            print(f"  [UMAP] No finite data for {title}; skip.")
            return False
        fig = plt.figure(figsize=(12, 12)); ax = fig.add_subplot(111, projection='3d')
        cats = sorted(sub[c_col].unique())
        for cat in cats:
            ss = sub[sub[c_col]==cat]
            ax.scatter(ss[x], ss[y], ss[z], s=15, alpha=0.7, linewidth=0, label=str(cat))
        ax.set_title(title); ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
        ax.legend(title=c_col, markerscale=2.5)
        ax.view_init(elev=20, azim=65); plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, fname), dpi=300); plt.close()
        return True

    if _scatter3d(cell_df, 'umap_3d_x','umap_3d_y','umap_3d_z','prototype_id',
                  '3D UMAP of Cell Prototypes','umap_cell_by_proto_3D.png'):
        print("  Saved 2/4: 3D Cell UMAP by Prototype.")
    if _scatter3d(gene_df, 'umap_3d_x','umap_3d_y','umap_3d_z','prototype_id',
                  '3D UMAP of Gene Prototypes','umap_gene_by_proto_3D.png'):
        print("  Saved 3/4: 3D Gene UMAP by Prototype.")
    cell_df2 = cell_df.copy()
    cell_df2['phenotype'] = cell_df2['label'].map({1:'Responder',0:'Non-Responder'})
    if _scatter3d(cell_df2, 'umap_3d_x','umap_3d_y','umap_3d_z','phenotype',
                  '3D UMAP of Cells by Clinical Phenotype','umap_cell_by_pheno_3D.png'):
        print("  Saved 4/4: 3D Cell UMAP by Phenotype.")

def analyze_cell_features(cell_df, feature_names, output_dir, use_merged: bool = False):
    """
    
      - Top
      - 
      - cell_prototype_composition.csv

    
    ----
    cell_df : pd.DataFrame
         'prototype_id' 'prototype_id_merged' raw_feat_i
    feature_names : list[str]
         raw_feat_i  raw_feat_i  
         feature_names  df 
    output_dir : str
         2_cell_feature_analysis 
    use_merged : bool
        True  df  'prototype_id_merged' 'prototype_id'
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print("\n[ANALYSIS 2/5] Cell prototype feature characteristics...")
    cell_dir = os.path.join(output_dir, "2_cell_feature_analysis")
    os.makedirs(cell_dir, exist_ok=True)

    # 1)  raw_feat_i  feature_names
    rename_map = {f"raw_feat_{i}": name for i, name in enumerate(feature_names)}
    df = cell_df.copy()
    df = df.rename(columns=rename_map)

    # 2) 
    pid_col = "prototype_id_merged" if (use_merged and "prototype_id_merged" in df.columns) else "prototype_id"
    if pid_col not in df.columns:
        raise KeyError(f"'{pid_col}' not found in cell_df. (use_merged={use_merged})")

    # 3)  df 
    feat_in_df = [n for n in feature_names if n in df.columns]
    if not feat_in_df:
        print("[WARN] No feature columns matched; skip feature means.")
        mean_feats = pd.DataFrame(index=sorted(df[pid_col].dropna().unique()))
    else:
        mean_feats = df.groupby(pid_col)[feat_in_df].mean()

    # 4)  Type_*  Top 
    report_path = os.path.join(cell_dir, "cell_prototype_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Cell Prototype Detailed Report\n" + "=" * 30 + "\n")

        #  type  feature_names df 
        type_cols = [c for c in df.columns if c.startswith("Type_")]
        for pid in mean_feats.index:
            s = mean_feats.loc[pid] if pid in mean_feats.index else pd.Series(dtype=float)
            f.write(f"\n\n********** Prototype {pid} **********\n")

            # 
            types = s[s.index.str.startswith("Type_")].sort_values(ascending=False) if not s.empty else pd.Series(dtype=float)
            if len(types) > 0:
                f.write("  Dominant Cell Types:\n")
                for n, v in types.items():
                    f.write(f"    - {n:<25}: {v:.3f}\n")
            else:
                f.write("  Dominant Cell Types: (no Type_* columns)\n")

            # Top
            morph = s[~s.index.str.startswith("Type_")] if not s.empty else pd.Series(dtype=float)
            if len(morph) > 0:
                f.write("\n  Top 15 Morphological Features:\n")
                for n, v in morph.nlargest(15).items():
                    f.write(f"    - {n:<45}: {v:.4f}\n")
            else:
                f.write("\n  Top 15 Morphological Features: (none)\n")

    # 5)  Type_* one-hot 
    type_cols = [c for c in df.columns if c.startswith("Type_")]
    if type_cols:
        # 
        # 0
        df["cell_type"] = df[type_cols].idxmax(axis=1).str.replace("Type_", "", regex=False)

        comp = df.groupby(pid_col)["cell_type"].value_counts(normalize=True).unstack(fill_value=0)
        comp_csv = os.path.join(cell_dir, "cell_prototype_composition.csv")
        comp.to_csv(comp_csv)

        ax = comp.plot(kind="bar", stacked=True, figsize=(14, 9), cmap="tab20", width=0.8)
        title_suffix = "merged" if pid_col == "prototype_id_merged" else "raw"
        plt.title(f"Cellular Composition per Prototype ({title_suffix})")
        plt.xlabel("Prototype")
        plt.ylabel("Proportion")
        plt.xticks(rotation=0)
        plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig_path = os.path.join(cell_dir, "cell_prototype_composition.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("[WARN] No Type_* columns found; skip composition plots.")


def center_cos_from_Wshared(W_shared):
    """
     W_shared 
    W_shared: (K, D) torch.Tensor  np.ndarray
    :
      S: (K, K) numpy
      C_norm: (K, D) numpyL2 
    """
    import numpy as np, torch
    if W_shared is None:
        raise RuntimeError("[center_cos_from_Wshared] W_shared is None.")
    C = W_shared.detach().float().cpu().numpy() if torch.is_tensor(W_shared) \
        else np.asarray(W_shared, dtype=np.float32)
    if C.ndim != 2 or C.shape[0] < 2:
        raise RuntimeError(f"[center_cos_from_Wshared] shape={C.shape} (K,D)  K>=2")
    C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    S = (C_norm @ C_norm.T).astype(np.float32)
    return S, C_norm




def analyze_phenotype_correlation(cell_df, output_dir):
    """
    Per-sample  vs  & Shannon 
    -  'prototype_id_merged' 'prototype_id'
    -  label  {0,1} 
    -     Mann-Whitney 
    -  Shannon entropy
    -  CSV & 

    props_dfindex = sample_name
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    try:
        from statannotations.Annotator import Annotator
    except Exception:
        Annotator = None  # 

    print("\n[ANALYSIS 3/5] Prototype vs phenotype...")

    pheno_dir = os.path.join(output_dir, "3_phenotype_analysis")
    os.makedirs(pheno_dir, exist_ok=True)

    #  1) 
    pid_col = "prototype_id_merged" if "prototype_id_merged" in cell_df.columns else "prototype_id"
    if pid_col not in cell_df.columns:
        print(f"  No prototype column ('{pid_col}') found."); 
        return pd.DataFrame()

    #  2) 
    df = cell_df.copy()
    #  label  0/1
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
    df = df[df['label'].isin([0, 1])]
    if df.empty:
        print("  No labeled cells."); 
        return pd.DataFrame()

    #  3)  x  
    # groupby  idunstack  =normalize=True 
    props_df = (
        df.groupby(['sample_name', 'label'])[pid_col]
          .value_counts(normalize=True)
          .unstack(fill_value=0)
          .reset_index()
    )
    props_df.columns.name = None

    #  "Proto_<id>_prop" float
    proto_cols_raw = [c for c in props_df.columns if c not in ('sample_name', 'label')]
    rename_map = {}
    for c in proto_cols_raw:
        #  id  Int64 / str
        try:
            pid = int(c)
        except Exception:
            #  "Proto_0_prop" 
            import re
            m = re.search(r"(\d+)", str(c))
            pid = int(m.group(1)) if m else str(c)
        rename_map[c] = f"Proto_{pid}_prop"
    props_df = props_df.rename(columns=rename_map)

    # 
    props_df['label_str'] = props_df['label'].map({0: 'Non-Responder', 1: 'Responder'})

    #  4) CSV
    out_comp_csv = os.path.join(pheno_dir, "prototype_proportion_per_sample.csv")
    props_df.to_csv(out_comp_csv, index=False)

    #  5)  + MWU
    plot_proto_cols = [c for c in props_df.columns if c.startswith("Proto_") and c.endswith("_prop")]
    if len(plot_proto_cols):
        melted = props_df.melt(
            id_vars=['label_str'], 
            value_vars=plot_proto_cols,
            var_name='prototype_id', value_name='proportion'
        )

        #  object/NaN
        melted['proportion'] = pd.to_numeric(melted['proportion'], errors='coerce').astype(float)
        melted = melted[np.isfinite(melted['proportion'])]

        # 
        width = max(10, melted['prototype_id'].nunique() * 1.5)
        plt.figure(figsize=(width, 7))
        ax = sns.boxplot(data=melted, x='prototype_id', y='proportion',
                         hue='label_str', palette="Set2", showfliers=False)
        plt.title('Prototype Proportions by Phenotype')
        plt.xlabel('Prototype'); plt.ylabel('Proportion')
        plt.xticks(rotation=0); plt.tight_layout()

        # 
        if Annotator is not None:
            pairs = [((proto, 'Non-Responder'), (proto, 'Responder')) 
                     for proto in sorted(melted['prototype_id'].unique())]
            try:
                annot = Annotator(ax, pairs, data=melted, x='prototype_id', y='proportion', hue='label_str')
                annot.configure(test='Mann-Whitney', text_format='star', verbose=False).apply_and_annotate()
            except Exception as e:
                print(f"  [WARN] annotate failed: {e}")
        else:
            print("  [INFO] statannotations ")

        plt.savefig(os.path.join(pheno_dir, "phenotype_correlation_boxplot.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    #  6) Shannon  float0
    if len(plot_proto_cols):
        V = props_df[plot_proto_cols].apply(pd.to_numeric, errors='coerce').astype('float64')
        V = V.clip(lower=0)

        # 
        row_sums = V.sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)   #  0  0/0
        V = V.div(row_sums, axis=0)

        #  e 2  base=2
        H = stats.entropy(V.values, axis=1)
        props_df['shannon_entropy'] = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        # 
        out_H_csv = os.path.join(pheno_dir, "prototype_entropy_per_sample.csv")
        props_df[['sample_name', 'label', 'label_str', 'shannon_entropy']].to_csv(out_H_csv, index=False)

        #  + strip
        plt.figure(figsize=(8, 7))
        ax = sns.violinplot(
            data=props_df, x='label_str', y='shannon_entropy',
            hue='label_str', legend=False, palette="muted",
            inner='quartile', cut=0
        )
        sns.stripplot(data=props_df, x='label_str', y='shannon_entropy', color=".3", size=4, jitter=True)

        if Annotator is not None:
            try:
                annot2 = Annotator(ax, [('Non-Responder', 'Responder')],
                                   data=props_df, x='label_str', y='shannon_entropy')
                annot2.configure(test='Mann-Whitney', text_format='simple', verbose=False).apply_and_annotate()
            except Exception as e:
                print(f"  [WARN] annotate failed: {e}")

        plt.title('Tumor Heterogeneity (Shannon Entropy) by Phenotype')
        plt.xlabel('Phenotype'); plt.ylabel('Shannon entropy')
        plt.tight_layout()
        plt.savefig(os.path.join(pheno_dir, "heterogeneity_vs_phenotype.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    #  7) index  sample_name merge
    return props_df.set_index('sample_name')



# =========================================================================
# 5) GENE ANALYSIS-bulk + 
# =========================================================================
def _pseudobulk_gene_mean(gene_df, gene_list):
    gcols = [f'gene_expr_{i}' for i in range(len(gene_list))]
    if not all(c in gene_df.columns for c in gcols):
        gcols = [c for c in gene_df.columns if str(c).startswith('gene_expr_')]
    if not gcols:
        return pd.DataFrame()
    bulk = gene_df.groupby(['sample_name','label'])[gcols].mean().reset_index()
    mapper = {f'gene_expr_{i}': gene_list[i] for i in range(min(len(gcols), len(gene_list)))}
    bulk = bulk.rename(columns=mapper)
    return bulk

def run_archetype_gene_analysis(gene_df, cell_df, gene_list, output_dir,
                                p_value_threshold=0.05, inter_log2fc_threshold=0.0, intra_log2fc_threshold=0.0,
                                min_cells_per_proto=50):
    print("\n[ANALYSIS 4/5] Gene programs per prototype (pseudobulk-based)...")
    if gene_df.empty or cell_df.empty:
        print("  Skipping: empty gene/cell df."); return
    gene_dir = os.path.join(output_dir, "4_gene_functional_analysis"); os.makedirs(gene_dir, exist_ok=True)
    intra_dir = os.path.join(gene_dir, "1_intra_prototype_DE_by_phenotype"); os.makedirs(intra_dir, exist_ok=True)
    inter_dir = os.path.join(gene_dir, "2_inter_prototype_DE_markers"); os.makedirs(inter_dir, exist_ok=True)

    bulk = _pseudobulk_gene_mean(gene_df, gene_list)
    if bulk.empty:
        print("  [WARN] no gene_expr_* columns for pseudobulk."); return

    cnt = cell_df.groupby(['sample_name','prototype_id']).size().unstack(fill_value=0)
    # scipy.stats.mode 
    def _mode_label(s):
        arr = s.dropna().to_numpy()
        if arr.size == 0: return -1
        m = stats.mode(arr, keepdims=True)
        return int(m.mode[0]) if hasattr(m, "mode") else int(m[0])
    sample_label = cell_df.groupby('sample_name')['label'].agg(_mode_label)

    # A) Intra-Prototype  Phenotype  (with BH-FDR)
    for pid in sorted(cnt.columns.tolist()):
        elig_samples = cnt.index[cnt[pid] >= min_cells_per_proto].tolist()
        if not elig_samples: continue
        sub_bulk = bulk[bulk['sample_name'].isin(elig_samples)].copy()
        if sub_bulk.empty: continue
        sub_bulk['label'] = sub_bulk['sample_name'].map(sample_label)
        sub_bulk = sub_bulk[sub_bulk['label'].isin([0,1])]
        if sub_bulk['label'].nunique() < 2: continue

        gcols = [c for c in sub_bulk.columns if c not in ('sample_name','label')]
        recs=[]; R = sub_bulk[sub_bulk['label']==1]; N = sub_bulk[sub_bulk['label']==0]
        for g in gcols:
            a, b = R[g].values, N[g].values
            if len(a) < 2 or len(b) < 2: continue
            t,p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
            if np.isnan(p): continue
            resp_mean = float(np.nanmean(a)) if np.isfinite(np.nanmean(a)) else np.nan
            non_mean = float(np.nanmean(b)) if np.isfinite(np.nanmean(b)) else np.nan
            log2fc = np.log2((resp_mean+1e-9)/(non_mean+1e-9)) if np.isfinite(resp_mean) and np.isfinite(non_mean) else np.nan
            if p < p_value_threshold and abs(log2fc) > intra_log2fc_threshold:
                recs.append({
                    'gene': g,
                    'log2fc_Resp_vs_NonResp': log2fc,
                    'p_value': p,
                    't_stat': t,
                    'resp_mean': resp_mean,
                    'nonresp_mean': non_mean,
                    'resp_sample_count': int(np.isfinite(a).sum()),
                    'nonresp_sample_count': int(np.isfinite(b).sum()),
                })
        if recs:
            df = pd.DataFrame(recs).sort_values('p_value')
            try:
                from statsmodels.stats.multitest import multipletests
                df['q_value'] = multipletests(df['p_value'].values, method='fdr_bh')[1]
            except Exception:
                df['q_value'] = np.nan
            df = df.sort_values(['q_value','p_value'])
            df.to_csv(os.path.join(intra_dir, f"P{pid}_DEGs.csv"), index=False)

    # B) Inter-Prototype markers one-vs-rest, with BH-FDR
    props = cnt.div(cnt.sum(axis=1), axis=0).fillna(0)
    dom = props.idxmax(axis=1).rename('dom_proto')
    bulk2 = bulk.merge(dom.reset_index(), on='sample_name', how='left')
    for pid in sorted(cnt.columns.tolist()):
        df1 = bulk2[bulk2['dom_proto']==pid]
        df2 = bulk2[bulk2['dom_proto']!=pid]
        if len(df1) < 3 or len(df2) < 3: continue
        gcols = [c for c in bulk2.columns if c not in ('sample_name','label','dom_proto')]
        recs=[]
        for g in gcols:
            a, b = df1[g].values, df2[g].values
            if len(a)<2 or len(b)<2: continue
            t,p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
            if np.isnan(p) or p >= p_value_threshold: continue
            log2fc = np.log2((a.mean()+1e-9)/(b.mean()+1e-9))
            if abs(log2fc) > inter_log2fc_threshold:
                recs.append({'gene':g,'log2fc':log2fc,'p_value':p})
        if recs:
            df = pd.DataFrame(recs)
            try:
                from statsmodels.stats.multitest import multipletests
                df['q_value'] = multipletests(df['p_value'].values, method='fdr_bh')[1]
            except Exception:
                df['q_value'] = np.nan
            df = df.sort_values(['q_value','log2fc'], ascending=[True, False])
            df.to_csv(os.path.join(inter_dir, f"P{pid}_marker_genes.csv"), index=False)
    print("  Gene analysis complete.")


# =========================================================================
# 6) Spatial interactions
# =========================================================================
def run_archetype_interface_analysis(cell_df: pd.DataFrame, output_dir: str, file_prefix: str):
    """
    -log2 enrichment
     cell_df ['sample_name','pos_x','pos_y','prototype_id']
     'prototype_id_merged' 
    <file_prefix>_interface_log2EN.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) 
    pid_col = 'prototype_id_merged' if 'prototype_id_merged' in cell_df.columns else 'prototype_id'
    df = cell_df.dropna(subset=['sample_name', 'pos_x', 'pos_y', pid_col]).copy()
    df['pid'] = pd.to_numeric(df[pid_col], errors='coerce').astype('Int64')
    df = df.dropna(subset=['pid'])
    if df.empty:
        raise RuntimeError("No valid cells for interface analysis.")
    df['pid'] = df['pid'].astype(int)

    #  prototype  0..K-1
    uniq = sorted(df['pid'].unique().tolist())
    pid_map = {p:i for i,p in enumerate(uniq)}
    df['pid_std'] = df['pid'].map(pid_map).astype(int)
    K = len(uniq)

    # 2) 
    #    /
    #     kNN 
    try:
        from sklearn.neighbors import KDTree
    except Exception as e:
        raise RuntimeError("scikit-learn  KDTree") from e

    observed = np.zeros((K, K), dtype=np.float64)
    expected = np.zeros((K, K), dtype=np.float64)

    for s, sub in df.groupby('sample_name'):
        sub = sub.dropna(subset=['pos_x','pos_y'])
        if len(sub) < 4:
            continue
        coords = sub[['pos_x','pos_y']].to_numpy(dtype=np.float64, copy=False)
        pids   = sub['pid_std'].to_numpy(dtype=np.int64, copy=False)

        # kNN
        k = min(8, len(sub)-1)
        tree = KDTree(coords)
        neigh = tree.query(coords, k=k+1, return_distance=False)[:,1:]

        #  p->q
        for i, row in enumerate(neigh):
            s_id = pids[i]
            for j in row:
                t_id = pids[j]
                observed[s_id, t_id] += 1.0

        # 
        cnt = np.bincount(pids, minlength=K).astype(np.float64)
        if cnt.sum() > 0:
            freq = cnt / cnt.sum()
            exp_mat = np.outer(freq, freq)
            exp_mat *= neigh.shape[1] * len(sub)  #  k 
            expected += exp_mat

    # 3)  +  DataFrame
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)

    #  0
    num  = obs + 1e-9
    denom= exp + 1e-9

    #  numpy  DF pandas  object 
    log2en_mat = np.log2(num/denom)

    idx = [f"P{pid_map_inv}" for pid_map_inv in range(K)]
    #  0..K-1
    inv = {v:k for k,v in pid_map.items()}
    idx = [f"P{inv[i]}" for i in range(K)]
    log2en = pd.DataFrame(log2en_mat, index=idx, columns=idx)

    out_csv = os.path.join(output_dir, f"{file_prefix}_interface_log2EN.csv")
    log2en.to_csv(out_csv)
    print(f"  -> saved interface log2EN to: {out_csv}")
    return log2en


def analyze_phenotype_specific_interactions(cell_df, output_dir):
    print("\n[ANALYSIS 5/5] Phenotype-specific interactions...")
    if 'label' not in cell_df.columns:
        print("  No label col."); return
    R = cell_df[cell_df['label']==1]
    N = cell_df[cell_df['label']==0]
    if R.empty or N.empty:
        print("  One phenotype missing."); return
    eR = run_archetype_interface_analysis(R, output_dir, "responders")
    eN = run_archetype_interface_analysis(N, output_dir, "non_responders")
    if eR is None or eN is None: return
    allp = sorted(list(set(eR.index)|set(eR.columns)|set(eN.index)|set(eN.columns)))
    eR = eR.reindex(index=allp, columns=allp, fill_value=0)
    eN = eN.reindex(index=allp, columns=allp, fill_value=0)
    diff = eR - eN
    plt.figure(figsize=(12,10))
    vmax = np.abs(diff.values).max() or 1
    sns.heatmap(diff, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax)
    plt.title('Differential Interaction Enrichment (Responder - Non-Responder)')
    plt.xlabel('Neighbor'); plt.ylabel('Center')
    os.makedirs(os.path.join(output_dir, "5_spatial_interaction_analysis"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "5_spatial_interaction_analysis", "interface_enrichment_DIFFERENTIAL.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

# =============== Gate interpretability ===============
def log_gate_stats(sample_dict, outdir):
    """Dump histogram of cross-modal gate weights for interpretability."""
    vals=[]
    for s, g in sample_dict.items():
        try:
            ea = getattr(g['cell','c_g','gene'],'edge_attr',None)
            if ea is not None and ea.numel()>0:
                v = ea.detach().cpu().numpy().reshape(-1)
                vals.extend(v.tolist())
        except Exception:
            pass
    if vals:
        plt.figure(figsize=(5,4))
        plt.hist(vals, bins=50, density=True)
        plt.title('Cross-modal gate weights')
        plt.tight_layout()
        Path(outdir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(outdir, "gate_weights_hist.png"), dpi=220)
        plt.close()

# =============== Prototype stability (bootstrap reference) ===============
from sklearn.metrics import adjusted_rand_score
def bootstrap_proto_stability(cell_df_ref, centers_tensor, n_boot=50, frac=0.8, tau=0.4):
    """Return ARI values between subsample assignments and reference assignments."""
    if cell_df_ref.empty:
        return np.array([], np.float32)
    hard_full = assign_prototypes_with_conf_dual_mod(cell_df_ref, centers_tensor, tau, False)['prototype_id'].to_numpy()
    ids = cell_df_ref.index.to_numpy()
    ARI = []
    rng = np.random.default_rng(42)
    for _ in range(int(n_boot)):
        idx = rng.choice(ids, size=max(1, int(frac * len(ids))), replace=True)
        sub = cell_df_ref.loc[idx]
        hard_sub = assign_prototypes_with_conf_dual_mod(sub, centers_tensor, tau, False)['prototype_id'].to_numpy()
        ref = hard_full[cell_df_ref.index.get_indexer(idx)]
        try:
            ARI.append(adjusted_rand_score(ref, hard_sub))
        except Exception:
            continue
    return np.array(ARI, np.float32)

# ===========================================
# CELL-ANCHORED ECOLOGY (kNN/radius based)
# ===========================================


def _ensure_celltype_column(cell_df, feature_names):
    type_cols = [f for f in feature_names if f.startswith("Type_") and f in cell_df.columns]
    if type_cols:
        #  Type_* 
        cell_df = cell_df.copy()
        cell_df['cell_type'] = cell_df[type_cols].idxmax(axis=1).str.replace('Type_','', regex=False)
    else:
        if 'cell_type' not in cell_df.columns:
            cell_df = cell_df.copy()
            cell_df['cell_type'] = 'NA'
    return cell_df

def compute_local_ecology_kNN(cell_df, k=15, include_unknown=True):
    need = {'pos_x','pos_y','prototype_id'}
    if not need.issubset(cell_df.columns):
        return (pd.DataFrame(index=cell_df.index),
                pd.DataFrame(index=cell_df.index))

    df = cell_df.dropna(subset=['pos_x','pos_y','prototype_id']).copy()
    if len(df) < k + 1:
        return (pd.DataFrame(index=cell_df.index),
                pd.DataFrame(index=cell_df.index))

    # 1) 
    if include_unknown:
        df['cell_type'] = df['cell_type'].fillna('Unknown')
    else:
        df = df[~df['cell_type'].isna()].copy()

    coords = df[['pos_x','pos_y']].to_numpy(np.float32)
    pids   = pd.to_numeric(df['prototype_id'], errors='coerce').astype(int).to_numpy()
    types  = df['cell_type'].astype(str).to_numpy()

    tree = KDTree(coords)
    idx = tree.query(coords, k=min(k+1, len(df)), return_distance=False)[:,1:]

    # 
    uniq_p = np.unique(pids)
    p_map = {p:i for i,p in enumerate(uniq_p)}
    P = np.zeros((len(df), len(uniq_p)), dtype=np.float32)
    for i, neigh in enumerate(idx):
        np.add.at(P[i], [p_map[p] for p in pids[neigh]], 1)
    P /= np.maximum(1, P.sum(axis=1, keepdims=True))
    neigh_proto_prop = pd.DataFrame(P, index=df.index, columns=[f'P{int(p)}' for p in uniq_p])

    # 
    uniq_t = np.unique(types)
    t_map = {t:i for i,t in enumerate(uniq_t)}
    T = np.zeros((len(df), len(uniq_t)), dtype=np.float32)
    for i, neigh in enumerate(idx):
        np.add.at(T[i], [t_map[t] for t in types[neigh]], 1)
    T /= np.maximum(1, T.sum(axis=1, keepdims=True))
    neigh_type_prop = pd.DataFrame(T, index=df.index, columns=[f'CT:{t}' for t in uniq_t])

    #  df  index
    neigh_proto_prop = neigh_proto_prop.reindex(index=cell_df.index).fillna(0.0)
    neigh_type_prop  = neigh_type_prop.reindex(index=cell_df.index).fillna(0.0)
    return neigh_proto_prop, neigh_type_prop


def aggregate_ecology_by_anchor(cell_df, neigh_proto_prop, neigh_type_prop, min_cells=200):
    """
    
      - anchor_proto_mix[anchor, P*]
      - anchor_type_mix [anchor, CT:*]
    """
    out_proto, out_type = {}, {}
    for pid, sdf in cell_df.groupby('prototype_id'):
        if len(sdf) < min_cells: 
            continue
        idx = sdf.index
        out_proto[int(pid)] = neigh_proto_prop.loc[idx].mean(axis=0)
        out_type[int(pid)]  = neigh_type_prop.loc[idx].mean(axis=0)
    if not out_proto:
        return pd.DataFrame(), pd.DataFrame()
    anchor_proto_mix = pd.DataFrame(out_proto).T.sort_index()
    anchor_type_mix  = pd.DataFrame(out_type).T.sort_index()
    anchor_proto_mix.index.name = 'anchor_pid'
    anchor_type_mix.index.name  = 'anchor_pid'
    return anchor_proto_mix, anchor_type_mix

def prototype_radial_profile(cell_df, anchor_pid, radii=(20,40,80,160)):
    """
     r 
    /DataFrame=r=
    """
    sdf = cell_df.dropna(subset=['pos_x','pos_y']).copy()
    sdf = sdf.sort_index()
    coords = sdf[['pos_x','pos_y']].to_numpy(np.float32)
    pids   = pd.to_numeric(sdf['prototype_id'], errors='coerce').astype(int).to_numpy()
    ctypes = sdf['cell_type'].astype(str).to_numpy()
    mask_anchor = (pids == int(anchor_pid))
    if mask_anchor.sum() == 0 or len(sdf) < 2:
        return pd.DataFrame(), pd.DataFrame()

    tree = KDTree(coords)
    # k radius k
    k_big = min(256, len(sdf)-1)
    dists, knn = tree.query(coords[mask_anchor], k=k_big, return_distance=True)
    # 
    dists, knn = dists[:,1:], knn[:,1:]

    uniq_p = sorted(np.unique(pids).tolist())
    uniq_t = sorted(np.unique(ctypes).tolist())
    P_acc = np.zeros((len(radii), len(uniq_p)), dtype=np.float64)
    T_acc = np.zeros((len(radii), len(uniq_t)), dtype=np.float64)
    cnt_anchor = mask_anchor.sum()

    for i_r, R in enumerate(radii):
        for i in range(cnt_anchor):
            # <=R 
            m = dists[i] <= R
            if not np.any(m): 
                continue
            neigh_idx = knn[i, m]
            P_here = pids[neigh_idx]
            T_here = ctypes[neigh_idx]
            # 
            for p in P_here:
                P_acc[i_r, uniq_p.index(int(p))] += 1
            for t in T_here:
                T_acc[i_r, uniq_t.index(str(t))] += 1

    # 
    P_prop = P_acc / np.maximum(1, P_acc.sum(axis=1, keepdims=True))
    T_prop = T_acc / np.maximum(1, T_acc.sum(axis=1, keepdims=True))
    dfP = pd.DataFrame(P_prop, index=[f"r{r}" for r in radii], columns=[f'P{p}' for p in uniq_p])
    dfT = pd.DataFrame(T_prop, index=[f"r{r}" for r in radii], columns=[f'CT:{t}' for t in uniq_t])
    return dfP, dfT

def knn_transition_matrix(cell_df, k=8):
    """
     kNN ==
    /
    """
    if not {'pos_x','pos_y','prototype_id'}.issubset(cell_df.columns):
        return pd.DataFrame()
    df = cell_df.dropna(subset=['pos_x','pos_y']).copy()
    coords = df[['pos_x','pos_y']].to_numpy(np.float32)
    pids = pd.to_numeric(df['prototype_id'], errors='coerce').astype(int).to_numpy()
    uniq = sorted(np.unique(pids).tolist())
    idx_map = {p:i for i,p in enumerate(uniq)}
    tree = KDTree(coords)
    neigh = tree.query(coords, k=min(k+1,len(df)), return_distance=False)[:,1:]
    M = np.zeros((len(uniq), len(uniq)), dtype=np.float64)
    for i, row in enumerate(neigh):
        s = idx_map[pids[i]]
        for j in row:
            t = idx_map[pids[j]]
            M[s,t] += 1
    M = M / np.maximum(1, M.sum(axis=1, keepdims=True))
    return pd.DataFrame(M, index=uniq, columns=uniq)

def per_sample_ecology_signature(cell_df, neigh_type_prop, anchor_proto_mix, min_cells=100):
    """
    
      -   
    sample_name, anchor_pid, <CT:*>...
    """
    rows = []
    for (sname, pid), sdf in cell_df.groupby(['sample_name','prototype_id']):
        if len(sdf) < min_cells: 
            continue
        idx = sdf.index
        v = neigh_type_prop.loc[idx].mean(axis=0)
        rec = {'sample_name': sname, 'anchor_pid': int(pid)}
        rec.update({k: float(v[k]) for k in v.index})
        rows.append(rec)
    sig = pd.DataFrame(rows)
    #  anchor_proto_mix 
    return sig


def plot_anchor_ecology(anchor_proto_mix, anchor_type_mix, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not anchor_proto_mix.empty:
        plt.figure(figsize=(8, 0.6*len(anchor_proto_mix.index)+3))
        sns.heatmap(anchor_proto_mix, cmap='viridis')
        plt.title('Anchor Prototype  Neighbor Prototype Mix')
        plt.xlabel('Neighbor Prototype'); plt.ylabel('Anchor Prototype')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "anchor_proto_mix.png"), dpi=240)
        plt.close()
    if not anchor_type_mix.empty:
        plt.figure(figsize=(10, 0.6*len(anchor_type_mix.index)+3))
        sns.heatmap(anchor_type_mix, cmap='mako')
        plt.title('Anchor Prototype  Neighbor Cell-Type Mix')
        plt.xlabel('Neighbor Cell-Type'); plt.ylabel('Anchor Prototype')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "anchor_type_mix.png"), dpi=240)
        plt.close()

def plot_radial_profile(dfP, dfT, anchor_pid, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not dfP.empty:
        plt.figure(figsize=(10,5))
        for c in dfP.columns:
            plt.plot(range(len(dfP.index)), dfP[c].values, marker='o', label=c)
        plt.xticks(range(len(dfP.index)), dfP.index, rotation=0)
        plt.ylabel('Proportion'); plt.title(f'Radial profile (prototypes) @ anchor P{anchor_pid}')
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"radial_prototypes_P{anchor_pid}.png"), dpi=260); plt.close()
    if not dfT.empty:
        plt.figure(figsize=(10,5))
        for c in dfT.columns:
            plt.plot(range(len(dfT.index)), dfT[c].values, marker='o', label=c)
        plt.xticks(range(len(dfT.index)), dfT.index, rotation=0)
        plt.ylabel('Proportion'); plt.title(f'Radial profile (cell-types) @ anchor P{anchor_pid}')
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"radial_celltypes_P{anchor_pid}.png"), dpi=260); plt.close()

def plot_transition_matrix(T, out_path):
    if T.empty: return
    plt.figure(figsize=(8,6))
    sns.heatmap(T, cmap='coolwarm', center=T.values.mean())
    plt.title('kNN-based Prototype Transition (row-normalized)')
    plt.xlabel('Neighbor '); plt.ylabel('Anchor (center)') 
    plt.tight_layout(); plt.savefig(out_path, dpi=260); plt.close()




# =========================================================================
# 7) MAIN glue
# =========================================================================
def load_data_and_init_model():
    """ (model, loader)"""
    CELL_IN_CHANNELS, GENE_IN_CHANNELS = 0, 0
    if CELL_IN_CHANNELS == 0: CELL_IN_CHANNELS = 74 # Fallback if no cell data in first train sample
    if GENE_IN_CHANNELS == 0: GENE_IN_CHANNELS = 215 # Fallback
    print(f"Cell In Channels: {CELL_IN_CHANNELS}, Gene In Channels: {GENE_IN_CHANNELS}")

    if CELL_IN_CHANNELS == 0 or GENE_IN_CHANNELS == 0:
        print("ERROR: Zero input channels for model. Check data. Exiting."); exit()

    HIDDEN_CHANNELS, EMBEDDING_DIM, OUT_CHANNELS = 64, 32, 1
    num_shared_clusters = 5
    GNN_TYPE, NUM_ATTENTION_HEADS, DROPOUT_RATE = 'Transformer', 4, 0.5
    NUM_INTRA_MODAL_LAYERS, NUM_INTER_MODAL_LAYERS = 3, 1

    model = HierarchicalMultiModalGNN(CELL_IN_CHANNELS, GENE_IN_CHANNELS, HIDDEN_CHANNELS, EMBEDDING_DIM, OUT_CHANNELS, num_shared_clusters, GNN_TYPE, NUM_ATTENTION_HEADS, DROPOUT_RATE, NUM_INTRA_MODAL_LAYERS, NUM_INTER_MODAL_LAYERS)
    loader, sample_index = load_all_cohorts(memory_lite=True)
    return model, loader


import numpy as np
import pandas as pd
from collections import Counter

def _proto_resp_prop(cell_df):
    #  0.5
    tab = cell_df[cell_df['label'].isin([0,1])].groupby(['prototype_id','label']).size().unstack(fill_value=0)
    prop = (tab.get(1,0) / (tab.sum(axis=1)+1e-9)).reindex(sorted(cell_df['prototype_id'].unique()), fill_value=np.nan)
    return prop.fillna(0.5)

def _proto_knn_transition(cell_df, k=8):
    # s  freq[s,t]
    try:
        from sklearn.neighbors import KDTree
    except:
        return None
    need = {'pos_x','pos_y','prototype_id'}
    if not need.issubset(cell_df.columns) or len(cell_df)<k+1:
        return None
    df = cell_df.dropna(subset=['pos_x','pos_y']).copy()
    coords = df[['pos_x','pos_y']].values
    pids = df['prototype_id'].astype(int).values
    tree = KDTree(coords)
    neigh = tree.query(coords, k=min(k+1,len(df)), return_distance=False)[:,1:]
    S = sorted(np.unique(pids))
    idx = {p:i for i,p in enumerate(S)}
    M = np.zeros((len(S), len(S)), dtype=np.float64)
    for i,row in enumerate(neigh):
        s = idx[pids[i]]
        for j in row:
            t = idx[pids[j]]
            M[s,t] += 1
    # 
    M = M / (M.sum(axis=1, keepdims=True)+1e-9)
    return pd.DataFrame(M, index=S, columns=S)


def _coerce_proto_index(mat, K=None):
    """ DataFrame  index/columns  'P3''3'  3 0"""
    if not isinstance(mat, pd.DataFrame):
        return None
    X = mat.copy()
    def _to_int_idx(idx):
        out = []
        for v in idx:
            s = str(v)
            if s.startswith('P') or s.startswith('p'):
                s = s[1:]
            try:
                out.append(int(s))
            except Exception:
                out.append(None)
        return out
    X.index   = _to_int_idx(X.index)
    X.columns = _to_int_idx(X.columns)
    X = X.dropna(axis=0).dropna(axis=1)
    #  0..K-1
    if K is not None:
        X = X.reindex(index=range(K), columns=range(K), fill_value=0.0)
    return X



def build_merge_map_multi_signal(
    model,
    cell_df,
    gene_bulk_df=None,
    log2EN=None,
    p_min=0.01,
    n_min=5000,
    keep_top_m=2,
    prefer_targets=None,
    w=(0.35, 0.25, 0.25, 0.10, 0.05),
    min_center_cos=0.0,
    centers_tensor=None,   # << 
):
    """
     (small  large)
      Score = w1* + w2* + w3* + w4*bulk + w5* (tie-breaker)
    center-cos >= min_center_cos 

    
    ----
    model : nn.Module   centers_*
    cell_df : pd.DataFrame
         'prototype_id'/ 'pos_x','pos_y','label','sample_name' 
    gene_bulk_df : DataFrame or None
        
    log2EN : DataFrame or None
         log2 
    p_min, n_min : float,int
        
    keep_top_m : int
         M
    prefer_targets : Iterable or None
         id
    w : tuple[5]
         (phenotype, spatial, knn, gene, center)
    min_center_cos : float
        ([0,1])
    centers_tensor : torch.Tensor[K, D] or None
         None  model.shared_cluster_centers 
         0.5

    
    ----
    mapping : dict[int,int]  old_id -> merged_id
    """
    import numpy as np
    import pandas as pd
    from collections import Counter

    # ---------- 0)  ----------
    if 'prototype_id' not in cell_df.columns:
        raise KeyError("cell_df  'prototype_id' ")

    # ---------- 1)  K  ----------
    #  id centers_tensor K
    ids_in_df = pd.to_numeric(cell_df['prototype_id'], errors='coerce').dropna().astype(int).tolist()
    cnt = Counter(ids_in_df)
    K_from_df = (max(cnt.keys()) + 1) if cnt else 0

    # centers  > model.shared_cluster_centers > 
    if centers_tensor is None and hasattr(model, 'shared_cluster_centers') and model.shared_cluster_centers is not None:
        centers_tensor = model.shared_cluster_centers

    if centers_tensor is not None:
        try:
            C_np = centers_tensor.detach().float().cpu().numpy()
        except Exception:
            C_np = None
    else:
        C_np = None

    if C_np is not None and C_np.ndim == 2 and C_np.shape[0] >= 1:
        K_from_centers = int(C_np.shape[0])
    else:
        K_from_centers = 0

    #  K
    K = K_from_centers if K_from_centers > 0 else K_from_df
    if K == 0:
        # 
        return {}

    counts = np.array([cnt.get(k, 0) for k in range(K)], dtype=np.int64)
    props = counts / (counts.sum() + 1e-9)

    # ---------- 2)  ----------
    keep = set(np.argsort(-props)[:max(1, int(keep_top_m))].tolist())
    if prefer_targets:
        keep |= set(int(x) for x in prefer_targets)
    targets = sorted([t for t in keep if 0 <= t < K])
    if not targets:
        #  0 
        targets = [0]

    # ---------- 3)  ----------
    # (1)  0.5
    def _proto_resp_prop(df, K):
        if 'label' not in df.columns:
            return pd.Series([0.5]*K, index=range(K), dtype='float64')
        tab = df[df['label'].isin([0, 1])].groupby(['prototype_id', 'label']).size().unstack(fill_value=0)
        prop = (tab.get(1, 0) / (tab.sum(axis=1) + 1e-9)).reindex(range(K), fill_value=np.nan)
        return prop.fillna(0.5)
    resp_prop = _proto_resp_prop(cell_df, K)

    # (2)  log2ENDataFrame
    A_spatial = _coerce_proto_index(log2EN, K=K) if isinstance(log2EN, pd.DataFrame) else None

    # (3) kNN 
    def _proto_knn_transition(df, K, k=8):
        need = {'pos_x', 'pos_y', 'prototype_id'}
        try:
            from sklearn.neighbors import KDTree
        except Exception:
            return None
        if (not need.issubset(df.columns)) or (len(df) < k + 1):
            return None
        d2 = df.dropna(subset=['pos_x', 'pos_y']).copy()
        if d2.empty or len(d2) < k + 1:
            return None
        coords = d2[['pos_x', 'pos_y']].to_numpy(np.float32)
        pids = pd.to_numeric(d2['prototype_id'], errors='coerce').astype(int).to_numpy()
        uniq = sorted(np.unique(pids).tolist())
        idx_map = {p: i for i, p in enumerate(uniq)}
        tree = KDTree(coords)
        neigh = tree.query(coords, k=min(k + 1, len(d2)), return_distance=False)[:, 1:]
        M = np.zeros((len(uniq), len(uniq)), dtype=np.float64)
        for i, row in enumerate(neigh):
            s = idx_map[pids[i]]
            for j in row:
                t = idx_map[pids[j]]
                M[s, t] += 1
        M = M / np.maximum(1, M.sum(axis=1, keepdims=True))
        out = pd.DataFrame(M, index=uniq, columns=uniq)
        return out.reindex(index=range(K), columns=range(K), fill_value=0.0)
    A_knn = _proto_knn_transition(cell_df, K, k=8)

    # (4) bulk
    A_gene = None

    # (5) KK, [0,1]
    if C_np is not None and C_np.shape[0] == K:
        C_np = C_np / (np.linalg.norm(C_np, axis=1, keepdims=True) + 1e-8)
        A_center = C_np @ C_np.T                 # [-1, 1]
        A_center_01 = (A_center + 1.0) / 2.0     # [0, 1]
    else:
        #  0.5 tie-breaker 
        A_center_01 = np.full((K, K), 0.5, dtype=np.float32)

    def _safe_lookup(mat, i, j, default=0.0):
        try:
            return float(mat.loc[i, j])
        except Exception:
            return default

    # ---------- 4)  ----------
    mapping = {}
    merged_pairs = []

    for s in range(K):
        # 
        if (props[s] >= float(p_min) and counts[s] >= int(n_min)) or (s in keep):
            mapping[s] = s
            continue

        # 
        scores = []
        for t in targets:
            # (a) 
            a1 = 1.0 - abs(float(resp_prop.get(s, 0.5)) - float(resp_prop.get(t, 0.5)))
            # (b)  0
            a2 = _safe_lookup(A_spatial, s, t, 0.0) if A_spatial is not None else 0.0
            a2 = max(0.0, a2)
            # (c) kNN 
            a3 = _safe_lookup(A_knn, s, t, 0.0) if A_knn is not None else 0.0
            # (d) 
            a4 = 0.0 if A_gene is None else _safe_lookup(A_gene, s, t, 0.0)
            # (e) tie-breaker
            a5 = float(A_center_01[s, t])

            score = w[0]*a1 + w[1]*a2 + w[2]*a3 + w[3]*a4 + w[4]*a5
            scores.append((t, score, a5))

        if not scores:
            mapping[s] = s
            continue

        t_star, best_score, best_center01 = max(scores, key=lambda x: x[1])

        # 
        if best_center01 >= float(min_center_cos):
            mapping[s] = int(t_star)
            merged_pairs.append((s, int(t_star), best_center01, best_score))
        else:
            mapping[s] = s

    # 
    try:
        n_merge = sum(1 for k, v in mapping.items() if k != v)
        print(f"[MERGE] multi-signal (center-cos{min_center_cos}): merged {n_merge}/{K}.")
        if n_merge > 0:
            preview = ", ".join([f"{a}->{b}(cos={c:.2f})"
                                 for a, b, c, _ in sorted(merged_pairs, key=lambda x: -x[2])[:10]])
            print(f"[MERGE] examples: {preview}")
    except Exception:
        pass

    return mapping



# =======================
# Gene modality anchored on cell prototypes
# =======================
def _per_cell_geneproto_mix_for_sample(sdf_cell, sdf_gene, data_sample,
                                       gene_proto_col='prototype_id', normalize=True):
    # 1) 
    X_hat, valid_mask, used_edges = _cg_weighted_cell_gene_matrix(
        sdf_cell, sdf_gene, [c for c in sdf_gene.columns if str(c).startswith('gene_expr_')], data_sample
    )
    if X_hat is None or used_edges == 0:
        return pd.DataFrame(index=sdf_cell.index)

    # 2)  sdf_gene
    if gene_proto_col not in sdf_gene.columns:
        return pd.DataFrame(index=sdf_cell.index)
    gpid_full = pd.to_numeric(sdf_gene[gene_proto_col], errors='coerce').astype('Int64')

    #  
    # cell 
    if 'node_idx' in sdf_cell.columns:
        cell_ids = pd.to_numeric(sdf_cell['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        cell_id2row = {cid:i for i, cid in enumerate(cell_ids)}
        Nc = len(cell_ids)
    else:
        Nc = len(sdf_cell)
        cell_id2row = {i:i for i in range(Nc)}

    # gene 
    if 'node_idx' in sdf_gene.columns:
        gene_ids = pd.to_numeric(sdf_gene['node_idx'], errors='coerce').astype('Int64').dropna().astype(int).to_numpy()
        gene_id2col = {gid:i for i, gid in enumerate(gene_ids)}
    else:
        gene_id2col = {i:i for i in range(len(sdf_gene))}

    #  =  Ng
    Ng = len(gene_id2col)
    gpid = np.full((Ng,), -1, dtype=int)
    for gid, cc in gene_id2col.items():
        val = gpid_full.iloc[cc]
        gpid[cc] = int(val) if pd.notna(val) else -1

    # 3)  cellgene 
    try:
        ei = data_sample['cell','c_g','gene'].edge_index.cpu().numpy()
        eattr = getattr(data_sample['cell','c_g','gene'],'edge_attr',None)
        w_all = eattr.detach().cpu().numpy().reshape(-1) if (eattr is not None and eattr.numel()>0) \
                else np.ones(ei.shape[1], np.float32)
    except Exception:
        return pd.DataFrame(index=sdf_cell.index)

    rows_all, cols_all = ei[0].astype(int), ei[1].astype(int)
    keep = [(r in cell_id2row) and (c in gene_id2col) for r, c in zip(rows_all, cols_all)]
    if not np.any(keep):
        return pd.DataFrame(index=sdf_cell.index)
    rows = rows_all[keep]
    cols = cols_all[keep]
    w    = w_all[keep]

    uniq_gp = sorted([int(x) for x in pd.Series(gpid).dropna().unique().tolist() if x >= 0])
    idx_gp = {p:i for i,p in enumerate(uniq_gp)}
    M = np.zeros((Nc, len(uniq_gp)), dtype=np.float32)

    for r, c, ww in zip(rows, cols, w):
        rr = cell_id2row.get(int(r), None)
        cc = gene_id2col.get(int(c), None)
        if rr is None or cc is None:
            continue
        pid = gpid[cc]
        if pid >= 0 and pid in idx_gp:
            M[rr, idx_gp[int(pid)]] += float(ww)

    if normalize and M.shape[1] > 0:
        M = M / np.maximum(1e-6, M.sum(axis=1, keepdims=True))

    #  M  == len(sdf_cell)
    df = pd.DataFrame(M, index=sdf_cell.index, columns=[f'GProto{p}' for p in uniq_gp])
    return df


def compute_anchor_geneproto_mix(cell_df, gene_df, sample_dict, min_cells=200):
    """
     per-cell 
     anchor_geneproto_mix (anchor_pid  )
    """
    out_list = []
    for sname, sdf_cell in cell_df.groupby('sample_name'):
        data_sample = get_sample_mod(sample_dict, sname)
        if data_sample is None: 
            continue
        sdf_gene = gene_df[gene_df['sample_name']==sname]
        if sdf_cell.empty or sdf_gene.empty:
            continue
        df_gp = _per_cell_geneproto_mix_for_sample(sdf_cell, sdf_gene, data_sample, gene_proto_col='prototype_id')
        if df_gp.empty:
            continue
        tmp = pd.concat([sdf_cell[['prototype_id']].reset_index(drop=True), df_gp.reset_index(drop=True)], axis=1)
        tmp['sample_name'] = sname
        out_list.append(tmp)

    if not out_list:
        return pd.DataFrame()

    per_cell_gp = pd.concat(out_list, axis=0, ignore_index=True)
    rows = []
    for pid, sdf in per_cell_gp.groupby('prototype_id'):
        if len(sdf) < min_cells:
            continue
        rows.append(pd.DataFrame(sdf.drop(columns=['prototype_id','sample_name']).mean(axis=0)).T.assign(anchor_pid=int(pid)))
    if not rows:
        return pd.DataFrame()
    anchor_geneproto_mix = pd.concat(rows, axis=0).set_index('anchor_pid').sort_index()
    return anchor_geneproto_mix


def gene_core_vs_edge_DE(cell_df, gene_df, sample_dict, anchor_pid, prop_thr=0.7,
                         p_value_threshold=0.05, log2fc_thr=0.1, min_patients_each=3, out_dir="./"):
    """
     neigh_proto_prop['P{anchor}'] >= prop_thr 
     gene  c_g  vs 
    """
    #  neigh_proto_prop A_ecology cell_df
    need_col = f'P{int(anchor_pid)}'
    if need_col not in cell_df.columns:
        print(f"[ECO-DE] missing {need_col} in cell_df (neigh_proto_prop)."); return

    # 1)  per-cell gene expression ()
    rows = []
    gcols = [c for c in gene_df.columns if str(c).startswith('gene_expr_')]
    for sname, sdf_cell in cell_df.groupby('sample_name'):
        data_sample = get_sample_mod(sample_dict, sname)
        if data_sample is None: continue
        sdf_gene = gene_df[gene_df['sample_name']==sname]
        X_hat, valid, used = _cg_weighted_cell_gene_matrix(sdf_cell, sdf_gene, gcols, data_sample)
        if X_hat is None or used==0: 
            continue
        # 2) /
        core_mask = (sdf_cell[need_col].to_numpy(dtype=float) >= float(prop_thr))
        lab = np.where(core_mask, 1, 0).astype(int)  # 1=core, 0=edge
        # 3) -bulk
        if lab.sum() < 5 or (len(lab)-lab.sum()) < 5:
            continue
        core_mean = X_hat[core_mask].mean(axis=0); edge_mean = X_hat[~core_mask].mean(axis=0)
        rows.append(pd.DataFrame({
            'sample_name': sname,
            'is_core': [1, 0],
            **{g: [core_mean[i], edge_mean[i]] for i,g in enumerate(gcols)}
        }))

    if not rows: 
        print("[ECO-DE] no valid samples."); return
    pb = pd.concat(rows, ignore_index=True)

    # 4)  t  core vs edge
    recs=[]
    eps = 1e-9
    for i,g in enumerate(gcols):
        a = pb[pb['is_core']==1][g].values; b = pb[pb['is_core']==0][g].values
        if len(a) < min_patients_each or len(b) < min_patients_each: 
            continue
        t,p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        if np.isnan(p): continue
        core_mean = float(np.nanmean(a)) if np.isfinite(np.nanmean(a)) else np.nan
        edge_mean = float(np.nanmean(b)) if np.isfinite(np.nanmean(b)) else np.nan
        log2fc = np.log2((core_mean + eps)/(edge_mean + eps)) if np.isfinite(core_mean) and np.isfinite(edge_mean) else np.nan
        if p < p_value_threshold and abs(log2fc) >= float(log2fc_thr):
            recs.append({
                'gene': g,
                'log2fc_core_vs_edge': log2fc,
                'p_value': p,
                'core_mean': core_mean,
                'edge_mean': edge_mean,
                'core_sample_count': int(np.isfinite(a).sum()),
                'edge_sample_count': int(np.isfinite(b).sum()),
            })
    if recs:
        df = pd.DataFrame(recs).sort_values(['p_value', 'log2fc_core_vs_edge'], ascending=[True, False])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"DE_core_vs_edge_P{int(anchor_pid)}.csv"), index=False)


# =========  5) GENE ANALYSIS-bulk +   =========
# ssGSEA on (A)  & (B) -bulk
def _ensure_gene_matrix_from_cells(cell_df, gene_df, gene_list, sample_dict, min_cells_per_proto=50):
    """
     proto_gene_expr (DataFrame: index=prototype_id, columns=gene symbols)
     gene  cell _cg_weighted_cell_gene_matrix
          prototype_id 
    """
    from collections import defaultdict
    gene_cols = [f'gene_expr_{i}' for i in range(len(gene_list))]
    have_cols = [c for c in gene_cols if c in gene_df.columns]
    if not have_cols: 
        return pd.DataFrame()
    agg = defaultdict(list)
    for sname, sdf in cell_df.groupby('sample_name'):
        gdf = gene_df[gene_df['sample_name']==sname]
        if gdf.empty: 
            continue
        data_sample = get_sample_mod(sample_dict, sname)
        if data_sample is None:
            continue
        X_hat, valid, used, w_cell = _cg_map_with_strength(sdf, gdf, have_cols, data_sample)
        if X_hat is None:
            continue
        #  cell cells x genes
        tmp = pd.DataFrame(X_hat, columns=gene_list[:len(have_cols)], index=sdf.index)
        tmp['prototype_id'] = sdf['prototype_id'].values
        # 
        byp = tmp.groupby('prototype_id')[gene_list[:len(have_cols)]].mean()
        for pid, row in byp.iterrows():
            agg[int(pid)].append(row.values)
    if not agg:
        return pd.DataFrame()
    rows = {}
    for pid, mats in agg.items():
        M = np.vstack(mats)
        rows[pid] = M.mean(axis=0)  # 
    proto_gene_expr = pd.DataFrame(rows).T
    proto_gene_expr.columns = gene_list[:proto_gene_expr.shape[1]]
    # 
    cnt = cell_df.groupby('prototype_id').size()
    keep = cnt[cnt >= min_cells_per_proto].index
    return proto_gene_expr.loc[proto_gene_expr.index.intersection(keep)].sort_index()

def _run_ssgsea(matrix_df, gene_sets, out_path_prefix, sample_norm=True):
    import pandas as pd, numpy as np
    try:
        import gseapy as gp
    except Exception as e:
        print("[WARN] gseapy not installed; skip ssGSEA:", e)
        return pd.DataFrame()

    if matrix_df is None or matrix_df.empty:
        print("[ssGSEA] input matrix empty; skip.")
        return pd.DataFrame()

    df = matrix_df.copy()

    #  sample/label/dom_proto
    if df.shape[1] >= 2 and not pd.api.types.is_numeric_dtype(df.dtypes.iloc[0]):
        first_col = df.columns[0]
        if str(first_col).lower() not in ("sample_name", "label", "dom_proto"):
            df = df.set_index(first_col)

    #  NaN /
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.empty or df.shape[0] < 5 or df.shape[1] < 3:
        print(f"[ssGSEA] matrix too small after cleaning: {df.shape}")
        return pd.DataFrame()

    # gseapy  =gene, =sample
    idx_str = df.index.astype(str)
    looks_like_proto = idx_str.str.fullmatch(r"P?\d+").fillna(False).any()
    if looks_like_proto:
        df = df.T

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 
    try:
        enr = gp.ssgsea(
            data=df, gene_sets=str(gene_sets),
            outdir=None, sample_norm=sample_norm, scale=False, threads=1, no_plot=True,
        )
    except Exception as e:
        print("[ssGSEA] FAILED with cleaned df:", df.shape)
        print("[ssGSEA] dtypes head:", df.dtypes.head().to_dict())
        ii = list(df.index[:3]); jj = list(df.columns[:3])
        print("[ssGSEA] index sample:", ii)
        print("[ssGSEA] columns sample:", jj)
        print("[ssGSEA] small corner values:\n", df.loc[ii, jj])
        raise

    scores = enr.res2d.T
    scores.to_csv(f"{out_path_prefix}_ssgsea_scores.csv")
    return scores




def run_ssgsea_cell_anchor_and_bulk(gene_df, cell_df, gene_list, out_dir, sample_dict,
                                    min_cells_per_proto=50, gene_sets='Hallmark'):
    """
    A)   ssGSEA
    B) -bulk  ssGSEA
    """
    os.makedirs(out_dir, exist_ok=True)
    # A) 
    A = _ensure_gene_matrix_from_cells(cell_df, gene_df, gene_list, sample_dict,
                                       min_cells_per_proto=min_cells_per_proto)
    if not A.empty:
        A_scores = _run_ssgsea(A, gene_sets, os.path.join(out_dir, "A_anchor_proto"))
        # 
        if not A_scores.empty:
            plt.figure(figsize=(12, max(4, 0.25*A_scores.shape[1])))
            sns.heatmap(A_scores, cmap='RdBu_r', center=0)
            plt.title('ssGSEA (cell-anchored prototypes)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "A_anchor_proto_ssgsea_heatmap.png"), dpi=280); plt.close()
    else:
        print("[ssGSEA-A] protogene matrix empty; skipped.")

    # B) -bulk
    bulk = _pseudobulk_gene_mean(gene_df, gene_list)
    if bulk.empty:
        print("[ssGSEA-B] pseudobulk empty; skip.")
        return

    gcols = [c for c in bulk.columns if c not in ('sample_name','label')]

    cnt = cell_df.groupby(['sample_name','prototype_id']).size().unstack(fill_value=0)
    props = cnt.div(cnt.sum(axis=1), axis=0).fillna(0)
    dom = props.idxmax(axis=1).rename('dom_proto')

    bulk2 = bulk.merge(dom.reset_index(), on='sample_name', how='left')

    #  ssGSEA ==
    #  _run_ssgsea **** genes  samples
    all_expr = bulk2.set_index('sample_name')[gcols].T
    all_scores = _run_ssgsea(all_expr, gene_sets, os.path.join(out_dir, "B_bulk_ALL"))
    if all_scores.empty:
        print("[ssGSEA-B] ALL scores empty; skip subgroup stats.")
        return

    for pid in sorted(cnt.columns.tolist()):
        sub = bulk2[bulk2['dom_proto']==pid]
        rest = bulk2[bulk2['dom_proto']!=pid]
        if len(sub) < 3 or len(rest) < 3:
            continue

        # /
        mat_sub = sub.set_index('sample_name')[gcols].T   #  
        B_scores = _run_ssgsea(mat_sub, gene_sets, os.path.join(out_dir, f"B_bulk_P{pid}"))

        #  all_scores dom_proto==pid vs  
        import numpy as np
        lab = (bulk2.set_index('sample_name')['dom_proto']==pid).reindex(all_scores.index).astype(int)

        recs=[]
        for pathway in all_scores.columns:
            a = all_scores.loc[lab==1, pathway].values
            b = all_scores.loc[lab==0, pathway].values
            if len(a) >= 3 and len(b) >= 3:
                t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
                recs.append({
                    'pathway': pathway, 't': t, 'pvalue': p,
                    f'mean_in_P{pid}': np.nanmean(a),
                    'mean_outside': np.nanmean(b)
                })
        if recs:
            df = pd.DataFrame(recs).sort_values('pvalue')
            try:
                from statsmodels.stats.multitest import multipletests
                df['q_value'] = multipletests(df['pvalue'].values, method='fdr_bh')[1]
            except Exception:
                df['q_value'] = np.nan
            df = df.sort_values(['q_value','pvalue'])
            df.to_csv(os.path.join(out_dir, f"B_bulk_P{pid}_ssgsea_diff.csv"), index=False)


# ===== A1)  &  =====
def _type_enrichment_mask(cell_df, sample_name, type_col_prefix="Type_", grid=512, thr=0.55, min_cells=200):
    import numpy as np, pandas as pd
    sdf = cell_df[cell_df['sample_name']==sample_name].dropna(subset=['pos_x','pos_y']).copy()
    type_cols = [c for c in sdf.columns if c.startswith(type_col_prefix)]
    if sdf.empty or not type_cols: 
        return {}
    W = int(sdf['pos_x'].max())+1; H = int(sdf['pos_y'].max())+1
    gx = max(1, W//grid); gy = max(1, H//grid)
    sdf['gx'] = (sdf['pos_x']/grid).astype(int)
    sdf['gy'] = (sdf['pos_y']/grid).astype(int)
    out={}
    for tcol in type_cols:
        agg = sdf.groupby(['gy','gx'])[tcol].mean().unstack(fill_value=0)
        cnt = sdf.groupby(['gy','gx']).size().unstack(fill_value=0)
        M = (agg.where(cnt>=min_cells, np.nan) >= thr)
        out[tcol] = M.fillna(False).values.astype(np.uint8)  # 
    return out  # dict: {"Type_Epithelial": HgWg, ...}

# ===== A2)    shapely  alpha-shape =====
def _mask_to_polygons(mask, grid=512, alpha=None):
    import numpy as np
    ys, xs = np.where(mask>0)
    if len(xs) < 3:
        return []
    pts = np.vstack([xs*grid, ys*grid]).T.astype(np.float32)
    polys=[]
    try:
        import shapely.geometry as geom, shapely.ops as ops
        from shapely.geometry import MultiPoint
        if alpha is not None:
            # alpha-shape shapely >=2 +  alphashape 
            try:
                import alphashape
                poly = alphashape.alphashape(MultiPoint(pts), alpha)
                if poly and not poly.is_empty:
                    polys = [poly] if poly.geom_type=="Polygon" else list(poly.geoms)
                else:
                    polys=[]
            except Exception:
                hull = MultiPoint(pts).convex_hull
                polys = [hull] if hull.geom_type=="Polygon" else list(hull.geoms)
        else:
            hull = MultiPoint(pts).convex_hull
            polys = [hull] if hull.geom_type=="Polygon" else list(hull.geoms)
        #  N2 numpy 
        return [np.array(list(pg.exterior.coords), np.float32) for pg in polys]
    except Exception:
        #  shapelyscipy
        from scipy.spatial import ConvexHull
        try:
            h = ConvexHull(pts)
            poly = pts[h.vertices]
            return [poly]
        except Exception:
            return []

# ===== A3) Tumor/EpiStroma/ConnInflamm/Imm =====
def make_histologic_zones(cell_df, sample_name, grid=512, thr=0.55, buffer_px=800):
    masks = _type_enrichment_mask(cell_df, sample_name, grid=grid, thr=thr)
    zones={}
    label_map = {"Type_Epithelial":"TumorLike", "Type_Connective":"StromaLike", "Type_Inflammatory":"InflammLike"}
    for key,label in label_map.items():
        if key in masks:
            polys = _mask_to_polygons(masks[key], grid=grid, alpha=None)
            zones[label] = polys  # list of Nx2
    # = TumorLike -
    peri=[]
    try:
        import shapely.geometry as geom
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        tl = [Polygon(p) for p in zones.get("TumorLike", []) if len(p)>=3]
        if tl:
            u = unary_union(tl)
            peri_ring = u.buffer(buffer_px).difference(u.buffer(-buffer_px))
            if peri_ring.is_empty: 
                peri=[]
            else:
                if peri_ring.geom_type=="Polygon":
                    peri=[np.array(list(peri_ring.exterior.coords), np.float32)]
                else:
                    peri=[np.array(list(g.exterior.coords), np.float32) for g in peri_ring.geoms if g.geom_type=="Polygon"]
    except Exception:
        #  shapely/
        peri=[]
        for P in zones.get("TumorLike", []):
            if len(P)<3: continue
            c = P.mean(axis=0, keepdims=True)
            v = P - c
            out = c + v*(1.0 + buffer_px/max(1.0, np.linalg.norm(v,axis=1,keepdims=True).mean()))
            inn = c + v*(1.0 - buffer_px/max(1.0, np.linalg.norm(v,axis=1,keepdims=True).mean()))
            peri.append(out.astype(np.float32))  # 
    zones["PeriTumorLike"] = peri
    return zones  # dict: label -> list of polygons (Nx2)


def build_hubs_from_signals(cell_df,
                            center_cos=None,   # KxK [-1,1]
                            log2EN=None,       # KxK  sigmoid
                            knn_trans=None,    # KxK  min-max
                            resp_prop_weight=0.2,
                            n_hubs=3):
    """
    
      - proto2hub: {proto_id -> hub_id}
      - hub_centers: (n_hubs, K)  hub 
      - S: (K, K) [0,1]=1

    
      *  sklearn>=1.6 ( metric='precomputed')  (affinity='precomputed')
      *  [0,1] 
      *  (0.35, 0.30, 0.25, resp_prop_weight) 
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering

    # ----------  ----------
    def _safe_K():
        k_from_center = center_cos.shape[0] if (center_cos is not None and hasattr(center_cos, "shape") and center_cos.ndim==2 and center_cos.shape[0]==center_cos.shape[1]) else None
        if k_from_center is not None:
            return int(k_from_center)
        #  cell_df  K
        pid = pd.to_numeric(cell_df.get('prototype_id'), errors='coerce')
        pid = pid.dropna().astype(int)
        return int(pid.max()) + 1 if pid.size else 0

    def _resize01(X, K, fill=0.5):
        """/NoneKxK np.float32None ->  fill"""
        if X is None:
            return np.full((K, K), float(fill), np.float32)
        X = np.asarray(X, np.float32)
        if X.shape != (K, K):
            Y = np.full((K, K), float(fill), np.float32)
            m = min(K, X.shape[0], X.shape[1])
            Y[:m, :m] = X[:m, :m]
            X = Y
        return X

    def _sym01_clip(X):
        """ + clip  [0,1] +  1"""
        X = 0.5*(X + X.T)
        X = np.clip(X, 0.0, 1.0)
        np.fill_diagonal(X, 1.0)
        return X

    def _norm_minmax(X):
        X = X.astype(np.float32, copy=False)
        lo, hi = float(np.nanmin(X)), float(np.nanmax(X))
        if not np.isfinite(hi-lo) or hi - lo < 1e-12:
            return np.zeros_like(X, np.float32)
        Y = (X - lo) / (hi - lo + 1e-12)
        return Y.astype(np.float32, copy=False)

    def _agglom_precomputed(n_clusters:int):
        """ sklearn  AgglomerativeClusteringprecomputed """
        try:
            return AgglomerativeClustering(n_clusters=n_clusters,
                                           metric='precomputed',
                                           linkage='average')
        except TypeError:
            return AgglomerativeClustering(n_clusters=n_clusters,
                                           affinity='precomputed',
                                           linkage='average')

    # ---------- 1)  K ----------
    K = _safe_K()
    if K <= 1:
        #  hub
        proto2hub = {int(i): 0 for i in range(max(1, K))}
        hub_centers = np.ones((1, max(1, K)), np.float32)
        S = np.eye(max(1, K), dtype=np.float32)
        return proto2hub, hub_centers, S

    # ---------- 2) / ----------
    # 2.1 center_cos: [-1,1] -> [0,1]
    C = _resize01(center_cos, K, fill=0.0)
    C = np.clip(C, -1.0, 1.0)
    C = (C + 1.0) / 2.0
    C = _sym01_clip(C)

    # 2.2 log2EN:  -> sigmoid -> [0,1]
    E = _resize01(log2EN, K, fill=0.0)
    E = 1.0 / (1.0 + np.exp(-E))
    E = _sym01_clip(E)

    # 2.3 knn_trans:  -> min-max -> [0,1]
    T = _resize01(knn_trans, K, fill=0.0)
    T = _norm_minmax(T)
    T = _sym01_clip(T)

    # 2.4 responder r in [0,1],  P_ij = 1 - |ri - rj|
    tab = (
        cell_df[cell_df['label'].isin([0, 1])]
        .groupby(['prototype_id','label'])
        .size()
        .unstack(fill_value=0)
    )
    #  1  0
    r = (tab.get(1, 0) / (tab.sum(axis=1) + 1e-9)).reindex(range(K), fill_value=0.5).values.astype(np.float32)
    P = 1.0 - np.abs(r[:, None] - r[None, :]).astype(np.float32)
    P = _sym01_clip(P)

    # ---------- 3)  ----------
    wC, wT, wE, wP = 0.35, 0.30, 0.25, float(resp_prop_weight)
    w_sum = wC + wT + wE + wP
    if w_sum <= 1e-12:
        wC = 1.0; wT = wE = wP = 0.0; w_sum = 1.0
    wC, wT, wE, wP = (wC/w_sum, wT/w_sum, wE/w_sum, wP/w_sum)

    S = (wC*C + wT*T + wE*E + wP*P).astype(np.float32)
    S = _sym01_clip(S)

    # ---------- 4) hub precomputed  ----------
    D = 1.0 - S
    D = 0.5*(D + D.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 1.0).astype(np.float32)

    Z = _agglom_precomputed(int(n_hubs)).fit(D)
    lab = Z.labels_.astype(int)

    # ---------- 5)  ----------
    proto2hub = {int(i): int(lab[i]) for i in range(K)}
    hub_centers = np.vstack([S[lab == h].mean(axis=0) if np.any(lab == h) else np.zeros((K,), np.float32)
                             for h in range(int(n_hubs))]).astype(np.float32)

    return proto2hub, hub_centers, S



def sci_lite(cell_df, k=8):
    """
     kNN / +  log2EN [0,1]  SCI
    DataFrame KK==
    """
    import numpy as np, pandas as pd
    from sklearn.neighbors import KDTree
    df = cell_df.dropna(subset=['pos_x','pos_y']).copy()
    coords = df[['pos_x','pos_y']].to_numpy(np.float32)
    pids = pd.to_numeric(df['prototype_id'], errors='coerce').astype(int).to_numpy()
    uniq = sorted(np.unique(pids).tolist()); idx = {p:i for i,p in enumerate(uniq)}
    if len(df) < k+1: return pd.DataFrame(np.zeros((len(uniq),len(uniq))), index=uniq, columns=uniq)
    tree = KDTree(coords)
    neigh = tree.query(coords, k=min(k+1,len(df)), return_distance=False)[:,1:]
    M = np.zeros((len(uniq),len(uniq)), np.float64)
    for i,row in enumerate(neigh):
        s = idx[pids[i]]
        for j in row:
            t = idx[pids[j]]
            M[s,t] += 1
    R = M / np.maximum(1, M.sum(axis=1, keepdims=True))  # 
    #  0..1
    same = np.diag(R).mean() if R.size else 0.0
    R01 = (R - R.min())/(R.max()-R.min()+1e-6)
    return pd.DataFrame(R01, index=[f"P{p}" for p in uniq], columns=[f"P{p}" for p in uniq])

def lr_lite_scores(expr_df, available_genes):
    """
    expr_df: =sample/proto/hub=
    available_genes: set([...]) -bulk
    (min-max) L*R 
    """
    import numpy as np, pandas as pd
    # IFNGIFNGR1, CXCL12CXCR4, TGFB1TGFBR1/2, EGFEGFR, IL6IL6R, SPP1CD44 ...
    PAIRS = {
        # --  --
        "IFN_axis": [("IFNG", "IFNGR1"), ("IFNG", "IFNGR2")], # IFNGR1IFNGR2
        "TGFb_axis": [("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),
                    ("TGFB3", "TGFBR1"), ("TGFB3", "TGFBR2")], # TGFB1/3  TGFBR1/2 
        "CXCL_axis": [("CXCL12", "CXCR4"),
                    ("CXCL14", "CXCR4")], # CXCL12/14  CXCR4 
        "EGFR_axis": [("EGF", "EGFR"), ("EREG", "EGFR")], # 
        "IL6_axis": [("IL6", "IL6R")], # 

        # --  --
        "CCL_axis": [("CCL5", "CCR1")], # CCL5  CCR1 
        
        "CSF1_axis": [("CSF1", "CSF1R")], # 1 ()
        
        "SPP1_axis": [("SPP1", "ITGAV")], # SPP1() 
        
        "VEGF_PGF_axis": [("VEGFA", "NRP1"), ("VEGFA", "NRP2"), # 
                        ("PGF", "NRP1"), ("PGF", "NRP2")],
                        
        "PDGF_axis": [("PDGFA", "PDGFRB")], # 

        "SLIT_ROBO_axis": [("SLIT2", "ROBO1")], # 
        
        "Immune_Adhesion_axis": [("ITGAL", "ICAM1"), # LFA-1ICAM
                                ("ITGAL", "ICAM2")],

        "MIF_axis": [("MIF", "CD74")], # MIF  CD74 

        "GRN_axis": [("GRN", "SORT1")] #  (GRN) 
    }
    genes_in = [g for g in expr_df.columns if g in available_genes]
    X = expr_df[genes_in].copy()
    # 
    X = (X - X.min())/(X.max()-X.min()+1e-9)
    out={}
    for axis, pairs in PAIRS.items():
        vals=[]
        for L,R in pairs:
            if L in X.columns and R in X.columns:
                vals.append(np.sqrt(np.clip(X[L].values,0,1)*np.clip(X[R].values,0,1)))
        if vals:
            out[axis] = np.mean(np.vstack(vals), axis=0)
    return pd.DataFrame(out, index=expr_df.index)


def diffusion_pseudotime(state_df, n_components=3, root_index=0):
    """
    state_df=hub=//
     coords(ndarray)   pseudotime(0..1)
    """
    import numpy as np
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.sparse.linalg import eigsh
    X = state_df.to_numpy(np.float64)
    if X.size == 0:
        return np.empty((0, 0)), np.array([])
    n_samples, n_features = X.shape
    if n_features == 0:
        return np.zeros((n_samples, min(1, n_components))), np.zeros(n_samples)
    if n_samples <= 1:
        return np.zeros((n_samples, min(max(1, n_components), n_features))), np.zeros(n_samples)
    X = (X - X.mean(0))/(X.std(0)+1e-9)
    gamma = 1.0/max(n_features, 1)
    K = rbf_kernel(X, gamma=gamma)  # 
    # 
    D = np.diag(1.0/(K.sum(axis=1)+1e-9))
    P = D @ K
    # 
    max_nontrivial = min(max(1, n_components), n_samples-1)
    k = max_nontrivial + 1
    if k >= n_samples:
        from scipy.linalg import eigh
        vals, vecs = eigh(P.T)
        order = np.argsort(vals)[::-1][:k]
        vecs = vecs[:, order]
    else:
        vals, vecs = eigsh(P.T, k=k, which='LM')
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order]
    U = vecs[:, 1:(max_nontrivial+1)]  # 
    if U.shape[1] < max_nontrivial:
        pad = max_nontrivial - U.shape[1]
        U = np.pad(U, ((0, 0), (0, pad)), mode='constant')
    # pseudotime root 
    d = np.linalg.norm(U - U[root_index], axis=1)
    pt = (d - d.min())/(d.max()-d.min()+1e-9)
    return U, pt




def proto_anchor_signatures_by_cohort(cohort_name, cell_df, gene_df, sample_dict, gene_list, out_dir,
                                      use_merged=True, q_thresh=0.05, log2fc_min=0.25,
                                      min_cells_per_proto=50, min_patients_each=3):
    os.makedirs(out_dir, exist_ok=True)
    pid_col = "prototype_id_merged" if (use_merged and "prototype_id_merged" in cell_df.columns) else "prototype_id"
    intra_dir = Path(out_dir).parents[1] / "4_gene_functional_analysis" / "1_intra_prototype_DE_by_phenotype"
    rows=[]
    if intra_dir.exists():
        for p in sorted(pd.to_numeric(cell_df[pid_col], errors='coerce').dropna().unique().astype(int)):
            f = intra_dir / f"P{p}_DEGs.csv"
            if f.exists():
                df = pd.read_csv(f); df['prototype_id'] = p; rows.append(df)
    out = (pd.concat(rows, ignore_index=True) if rows else
           pd.DataFrame(columns=['gene','log2fc_Resp_vs_NonResp','p_value','t_stat','prototype_id']))
    out.to_csv(os.path.join(out_dir, f"{cohort_name}_prototype_signatures.csv"), index=False)


def celltype_proportion_per_prototype(cell_df, feature_names, cohort, out_dir, use_merged=True):
    os.makedirs(out_dir, exist_ok=True)
    pid_col = "prototype_id_merged" if (use_merged and "prototype_id_merged" in cell_df.columns) else "prototype_id"
    type_cols = [f for f in feature_names if f.startswith('Type_') and f in cell_df.columns]
    df = cell_df.copy()
    if type_cols:
        df['cell_type'] = df[type_cols].idxmax(axis=1).str.replace('Type_','', regex=False)
    else:
        df['cell_type'] = 'NA'
    comp = df.groupby(pid_col)['cell_type'].value_counts(normalize=True).unstack(fill_value=0)
    comp.to_csv(os.path.join(out_dir, f"{cohort}_celltype_proportion_by_proto.csv"))
    comp.plot(kind='bar', stacked=True, figsize=(max(8, 0.8*len(comp.index)), 6), width=0.8)
    plt.title(f'{cohort}  Cell-type composition per prototype')
    plt.xlabel('Prototype'); plt.ylabel('Proportion')
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cohort}_celltype_proportion_by_proto.png"), dpi=260, bbox_inches='tight')
    plt.close()


def _resolve_pid_col(df, pid_col=None):
    if pid_col and pid_col in df.columns:
        return pid_col
    return 'prototype_id_merged' if 'prototype_id_merged' in df.columns else 'prototype_id'


def attach_cell_gate_strength(cell_df, sample_dict, agg="mean"):
    out = []
    for sname, sdf in cell_df.groupby('sample_name'):
        g = get_sample_mod(sample_dict, sname)
        if g is None:
            continue
        try:
            ei = g['cell','c_g','gene'].edge_index.detach().cpu().numpy()
            eattr = getattr(g['cell','c_g','gene'],'edge_attr', None)
            if eattr is None or eattr.numel()==0:
                continue
            w = eattr.detach().cpu().numpy().reshape(-1)
            rows = ei[0].astype(int)
        except Exception:
            continue

        #  gate  index
        if 'node_idx' in sdf.columns:
            cell_ids = pd.to_numeric(sdf['node_idx'], errors='coerce').astype('Int64')
            valid_idx = cell_ids.notna()
            id2r = {int(cid): i for i, cid in enumerate(cell_ids[valid_idx].astype(int).tolist())}
            gate_vec = np.zeros((len(sdf),), np.float32)
            cnt_vec  = np.zeros((len(sdf),), np.float32)
            for r, ww in zip(rows, w):
                rr = id2r.get(int(r), None)
                if rr is None:
                    continue
                if agg == "max":
                    gate_vec[rr] = max(gate_vec[rr], float(ww))
                else:
                    gate_vec[rr] += float(ww)
                    cnt_vec[rr]  += 1.0
            if agg == "mean":
                m = cnt_vec > 0
                gate_vec[m] = gate_vec[m] / cnt_vec[m]
        else:
            gate_vec = np.full((len(sdf),), np.nan, np.float32)

        #  gate_strength  cell_df 
        tmp = pd.DataFrame({'gate_strength': gate_vec}, index=sdf.index)
        out.append(tmp)

    #  merge sample_name
    cell_df2 = cell_df.copy()
    if out:
        merged = pd.concat(out, axis=0)
        #  join
        cell_df2 = cell_df2.join(merged['gate_strength'], how='left')
    else:
        cell_df2['gate_strength'] = np.nan

    #  [0,1] 
    gs = cell_df2['gate_strength'].to_numpy(np.float32)
    finite = np.isfinite(gs)
    if finite.any():
        lo, hi = np.percentile(gs[finite], [1, 99])
        rng = max(1e-6, hi - lo)
        cell_df2['gate_strength_norm'] = np.clip((gs - lo) / rng, 0, 1)
    else:
        cell_df2['gate_strength_norm'] = np.nan

    return cell_df2



def overlay_cell_gate_per_sample(cell_df, out_dir="./analysis_out/gate_overlays",
                                 min_cells=50, use_merged=False):
    os.makedirs(out_dir, exist_ok=True)
    need = {'pos_x','pos_y','sample_name','gate_strength_norm'}
    if not need.issubset(cell_df.columns):
        print("[GATE-OVERLAY] missing columns; skip.")
        return
    for sample, sdf in cell_df.groupby('sample_name'):
        sub = sdf.dropna(subset=['pos_x','pos_y','gate_strength_norm'])
        if len(sub) < min_cells: 
            continue
        plt.figure(figsize=(7,6))
        sc = plt.scatter(sub['pos_x'], sub['pos_y'], s=2, alpha=0.7,
                         c=sub['gate_strength_norm'], cmap='viridis')
        plt.gca().invert_yaxis(); plt.axis("equal")
        plt.title(f"{sample}  Gate strength (cellgene)")
        cbar = plt.colorbar(sc); cbar.set_label("normalized gate")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample}_gate_strength.png"), dpi=260)
        plt.close()


def visualize_gate_on_umap_and_proto(cell_df, out_dir="./analysis_out/gate_viz"):
    os.makedirs(out_dir, exist_ok=True)
    # UMAP 
    if {'umap_x','umap_y','gate_strength_norm'}.issubset(cell_df.columns):
        df2d = cell_df.dropna(subset=['umap_x','umap_y','gate_strength_norm'])
        if not df2d.empty:
            plt.figure(figsize=(8,7))
            sc = plt.scatter(df2d['umap_x'], df2d['umap_y'], s=3, alpha=0.6,
                             c=df2d['gate_strength_norm'], cmap='magma')
            plt.title("Gate strength on UMAP"); plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
            cbar = plt.colorbar(sc); cbar.set_label("normalized gate")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "umap_gate_strength.png"), dpi=280); plt.close()
    else:
        print("[GATE] Missing UMAP coordinates; skip UMAP gate scatter.")

    # /
    pid_col = "prototype_id_merged" if "prototype_id_merged" in cell_df.columns else "prototype_id"
    dfp = cell_df.dropna(subset=[pid_col, 'gate_strength']).copy()
    if not dfp.empty:
        dfp[pid_col] = pd.to_numeric(dfp[pid_col], errors='coerce').astype('Int64')
        dfp = dfp.dropna(subset=[pid_col])
        dfp[pid_col] = dfp[pid_col].astype(int)
        plt.figure(figsize=(max(10, 1.3*dfp[pid_col].nunique()), 6))
        sns.boxplot(data=dfp, x=pid_col, y='gate_strength', showfliers=False)
        sns.stripplot(data=dfp.sample(min(len(dfp), 20000), random_state=42),
                      x=pid_col, y='gate_strength', color='0.3', size=2, alpha=0.3)
        plt.title("Gate strength by prototype")
        plt.xlabel("Prototype"); plt.ylabel("gate (cellgene, aggregated)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "gate_by_prototype.png"), dpi=280); plt.close()



def viz_ssgsea_advanced(scores_df, out_dir, title="ssGSEA", top_n=30, two_group_diff=None):
    """
    scores_df: index = samples/prototypes, columns = pathways ()
    two_group_diff:  DataFrame ['pathway','delta','q_value'] /
    """
    os.makedirs(out_dir, exist_ok=True)
    if scores_df.empty:
        print("[ssGSEA-VIZ] empty"); return
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
    # 1) z-score by pathway clustermap
    Z = scores_df.copy()
    Z = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-9)
    #  top_n 
    var = Z.var().sort_values(ascending=False)
    pick = var.head(min(top_n, len(var))).index
    Zp = Z[pick]
    g = sns.clustermap(Zp, cmap="RdBu_r", center=0, linewidths=0.2, figsize=(12, max(6, 0.35*len(Zp))))
    g.fig.suptitle(f"{title}  clustered (top var {len(pick)})", y=1.02)
    g.savefig(os.path.join(out_dir, "ssgsea_clustermap.png"), dpi=320, bbox_inches="tight"); plt.close(g.fig)

    # 2)  two_group_diff B P{pid} vs /
    if two_group_diff is not None and not two_group_diff.empty:
        df = two_group_diff.copy()
        df['neglog10q'] = -np.log10(df['q_value'].replace(0, 1e-300))
        df = df.sort_values(['q_value', 'delta'], ascending=[True, False]).head(80)
        plt.figure(figsize=(9,6))
        sc = plt.scatter(df['delta'], df['neglog10q'], s=40+120* (df['neglog10q']/ (df['neglog10q'].max()+1e-6)),
                         alpha=0.8)
        for _, r in df.head(15).iterrows():
            plt.text(r['delta'], r['neglog10q'], r['pathway'], fontsize=8)
        plt.axhline(-np.log10(0.05), ls='--', lw=1, color='grey')
        plt.xlabel("Enrichment difference ()"); plt.ylabel("-log10(q)")
        plt.title(f"{title}  differential pathways")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "ssgsea_diff_volcano.png"), dpi=320); plt.close()

def collect_and_viz_ssgsea_outputs(base_dir):
    """
     A/B  ssGSEA 
    - A_anchor_proto_ssgsea_scores.csv   clustermap
    - B_bulk_P{pid}_ssgsea_diff.csv     volcano
    """
    A_csv = os.path.join(base_dir, "A_anchor_proto_ssgsea_scores.csv")
    if os.path.exists(A_csv):
        A = pd.read_csv(A_csv, index_col=0)
        viz_ssgsea_advanced(A, os.path.join(base_dir, "A_viz"), title="ssGSEA (cell-anchored prototypes)", top_n=30)

    #  pid  diff
    for f in sorted(Path(base_dir).glob("B_bulk_P*_ssgsea_diff.csv")):
        df = pd.read_csv(f)
        pid = re.search(r"P(\d+)", f.name)
        tag = f"P{pid.group(1)}" if pid else "Px"
        viz_ssgsea_advanced(
            scores_df=pd.DataFrame(),  #  clustermap
            out_dir=os.path.join(base_dir, f"B_viz_{tag}"),
            title=f"ssGSEA (bulk) {tag}",
            two_group_diff=df.rename(columns={"mean_in_"+tag: "mean_in"})
                                 .rename(columns={"pvalue":"p_value"})
                                 .assign(delta=lambda x: x['mean_in'] - x['mean_outside'])
                                 [['pathway','delta','q_value']]
        )



# =========================================================================
# NEW: Run per-cohort (fully isolated outputs)
# =========================================================================
def run_per_cohort_pipeline(
    cohort_name: str,
    model,
    cell_df_all: pd.DataFrame,
    gene_df_all: pd.DataFrame,
    sample_dict: dict,
    gene_list: list,
    cell_features: list,
    svs_root_map: dict,
    cellgeojson_root_map: dict,
    base_out_dir: str,
    core_thr: float = 0.8,
):
    # 
    out_dir = os.path.join(base_out_dir, f"COHORT_{cohort_name}")
    os.makedirs(out_dir, exist_ok=True)
    done_flag = Path(out_dir) / ".done"
    if CACHE_ENABLED and done_flag.exists():
        print(f"[{cohort_name}] cache hit. Outputs: {out_dir}")
        return

    #  __main__ 
    cell_df = cell_df_all.copy()
    gene_df = gene_df_all.copy()

    cohort_norm = _normalize_cohort_key(cohort_name)
    if 'cohort' in cell_df.columns:
        mask = cell_df['cohort'].map(_normalize_cohort_key) == cohort_norm
        cell_df_coh = cell_df[mask].copy()
    else:
        cell_df_coh = cell_df.copy()
    if cell_df_coh.empty:
        print(f"[{cohort_name}] no cells; skip.")
        return

    # gene_df  cohort  sample_name 
    if 'cohort' in gene_df.columns:
        gene_mask = gene_df['cohort'].map(_normalize_cohort_key) == cohort_norm
        gene_df_coh = gene_df[gene_mask].copy()
    else:
        gene_df_coh = gene_df[gene_df['sample_name'].isin(cell_df_coh['sample_name'].unique())].copy()

    # 3) UMAP & 
    plot_all_umaps(cell_df_coh, gene_df_coh, out_dir)

    # 4) 
    analyze_cell_features(cell_df_coh, cell_features, out_dir, use_merged=True)

    # 5) 
    patient_summary_df = analyze_phenotype_correlation(cell_df_coh, out_dir)
    if isinstance(patient_summary_df, pd.DataFrame) and not patient_summary_df.empty:
        summary_dir = os.path.join(out_dir, "6_patient_summary")
        os.makedirs(summary_dir, exist_ok=True)
        patient_summary_df.to_csv(os.path.join(summary_dir, "patient_feature_summary.csv"))

    # 6) -bulk &  DE
    run_archetype_gene_analysis(
        gene_df=gene_df_coh, cell_df=cell_df_coh, gene_list=gene_list,
        output_dir=out_dir, intra_log2fc_threshold=0.10,
        inter_log2fc_threshold=0.10, min_cells_per_proto=50
    )

    # 7) 
    analyze_phenotype_specific_interactions(cell_df_coh, out_dir)

    # 8) /3D + 
    viz_dir = os.path.join(out_dir, "ANCHOR_ON_CELL", "B_visualize_protos")
    os.makedirs(viz_dir, exist_ok=True)
    plot_cell_prototypes_umap_mod(cell_df_coh, out_dir=viz_dir, use_merged=True)
    overlay_cell_prototypes_per_sample_mod(
        cell_df=cell_df_coh,
        sample_dict=sample_dict,
        svs_root_map=svs_root_map,
        out_dir=os.path.join(viz_dir, "overlays"),
        use_merged=True
    )

    # 9)  & 
    sig_dir = os.path.join(out_dir, "ANCHOR_ON_CELL", "C_signatures")
    os.makedirs(sig_dir, exist_ok=True)
    comp_dir = os.path.join(out_dir, "ANCHOR_ON_CELL", "D_composition")
    os.makedirs(comp_dir, exist_ok=True)

    # 9.1 
    try:
        from statsmodels.stats.multitest import multipletests  # noqa: F401
        proto_anchor_signatures_by_cohort(
            cohort_name=cohort_name,
            cell_df=cell_df_coh, gene_df=gene_df_coh,
            sample_dict=sample_dict, gene_list=gene_list,
            out_dir=sig_dir, use_merged=True,
            q_thresh=0.05, log2fc_min=0.25,
            min_cells_per_proto=50, min_patients_each=2
        )
    except Exception as e:
        print(f"[WARN] signatures failed on {cohort_name}: {e}")

    # 9.2 
    try:
        celltype_proportion_per_prototype(
            cell_df=cell_df_coh, feature_names=cell_features,
            cohort=cohort_name, out_dir=comp_dir, use_merged=True
        )
    except Exception as e:
        print(f"[WARN] composition failed on {cohort_name}: {e}")

    # === ANCHOR_ON_CELL  A) Ecology ===
    ecol_dir = os.path.join(out_dir, "ANCHOR_ON_CELL", "A_ecology")
    os.makedirs(ecol_dir, exist_ok=True)

    # 0)  cell_type Type_* one-hot
    cell_df_coh = _ensure_celltype_column(cell_df_coh, cell_features)

    # 1) kNN
    neigh_proto_prop, neigh_type_prop = compute_local_ecology_kNN(cell_df_coh, k=15)
    if not neigh_proto_prop.empty:
        cell_df_coh = pd.concat([cell_df_coh, neigh_proto_prop], axis=1)

    # 2)   
    anchor_proto_mix, anchor_type_mix = aggregate_ecology_by_anchor(
        cell_df_coh, neigh_proto_prop, neigh_type_prop, min_cells=200
    )
    if not anchor_proto_mix.empty:
        anchor_proto_mix.to_csv(os.path.join(ecol_dir, "anchor_proto_mix.csv"))
    if not anchor_type_mix.empty:
        anchor_type_mix.to_csv(os.path.join(ecol_dir, "anchor_type_mix.csv"))
    plot_anchor_ecology(anchor_proto_mix, anchor_type_mix, ecol_dir)

    # 3) 
    for pid in sorted(cell_df_coh['prototype_id'].dropna().unique()):
        dfP, dfT = prototype_radial_profile(cell_df_coh, anchor_pid=int(pid), radii=(20,40,80,160))
        if not dfP.empty or not dfT.empty:
            plot_radial_profile(dfP, dfT, anchor_pid=int(pid), out_dir=ecol_dir)

    # 4) kNN 
    T = knn_transition_matrix(cell_df_coh, k=8)
    if not T.empty:
        T.to_csv(os.path.join(ecol_dir, "knn_transition_matrix.csv"))
        plot_transition_matrix(T, os.path.join(ecol_dir, "knn_transition_matrix.png"))

    # 5) 
    sig = per_sample_ecology_signature(cell_df_coh, neigh_type_prop, anchor_proto_mix, min_cells=100)
    if not sig.empty:
        sig_out = os.path.join(ecol_dir, "per_sample_ecology_signature.csv")
        sig.to_csv(sig_out, index=False)

    # Quick test:  vs 
    try:
        eco_sig = pd.read_csv(os.path.join(ecol_dir, "per_sample_ecology_signature.csv"))
        lab_map = cell_df_coh.groupby('sample_name')['label']\
                             .agg(lambda x: int(stats.mode(x, keepdims=True).mode[0]))\
                             .to_dict()
        eco_sig['label'] = eco_sig['sample_name'].map(lab_map)
        test_cols = [c for c in eco_sig.columns if c.startswith('CT:')]
        if test_cols:
            from itertools import product
            for pid, c in product(sorted(eco_sig['anchor_pid'].unique()), test_cols):
                sub = eco_sig[eco_sig['anchor_pid'] == pid]
                a = sub[sub['label'] == 1][c].values
                b = sub[sub['label'] == 0][c].values
                if len(a) >= 3 and len(b) >= 3:
                    t, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                    if p < 0.05:
                        print(f"[{cohort_name}] ECO assoc: anchor P{pid}  {c}: p={p:.3g} (R={a.mean():.3f} vs N={b.mean():.3f})")
    except Exception as e:
        print(f"[WARN] quick ecology test failed: {e}")

    # === ANCHOR_ON_CELL  A2) Gene modality on anchors ===
    a2_dir = os.path.join(out_dir, "ANCHOR_ON_CELL", "A2_gene_on_anchor")
    os.makedirs(a2_dir, exist_ok=True)
    anchor_geneproto_mix = compute_anchor_geneproto_mix(cell_df_coh, gene_df_coh, sample_dict, min_cells=200)
    if not anchor_geneproto_mix.empty:
        anchor_geneproto_mix.to_csv(os.path.join(a2_dir, "anchor_geneproto_mix.csv"))
        plt.figure(figsize=(8, 0.6*len(anchor_geneproto_mix.index) + 3))
        sns.heatmap(anchor_geneproto_mix, cmap='rocket')
        plt.title('Anchor Prototype  Gene-Prototype Mix')
        plt.xlabel('Gene prototype'); plt.ylabel('Anchor prototype')
        plt.tight_layout()
        plt.savefig(os.path.join(a2_dir, "anchor_geneproto_mix.png"), dpi=240)
        plt.close()

    # === ANCHOR_ON_CELL  A3) Eco-core vs edge DE ===
    a3_dir = os.path.join(out_dir, "ANCHOR_ON_CELL", "A3_core_edge_DE")
    os.makedirs(a3_dir, exist_ok=True)
    top_pids = cell_df_coh['prototype_id'].value_counts().head(3).index.tolist()
    for pid in top_pids:
        try:
            gene_core_vs_edge_DE(
                cell_df_coh, gene_df_coh, sample_dict, anchor_pid=int(pid),
                prop_thr=core_thr, p_value_threshold=0.05, log2fc_thr=0.1,
                min_patients_each=3, out_dir=a3_dir
            )
        except Exception as e:
            print(f"[WARN] eco DE for P{pid} failed: {e}")

    # === ANCHOR_ON_CELL  E) ssGSEA + bulk
    try:
        ssgsea_dir = os.path.join(out_dir, "ANCHOR_ON_CELL", "E_ssGSEA")
        os.makedirs(ssgsea_dir, exist_ok=True)
        run_ssgsea_cell_anchor_and_bulk(
            gene_df=gene_df_coh, cell_df=cell_df_coh, gene_list=gene_list,
            out_dir=ssgsea_dir, sample_dict=sample_dict,
            min_cells_per_proto=50, gene_sets='/DATA/linzhiquan/lzq/HEST-main/archetype_analysis_results_final_v11_fix1/h.all.v2023.2.Hs.symbols.gmt'
        )
        collect_and_viz_ssgsea_outputs(out_dir)
    except Exception as e:
        print(f"[WARN] ssGSEA failed on {cohort_name}: {e}")

    # 10) WSI 
    wsi_dir = os.path.join(viz_dir, "wsi_proto_overlays")
    os.makedirs(wsi_dir, exist_ok=True)
    svs_root  = _resolve_cohort_path(svs_root_map, cohort_name)
    gj_root   = _resolve_cohort_path(cellgeojson_root_map, cohort_name)
    if svs_root and gj_root:
        for s, sdf in cell_df_coh.groupby('sample_name'):
            out_png = os.path.join(
                wsi_dir, f"{cohort_name}__{s}__proto_{'merged' if 'prototype_id_merged' in sdf.columns else 'raw'}.png"
            )
            overlay_wsi_prototypes_for_sample_mod(
                sample_name=str(s), cell_df=cell_df_coh,
                svs_root=svs_root, cellgeojson_root=gj_root,
                out_png=out_png, use_merged=True, target_level=1,
                alpha_fill=130, k_vote=3, draw_outline=True,
                core_threshold=core_thr
            )


    # === WSI Overlays: prototypes / gate / core-edge ===
    svs_root = _resolve_cohort_path(svs_root_map, cohort_name)
    geo_root = _resolve_cohort_path(cellgeojson_root_map, cohort_name)
    wsi_dir = os.path.join(out_dir, "WSI_OVERLAYS")
    os.makedirs(wsi_dir, exist_ok=True)

    for sname in sorted(cell_df_coh['sample_name'].unique().tolist()):
        try:
            overlay_wsi_prototypes_for_sample_mod(
                sample_name=sname, cell_df=cell_df_coh,
                svs_root=svs_root, cellgeojson_root=geo_root,
                out_png=os.path.join(wsi_dir, f"{sname}__proto_merged.png"),
                use_merged=True, target_level=1, k_vote=3,
                core_threshold=core_thr
            )
        except Exception as e:
            print(f"[WARN] WSI proto for {sname}: {e}")

        # gate  attach_cell_gate_strength
        if 'gate_strength_norm' in cell_df_coh.columns:
            try:
                overlay_wsi_gate_for_sample_mod(
                    sample_name=sname, cell_df=cell_df_coh,
                    svs_root=svs_root, cellgeojson_root=geo_root,
                    out_png=os.path.join(wsi_dir, f"{sname}__gate.png"),
                    scalar_col='gate_strength_norm', target_level=1
                )
            except Exception as e:
                print(f"[WARN] WSI gate for {sname}: {e}")

        # core-edge anchor 
        try:
            top_anchor = int(cell_df_coh['prototype_id'].value_counts().idxmax())
            overlay_wsi_core_edge_for_sample_mod(
                sample_name=sname, cell_df=cell_df_coh,
                svs_root=svs_root, cellgeojson_root=geo_root,
                out_png=os.path.join(wsi_dir, f"{sname}__core_edge_P{top_anchor}.png"),
                anchor_pid=top_anchor, prop_thr=core_thr, k_vote=3, target_level=1
            )
        except Exception as e:
            print(f"[WARN] WSI core-edge for {sname}: {e}")


    print(f"[{cohort_name}]  done. Outputs: {out_dir}")

    if CACHE_ENABLED:
        summary = {
            "cohort": cohort_name,
            "script_sha1": SCRIPT_SHA1,
            "sample_hash": _json_hash(sorted(cell_df_coh['sample_name'].astype(str).unique())),
        }
        try:
            with open(done_flag, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[WARN] failed to write done flag for {cohort_name}: {e}")



def _get_any_centers_shared(model, device):
    """
    
      (A) 
      (B)  center/proto/cluster 
      (C) / weight  embedding_dim 
      (D) None/KMeans
    """
    # A) 
    cand_names = [
        "shared_cluster_centers", "shared_centers", "cluster_centers",
        "centers_shared", "centers", "prototypes", "prototype_centers",
        "W_shared", "W", "proto_weight",
    ]
    for nm in cand_names:
        if hasattr(model, nm):
            t = getattr(model, nm)
            if t is not None and torch.is_tensor(t) and t.ndim == 2 and t.numel() > 0:
                return t.detach().float().to(device)

    # B)  + 
    emb_dim = getattr(model, "embedding_dim", None)
    for name, p in model.named_parameters(recurse=True):
        if not (torch.is_tensor(p) and p.ndim == 2 and p.numel() > 0):
            continue
        lowname = name.lower()
        if any(k in lowname for k in ["center", "proto", "cluster"]):
            if (emb_dim is None) or (p.size(1) == emb_dim) or (p.size(0) == emb_dim):
                return p.detach().float().to(device)

    # C)  headweight  embedding_dim 
    if emb_dim is not None:
        for _, mod in model.named_modules():
            w = getattr(mod, "weight", None)
            if torch.is_tensor(w) and w.ndim == 2 and w.size(1) == emb_dim and w.numel() > 0:
                return w.detach().float().to(device)

    # D) 
    return None



def _bootstrap_centers_from_embeddings(cell_df, gene_df, k=None, device="cpu"):
    """
     embed_*  KMeans 
     K model.k_cell / model.num_shared_clusters / 6
    """
    from sklearn.cluster import KMeans
    import numpy as np
    import pandas as pd
    #  embeddings
    def _X(df):
        cols = [c for c in df.columns if str(c).startswith("embed_")]
        return df[cols].to_numpy(np.float32, copy=False) if (not df.empty and cols) else None
    Xc = _X(cell_df); Xg = _X(gene_df)
    if Xc is None and Xg is None:
        raise RuntimeError(" embed_* ")
    X = Xc if (Xg is None) else (Xg if (Xc is None) else np.vstack([Xc, Xg]))
    # 
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    #  K
    if k is None:
        k =  (globals().get("NUM_SHARED_CLUSTERS", None)
              or getattr(globals().get("model", None), "k_cell", None)
              or getattr(globals().get("model", None), "num_shared_clusters", None)
              or 6)
    k = int(max(2, k))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(X)
    C = km.cluster_centers_.astype(np.float32, copy=False)
    return torch.from_numpy(C).to(device)

def debug_wsi_assets(sample_name, svs_root, cellgeojson_root):
    print(f"[WSI-DEBUG] sample={sample_name}")
    print(f"  svs_root={svs_root}  geojson_root={cellgeojson_root}")
    print(f"  svs_found={_find_svs(svs_root, sample_name)}")
    print(f"  geojson_found={_find_geojson(cellgeojson_root, sample_name)}")

def ensure_celltype_from_onehot(cell_df, feature_names=None):
    """
     df.columns  Type_* feature_names NA
    """
    df = cell_df.copy()
    if 'cell_type' in df.columns:
        return df

    # 1)  df.columns 
    type_cols = [c for c in df.columns if str(c).startswith('Type_')]

    # 2)  df  feature_names Type_* 
    if not type_cols and feature_names is not None:
        type_cols = [c for c in feature_names if str(c).startswith('Type_') and c in df.columns]

    if type_cols:
        df['cell_type'] = df[type_cols].idxmax(axis=1).str.replace('Type_', '', regex=False)
    else:
        df['cell_type'] = 'NA'
    return df



def gate_strength_within_proto_by_type(cell_df,
                                       gate_col='gate_strength_norm',
                                       min_cells_per_type=50,
                                       out_dir='./analysis_out/gate_by_type'):
    """
     cell_type  gate  z-score 
    """
    import os, numpy as np, pandas as pd
    from scipy import stats
    os.makedirs(out_dir, exist_ok=True)
    df = ensure_celltype_from_onehot(cell_df)

    if gate_col not in df.columns or 'prototype_id' not in df.columns:
        print(f"[GATE] missing columns: {gate_col} or prototype_id"); return df

    # 
    df = df.copy()
    df[gate_col] = pd.to_numeric(df[gate_col], errors='coerce').astype(float)
    df = df.dropna(subset=[gate_col, 'prototype_id', 'cell_type'])

    # --- A)  ---
    g = df.groupby(['prototype_id','cell_type'])[gate_col]
    summary = g.agg(n='size', mean='mean', median='median', std='std').reset_index()

    # 
    proto_mean = df.groupby('prototype_id')[gate_col].mean().rename('proto_mean')
    summary = summary.merge(proto_mean, on='prototype_id', how='left')
    summary['delta_type_vs_proto'] = summary['mean'] - summary['proto_mean']
    summary = summary.sort_values(['prototype_id','delta_type_vs_proto'], ascending=[True, False])
    summary.to_csv(os.path.join(out_dir, 'gate_proto_type_summary.csv'), index=False)

    # --- B)  z-score ---
    # z1
    df['gate_z_within_proto'] = (
        (df[gate_col] - df.groupby('prototype_id')[gate_col].transform('mean')) /
        (df.groupby('prototype_id')[gate_col].transform('std') + 1e-9)
    )
    # z2
    mu_pt = df.groupby(['prototype_id','cell_type'])[gate_col].transform('mean')
    sd_pt = df.groupby(['prototype_id','cell_type'])[gate_col].transform('std')
    df['gate_type_z_within_proto'] = (df[gate_col] - mu_pt) / (sd_pt + 1e-9)

    # =
    df['gate_residual_vs_type_in_proto'] = df[gate_col] - mu_pt

    # --- C) Kruskal + Dunn  scikit-posthocs ---
    rows=[]
    for pid, sub in df.groupby('prototype_id'):
        # 
        ok_types = [t for t, ss in sub.groupby('cell_type') if len(ss)>=min_cells_per_type]
        if len(ok_types) < 2:
            continue
        arrays = [sub[sub['cell_type']==t][gate_col].values for t in ok_types]
        try:
            H,p = stats.kruskal(*arrays, nan_policy='omit')
        except Exception:
            H,p = np.nan, np.nan
        rows.append({'prototype_id': int(pid), 'n_types': len(ok_types), 'kruskal_H': H, 'kruskal_p': p})
    stat_df = pd.DataFrame(rows)
    if not stat_df.empty:
        stat_df.to_csv(os.path.join(out_dir, 'gate_proto_type_stats.csv'), index=False)

    # Dunn pip install scikit-posthocs
    try:
        import scikit_posthocs as sp
        dd=[]
        for pid, sub in df.groupby('prototype_id'):
            # 
            if sub['cell_type'].value_counts().ge(min_cells_per_type).sum() < 2:
                continue
            dunn = sp.posthoc_dunn(sub, val_col=gate_col, group_col='cell_type', p_adjust='fdr_bh')
            dunn.index = [f"{pid}:{i}" for i in dunn.index]
            dunn.columns = [f"{pid}:{j}" for j in dunn.columns]
            dd.append(dunn)
        if dd:
            pd.concat(dd, axis=0).to_csv(os.path.join(out_dir, 'gate_dunn_pairwise_by_proto.csv'))
    except Exception:
        pass

    return df

def safe_center_cos(model, W_shared=None, use_ema=True):
    """
     S  C
    model.centers_cell_ema / model.centers_cellW_shared None
    """
    import torch, numpy as np
    C = None

    # 1) 
    if use_ema and hasattr(model, 'centers_cell_ema') and model.centers_cell_ema is not None:
        C = model.centers_cell_ema
    elif hasattr(model, 'centers_cell') and model.centers_cell is not None:
        C = model.centers_cell

    # 2) /
    if C is None and W_shared is not None:
        C = W_shared

    if C is None:
        return None, None

    #  ->  -> 
    if not torch.is_tensor(C):
        C = torch.as_tensor(C)
    C = C.detach().float().cpu()
    C = torch.nn.functional.normalize(C, dim=1)
    S = (C @ C.T).numpy()
    return S, C.numpy()



def overlay_wsi_gate_by_celltype(sample_name,
                                 cell_df,
                                 svs_root,
                                 cellgeojson_root,
                                 out_png,
                                 target_cell_type: str,
                                 scalar_col='gate_type_z_within_proto',  #  B 
                                 target_level=1,
                                 k_vote=3,
                                 alpha_outline=70,
                                 z_clip=(-3, 3)):
    """
     WSI  target_cell_type  gate 
     OpenSlide + geojson 
    """
    import numpy as np, os
    if (OpenSlide is None) or (Image is None):
        print("[WSI-GATE-CT] OpenSlide or PIL not available; skip.")
        return

    svs_path = _find_svs(svs_root, sample_name)
    gj_path  = _find_geojson(cellgeojson_root, sample_name)
    if not (svs_path and os.path.exists(svs_path) and gj_path and os.path.exists(gj_path)):
        print(f"[WSI-GATE-CT] skip {sample_name}: missing svs/geojson (svs_root={svs_root}, geojson_root={cellgeojson_root})")
        return

    sdf = cell_df[(cell_df['sample_name']==sample_name)].copy()
    if sdf.empty or scalar_col not in sdf.columns or 'cell_type' not in sdf.columns:
        print(f"[WSI-GATE-CT] {sample_name}: missing {scalar_col}/cell_type"); return

    sdf = sdf.dropna(subset=['pos_x', 'pos_y', scalar_col])
    if sdf.empty:
        print(f"[WSI-GATE-CT] {sample_name}: no cells with positions.")
        return

    #  polygon  alpha 
    scalar = np.clip(pd.to_numeric(sdf[scalar_col], errors='coerce').astype(float), *z_clip)
    target_mask = (sdf['cell_type'].astype(str) == str(target_cell_type)).to_numpy(np.bool_)

    cell_xy_lvl0 = sdf[['pos_x', 'pos_y']].to_numpy(np.float32)
    norm_val = (np.clip(scalar, z_clip[0], z_clip[1]) - z_clip[0]) / (z_clip[1] - z_clip[0] + 1e-6)
    cell_scalar = np.where(target_mask, norm_val, 0.0).astype(np.float32)

    slide = OpenSlide(str(svs_path))
    polys_lvl0 = load_cell_polygons(gj_path, slide=slide)
    slide.close()
    if not polys_lvl0:
        print(f"[WSI-GATE-CT] {sample_name}: polygons empty.")
        return

    scores, conf = _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, cell_scalar, k=k_vote)
    poly_scores = np.clip(scores * np.power(np.clip(conf, 0.0, 1.0), 0.5), 0, 1)
    if not np.any(poly_scores > 1e-6):
        print(f"[WSI-GATE-CT] {sample_name}: no signal for {target_cell_type}.")
        return

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    overlay_mask_heatmap_on_wsi_mod(
        svs_path=svs_path,
        polys_lvl0=polys_lvl0,
        poly_scores01=poly_scores,
        out_png=out_png,
        target_level=target_level,
        blend='sum',
        sigma_lvl0=32.0,
        cmap_name='magma',
        cmap_use_full=True,
        alpha_gamma=0.5,
        alpha_max=240,
        alpha_floor=150,
        brighten_bg=0.08,
        draw_all_outlines=False,
        low_clip_q=8.0,
        high_clip_q=99.2,
        colorbar_h_ratio=0.45,
        colorbar_w_px=90,
        colorbar_pad_px=28,
        colorbar_title=str(target_cell_type),
        fallback_disk_px=4,
        grow_cover_px=4,
        post_smooth_frac=0.0,
        alpha_smooth_frac=0.0,
    )


def plot_hub_visuals(cell_df, S_all, proto2hub, out_dir, umap_cols=None, vmax=None, proto_labels=None):
    """
    
      -  S_all /
      - hub proto->hub 
      - UMAP  hub_id umap_cols
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # --- 1) S_all  ---
    S = np.asarray(S_all, dtype=np.float32)
    S = 0.5*(S+S.T); np.fill_diagonal(S, 1.0)
    K = S.shape[0]
    if vmax is None:
        vmax = float(max(1.0, np.percentile(np.abs(S), 98))) if K>0 else 1.0
    default_labels = [f'P{i}' for i in range(K)]
    if proto_labels is not None and len(proto_labels) == K:
        labels = [str(l) for l in proto_labels]
    else:
        labels = default_labels
    plt.figure(figsize=(max(6,0.35*K+2), max(6,0.35*K+2)))
    im = plt.imshow(S, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    plt.xticks(range(K), labels, rotation=45, ha='right')
    plt.yticks(range(K), labels)
    plt.title('Prototype similarity (fused S_all)')
    plt.colorbar(im, fraction=0.03, pad=0.02, label='similarity [0,1]')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hub_similarity_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, 'hub_similarity_heatmap.svg'))
    plt.close()

    # --- 2) hub  +  ---
    m = pd.DataFrame({'prototype_id': list(proto2hub.keys()),
                      'hub_id': [proto2hub[k] for k in proto2hub]})
    m = m.sort_values(['hub_id','prototype_id'])
    m.to_csv(os.path.join(out_dir, 'proto2hub.csv'), index=False)

    edges=[]
    for i in range(K):
        for j in range(i+1, K):
            src = labels[i] if i < len(labels) else f'P{i}'
            dst = labels[j] if j < len(labels) else f'P{j}'
            edges.append((src, dst, float(S[i,j])))
    pd.DataFrame(edges, columns=['src','dst','weight']).to_csv(
        os.path.join(out_dir, 'hub_edges_from_S.csv'), index=False)

    # --- 3) UMAP  ---
    if umap_cols and all(c in cell_df.columns for c in umap_cols):
        sub = cell_df.dropna(subset=list(umap_cols)+['hub_id'])
        if not sub.empty:
            plt.figure(figsize=(6.8,6.0))
            sc = plt.scatter(sub[umap_cols[0]], sub[umap_cols[1]],
                             c=sub['hub_id'].astype(int), s=3, alpha=0.7)
            plt.xlabel(umap_cols[0]); plt.ylabel(umap_cols[1])
            plt.title('Cells colored by hub_id')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'umap_cells_by_hub.png'), dpi=300)
            plt.close()

def add_interface_strength_columns(cell_df, min_pid=None, max_pid=None, thr_exist=0.05):
    """
     Pk
      IF_Pi_Pj   = min(Pi, Pj)
      COREi_...  = Pi
      COREj_...  = Pj
     >= thr_exist
    cell_df(),  pairs = [(i,j), ...]
    """
    import numpy as np, pandas as pd
    df = cell_df.copy()
    #  K
    pcols = [c for c in df.columns if c.startswith('P') and c[1:].isdigit()]
    if not pcols:
        return df, []
    Ks = sorted({int(c[1:]) for c in pcols})
    if min_pid is None: min_pid = min(Ks)
    if max_pid is None: max_pid = max(Ks)

    pairs = []
    for i in range(min_pid, max_pid+1):
        ci = f'P{i}'
        if ci not in df.columns: continue
        for j in range(i+1, max_pid+1):
            cj = f'P{j}'
            if cj not in df.columns: continue
            # 
            ok = False
            for s, sdf in df.groupby('sample_name'):
                vi, vj = float(sdf[ci].mean()), float(sdf[cj].mean())
                if vi >= thr_exist and vj >= thr_exist:
                    ok = True; break
            if not ok: continue

            pairs.append((i,j))
            if_name = f'IF_P{i}_P{j}'
            df[if_name] = np.minimum(df[ci].values, df[cj].values).astype(np.float32)
            df[f'COREi_P{i}_P{j}'] = df[ci].astype(np.float32)
            df[f'COREj_P{i}_P{j}'] = df[cj].astype(np.float32)
    return df, pairs


def run_interface_DE_all_pairs(cell_df, gene_df, gene_list,
                               out_root, pairs, t_if=0.45, t_core=0.8,
                               min_cells_per_group=60, p_thr=0.05, log2fc_thr=0.10,
                               tag_suffix='', fallback_flag=False,
                               debug=False, debug_label='default',
                               sample_dict=None,
                               balance_delta=None,
                               balance_pairs=None,
                               interface_quantile=None,
                               agg_method='median',
                               agg_subsample=0,
                               agg_winsor=0.0,
                               test_method='wilcoxon',
                               enable_gene_sets=False,
                               gene_sets=None,
                               rng_seed=1337):
    """
     (i,j)  Interface vs Core 
    -  cg  bulk median/winsor/subsample
    -  |Pi-Pj| balance Wilcoxon 
    """
    import os, numpy as np, pandas as pd
    from scipy import stats
    try:
        from statsmodels.stats.multitest import multipletests
    except Exception:
        multipletests = None

    os.makedirs(out_root, exist_ok=True)

    gcols = [c for c in gene_df.columns if c.startswith('gene_expr_')]
    if not gcols:
        print("[IF-DE] no gene_expr_* columns; skip.")
        return []

    # Normalise pair list for balance/quantile application
    balance_pairs_norm = set()
    if balance_pairs:
        for pair in balance_pairs:
            try:
                a, b = pair
                balance_pairs_norm.add(tuple(sorted((int(a), int(b)))))
            except Exception:
                continue

    rng = np.random.default_rng(int(rng_seed))

    # Column  gene name mapping
    col_gene_names = []
    gene_index_map = {}
    for idx, g in enumerate(gcols):
        name = g
        if g.startswith('gene_expr_'):
            try:
                gid = int(g.split('_')[-1])
                if 0 <= gid < len(gene_list):
                    name = gene_list[gid]
            except Exception:
                name = g
        col_gene_names.append(str(name))
        key = str(name).upper()
        gene_index_map.setdefault(key, []).append(idx)

    if enable_gene_sets:
        if gene_sets is None:
            gene_sets = {
                'P4_immune_core': {
                    'genes': ['STAT1', 'IRF7', 'PSMB8', 'HLA-DRA', 'CXCL9', 'CXCL10', 'FCGR3A', 'IFI44L'],
                    'direction': 'core>if',
                },
                'P2_interface_adhesion': {
                    'genes': ['ICAM2', 'ITGAL', 'LTB', 'CCL19', 'CCL21', 'VCAM1', 'SELL'],
                    'direction': 'if>core',
                },
            }
        gene_sets_indices = {}
        for gs_name, cfg in gene_sets.items():
            genes_upper = [str(g).upper() for g in cfg.get('genes', [])]
            idxs = set()
            for gu in genes_upper:
                idxs.update(gene_index_map.get(gu, []))
            if idxs:
                gene_sets_indices[gs_name] = {
                    'indices': sorted(idxs),
                    'direction': cfg.get('direction', 'two-sided'),
                }
        if debug and enable_gene_sets and not gene_sets_indices:
            print("[IF-DE][debug] gene sets enabled but no overlap with available gene_expr_* columns.")
    else:
        gene_sets_indices = {}

    pair_stats = []
    diag_rows = [] if debug else None

    if debug:
        print(f"[IF-DE][debug] pairs={len(pairs)}, min_cells={min_cells_per_group}, t_core={t_core:.2f}, t_if={t_if:.2f}")
        if sample_dict is None:
            print("[IF-DE][debug] sample_dict  cg ")

    def _winsor_clip(arr: np.ndarray) -> np.ndarray:
        if agg_winsor > 0.0 and arr.size:
            lower = np.quantile(arr, agg_winsor, axis=0)
            upper = np.quantile(arr, 1.0 - agg_winsor, axis=0)
            return np.clip(arr, lower, upper)
        return arr

    def _aggregate(arr: np.ndarray) -> np.ndarray:
        arr = _winsor_clip(arr)
        if agg_method == 'median':
            return np.nanmedian(arr, axis=0)
        return np.nanmean(arr, axis=0)

    for (i, j) in pairs:
        tag = f'P{i}_P{j}'
        if_col = f'IF_P{i}_P{j}'
        if if_col not in cell_df.columns:
            continue
        out_tag = f'{tag}{tag_suffix}'
        pair_key = tuple(sorted((int(i), int(j))))

        recs = []
        sample_count = 0
        total_if_cells = 0
        total_core_cells = 0
        gene_set_samples = {name: {'if': [], 'core': []} for name in gene_sets_indices}

        for sname, sdf in cell_df.groupby('sample_name'):
            Pi = pd.to_numeric(sdf.get(f'P{i}', pd.Series(0.0, index=sdf.index)), errors='coerce').fillna(0.0).to_numpy()
            Pj = pd.to_numeric(sdf.get(f'P{j}', pd.Series(0.0, index=sdf.index)), errors='coerce').fillna(0.0).to_numpy()
            if_col_vals = pd.to_numeric(sdf[if_col], errors='coerce').fillna(0.0).to_numpy()

            m_if = (if_col_vals >= t_if)
            m_corei = (Pi >= t_core) & (Pj <= (1.0 - t_if))
            m_corej = (Pj >= t_core) & (Pi <= (1.0 - t_if))

            if balance_delta is not None and pair_key in balance_pairs_norm:
                balance_mask = np.abs(Pi - Pj) <= float(balance_delta)
                m_if &= balance_mask
            if interface_quantile is not None and pair_key in balance_pairs_norm:
                min_pair = np.minimum(Pi, Pj)
                try:
                    q_thr = float(np.quantile(min_pair, interface_quantile))
                    m_if &= (min_pair >= q_thr)
                except Exception:
                    pass

            if m_if.sum() < min_cells_per_group or (m_corei.sum() + m_corej.sum()) < min_cells_per_group:
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'low_cells_raw',
                        'interface_cells': int(m_if.sum()),
                        'core_cells': int((m_corei | m_corej).sum()),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            if sample_dict is None:
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'sample_dict_none',
                        'interface_cells': int(m_if.sum()),
                        'core_cells': int((m_corei | m_corej).sum()),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            data_sample = get_sample_mod(sample_dict, sname)
            if data_sample is None:
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'sample_graph_missing',
                        'interface_cells': int(m_if.sum()),
                        'core_cells': int((m_corei | m_corej).sum()),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            sdf_gene = gene_df[gene_df['sample_name'] == sname]
            if sdf_gene.empty:
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'gene_rows_empty',
                        'interface_cells': int(m_if.sum()),
                        'core_cells': int((m_corei | m_corej).sum()),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            X_hat, valid_mask, used_edges = _cg_weighted_cell_gene_matrix(sdf, sdf_gene, gcols, data_sample)
            if X_hat is None or used_edges == 0:
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'cg_no_edges',
                        'interface_cells': int(m_if.sum()),
                        'core_cells': int((m_corei | m_corej).sum()),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            if 'node_idx' in sdf.columns:
                node_idx_series = pd.to_numeric(sdf['node_idx'], errors='coerce')
            else:
                node_idx_series = pd.Series(np.arange(len(sdf)), index=sdf.index, dtype=float)

            node_idx_vals = node_idx_series.to_numpy()
            valid_node = ~np.isnan(node_idx_vals)
            if not valid_node.any():
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'no_valid_nodes',
                        'interface_cells': int(m_if.sum()),
                        'core_cells': int((m_corei | m_corej).sum()),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            node_idx_int = node_idx_vals[valid_node].astype(int)
            idx_map = {cid: pos for pos, cid in enumerate(node_idx_int)}
            row_indices = np.full(len(sdf), -1, dtype=int)
            for pos, val in enumerate(node_idx_vals):
                if np.isnan(val):
                    continue
                row_indices[pos] = idx_map.get(int(val), -1)

            valid_mask_arr = np.asarray(valid_mask).astype(bool) if valid_mask is not None else np.ones(X_hat.shape[0], dtype=bool)
            row_valid = np.zeros(len(sdf), dtype=bool)
            sel = row_indices >= 0
            if sel.any():
                row_valid[sel] = valid_mask_arr[row_indices[sel]]

            mask_if_valid = m_if & row_valid
            mask_core_valid = (m_corei | m_corej) & row_valid

            if mask_if_valid.sum() < min_cells_per_group or mask_core_valid.sum() < min_cells_per_group:
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'low_cells_after_map',
                        'interface_cells': int(mask_if_valid.sum()),
                        'core_cells': int(mask_core_valid.sum()),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            row_if = row_indices[mask_if_valid]
            row_core = row_indices[mask_core_valid]
            row_if = row_if[row_if >= 0]
            row_core = row_core[row_core >= 0]
            if len(row_if) < min_cells_per_group or len(row_core) < min_cells_per_group:
                if debug:
                    diag_rows.append({
                        'pair': tag,
                        'sample_name': sname,
                        'reason': 'low_cells_after_rows',
                        'interface_cells': int(len(row_if)),
                        'core_cells': int(len(row_core)),
                        't_if': float(t_if),
                        't_core': float(t_core),
                        'min_cells': int(min_cells_per_group),
                        'tag_suffix': tag_suffix or '',
                        'fallback': bool(fallback_flag)
                    })
                continue

            expr_if = X_hat[row_if, :]
            expr_core = X_hat[row_core, :]

            if agg_subsample > 0:
                sample_size = min(agg_subsample, expr_if.shape[0], expr_core.shape[0])
                if sample_size >= 3:
                    if expr_if.shape[0] > sample_size:
                        expr_if = expr_if[rng.choice(expr_if.shape[0], size=sample_size, replace=False)]
                    if expr_core.shape[0] > sample_size:
                        expr_core = expr_core[rng.choice(expr_core.shape[0], size=sample_size, replace=False)]

            agg_if_vec = _aggregate(expr_if)
            agg_core_vec = _aggregate(expr_core)

            rec = {'sample_name': sname}
            for idx_col, g in enumerate(gcols):
                rec[g] = float(agg_if_vec[idx_col])
                rec[f'{g}__core'] = float(agg_core_vec[idx_col])
            recs.append(rec)
            sample_count += 1
            total_if_cells += int(mask_if_valid.sum())
            total_core_cells += int(mask_core_valid.sum())

            for gs_name, cfg in gene_sets_indices.items():
                idxs = cfg['indices']
                if not idxs:
                    continue
                gs_if_vals = agg_if_vec[idxs]
                gs_core_vals = agg_core_vec[idxs]
                if np.all(np.isnan(gs_if_vals)) or np.all(np.isnan(gs_core_vals)):
                    continue
                score_if = float(np.nanmean(gs_if_vals))
                score_core = float(np.nanmean(gs_core_vals))
                gene_set_samples[gs_name]['if'].append(score_if)
                gene_set_samples[gs_name]['core'].append(score_core)

        if not recs:
            continue

        P = pd.DataFrame(recs)
        gene_out = []
        for g in gcols:
            a = P[g].values
            b = P[f'{g}__core'].values
            if len(a) < 3 or len(b) < 3:
                continue
            if test_method == 'wilcoxon':
                try:
                    stat_res = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
                    p_val = float(stat_res.pvalue)
                except ValueError:
                    continue
            else:
                _, p_val = stats.ttest_rel(a, b, nan_policy='omit')
            if not np.isfinite(p_val):
                continue
            l2fc = np.log2((np.nanmean(a) + 1e-9) / (np.nanmean(b) + 1e-9))
            if abs(l2fc) >= log2fc_thr and p_val <= p_thr:
                gene_name = g
                try:
                    gid = int(g.split('_')[-1])
                    if 0 <= gid < len(gene_list):
                        gene_name = gene_list[gid]
                except Exception:
                    gene_name = g.replace('gene_expr_', '')
                gene_out.append((gene_name, l2fc, p_val))

        if not gene_out:
            if debug:
                diag_rows.append({
                    'pair': tag,
                    'sample_name': 'ALL',
                    'reason': 'no_significant_genes',
                    'interface_cells': int(total_if_cells),
                    'core_cells': int(total_core_cells),
                    't_if': float(t_if),
                    't_core': float(t_core),
                    'min_cells': int(min_cells_per_group),
                    'samples_used': int(sample_count),
                    'tag_suffix': tag_suffix or '',
                    'fallback': bool(fallback_flag)
                })
            continue

        df = pd.DataFrame(gene_out, columns=['gene', 'log2fc_IF_vs_CORE', 'p_value']).sort_values('p_value')
        if multipletests is not None:
            df['q_value'] = multipletests(df['p_value'].values, method='fdr_bh')[1]
            df = df.sort_values(['q_value', 'p_value'])
        df['pair'] = tag
        df['fallback'] = bool(fallback_flag)
        out_csv = os.path.join(out_root, f"{out_tag}_interface_DEGs.csv")
        df.to_csv(out_csv, index=False)
        print(f"[IF-DE] {tag}: {len(df)} DE genes -> {out_csv}")
        pair_stats.append({
            'pair': tag,
            'pair_tag': out_tag,
            'fallback': bool(fallback_flag),
            'n_samples_used': int(sample_count),
            'interface_total_cells': int(total_if_cells),
            'core_total_cells': int(total_core_cells)
        })

        if gene_sets_indices and gene_set_samples:
            gs_rows = []
            for gs_name, buf in gene_set_samples.items():
                vals_if = np.array(buf['if'], dtype=float)
                vals_core = np.array(buf['core'], dtype=float)
                mask_valid = (~np.isnan(vals_if)) & (~np.isnan(vals_core))
                vals_if = vals_if[mask_valid]
                vals_core = vals_core[mask_valid]
                if len(vals_if) < 3:
                    continue
                if test_method == 'wilcoxon':
                    try:
                        stat_res = stats.wilcoxon(vals_if, vals_core, alternative='two-sided', zero_method='wilcox')
                        p_val = float(stat_res.pvalue)
                    except ValueError:
                        continue
                else:
                    _, p_val = stats.ttest_rel(vals_if, vals_core, nan_policy='omit')
                gs_rows.append({
                    'pair': tag,
                    'pair_tag': out_tag,
                    'gene_set': gs_name,
                    'direction_hint': gene_sets_indices[gs_name]['direction'],
                    'n_samples': int(len(vals_if)),
                    'score_if_mean': float(np.nanmean(vals_if)),
                    'score_core_mean': float(np.nanmean(vals_core)),
                    'delta_mean': float(np.nanmean(vals_if - vals_core)),
                    'test_method': test_method,
                    'p_value': float(p_val)
                })
            if gs_rows:
                gs_df = pd.DataFrame(gs_rows).sort_values('p_value')
                if multipletests is not None and len(gs_df) > 1:
                    gs_df['q_value'] = multipletests(gs_df['p_value'].values, method='fdr_bh')[1]
                out_gs = os.path.join(out_root, f"{out_tag}_interface_gene_sets.csv")
                gs_df.to_csv(out_gs, index=False)
                if debug:
                    print(f"[IF-DE][debug] gene-set stats saved: {out_gs}")

    if debug and diag_rows:
        diag_df = pd.DataFrame(diag_rows)
        diag_path = os.path.join(out_root, f"interface_DE_diagnostics_{debug_label or 'default'}.csv")
        try:
            diag_df.to_csv(diag_path, index=False)
            print(f"[IF-DE][debug] diagnostics saved: {diag_path}")
        except Exception as e:
            print(f"[IF-DE][debug] failed to write diagnostics: {e}")

    return pair_stats


def _map_gene_token(token, idx_to_gene):
    if isinstance(token, str) and token.startswith('gene_expr_'):
        try:
            gid = int(token.split('_')[-1])
        except ValueError:
            gid = None
    elif str(token).isdigit():
        gid = int(token)
    else:
        gid = None
    if gid is None:
        return str(token)
    return idx_to_gene.get(gid, _canonical_gene_symbol(token))


def _canonical_gene_symbol(token):
    if token is None:
        return ''
    base = str(token).strip()
    if not base:
        return ''
    base = base.split('.')[0]
    base = re.sub(r'[^A-Za-z0-9\-]', '', base)
    return base.upper()


def _load_core_edge_table(path, idx_to_gene):
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'gene' not in df.columns or 'log2fc_core_vs_edge' not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df['gene'] = df['gene'].apply(lambda g: _map_gene_token(g, idx_to_gene))
    df = df.drop_duplicates(subset=['gene']).set_index('gene')
    return df


def _load_pcr_table(path):
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    req = {'gene', 'log2fc_Resp_vs_NonResp'}
    if not req.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy().drop_duplicates(subset=['gene']).set_index('gene')
    return df


def generate_publication_interface_figures(analysis_root,
                                           gene_list,
                                           ref_cohort,
                                           focus_protos=(0, 2, 4),
                                           top_n=25,
                                           cached_core_data=None,
                                           core_threshold=None,
                                           min_cells_per_group=5,
                                           min_samples_core=3,
                                           min_samples_pcr=2):
    """Build publication-ready summaries for interface/core and pCR contrasts.

    Besides reproducing the log2FC-centric publication artefacts, this variant optionally
    computes per-gene core/edge/pCR means so downstream plots can highlight absolute
    expression levels.  The extra statistics are appended to the exported CSV and also
    saved separately when available.
    """
    import glob
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path
    import torch

    idx_to_gene = {i: _canonical_gene_symbol(gene_list[i]) for i in range(min(len(gene_list), 5000))}
    gene_canon_set = {_canonical_gene_symbol(g) for g in gene_list}
    out_dir = os.path.join(analysis_root, "figures_publication")
    os.makedirs(out_dir, exist_ok=True)
    summary = None
    unmapped_records = []
    proto_mean_df = None

    def _load_analysis_meta(root: Path) -> dict:
        meta_path = root / 'analysis_meta.json'
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _resolve_core_threshold(default_thr: float) -> float:
        if core_threshold is not None:
            return float(core_threshold)
        meta = _load_analysis_meta(Path(analysis_root))
        return float(meta.get('core_threshold', default_thr))

    def _read_sample_df(parquet_path: Path, sample_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if not parquet_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(parquet_path, columns=columns, filters=[('sample_name', '==', sample_name)])
        except Exception:
            df = pd.read_parquet(parquet_path, columns=columns)
            if 'sample_name' in df.columns:
                df = df[df['sample_name'] == sample_name]
        return df.copy()

    def _ensure_proto_probs(cell_df: pd.DataFrame,
                            stage3_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        proto_cols = [c for c in cell_df.columns if str(c).startswith('P') and str(c)[1:].isdigit()]
        if proto_cols:
            return cell_df
        if stage3_df is None or stage3_df.empty:
            return cell_df
        stage3_proto = [c for c in stage3_df.columns if str(c).startswith('P') and str(c)[1:].isdigit()]
        if not stage3_proto:
            return cell_df
        join_keys = ['node_idx', 'cell_idx', 'cell_id', 'global_index', 'global_idx']
        join_key = next((k for k in join_keys if k in cell_df.columns and k in stage3_df.columns), None)
        if join_key is None:
            return cell_df
        merged = cell_df.merge(stage3_df[[join_key] + stage3_proto], on=join_key, how='left', suffixes=('', '_st3'))
        for col in stage3_proto:
            fallback = f'{col}_st3'
            if fallback in merged.columns:
                if col not in cell_df.columns:
                    merged[col] = merged[fallback]
                else:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(
                        pd.to_numeric(merged[fallback], errors='coerce'))
                merged.drop(columns=fallback, inplace=True)
        return merged

    def _compute_core_edge_means() -> Optional[pd.DataFrame]:
        cache_dir = Path(analysis_root) / 'cache'
        cell_path = cache_dir / 'stage4_cell_df.parquet'
        gene_path = cache_dir / 'stage4_gene_df.parquet'
        stage3_path = cache_dir / 'stage3_cell_df.parquet'
        sample_dict_path = cache_dir / 'stage2_sample_dict.pt'
        if not (cell_path.exists() and gene_path.exists() and sample_dict_path.exists()):
            return None

        try:
            sample_dict = torch.load(sample_dict_path, map_location='cpu')
        except Exception:
            return None

        sample_names = sorted(sample_dict.keys())
        if not sample_names:
            return None

        stage3_available = stage3_path.exists()
        gene_cols = None
        gene_names = None

        focus_set = sorted({int(pid) for pid in focus_protos})
        proto_stats: Dict[int, Dict[str, List[np.ndarray]]] = {
            pid: {'core': [], 'edge': [], 'resp': [], 'nonresp': []}
            for pid in focus_set
        }

        core_thr = _resolve_core_threshold(0.40)

        # Preload mapping from gene expr column to canonical symbol
        def _lazy_gene_init(gdf: pd.DataFrame):
            nonlocal gene_cols, gene_names
            if gene_cols is not None:
                return
            gene_cols = [c for c in gdf.columns if str(c).startswith('gene_expr_')]
            gene_names = []
            for col in gene_cols:
                try:
                    idx = int(col.split('_')[-1])
                except ValueError:
                    idx = None
                if idx is not None and 0 <= idx < len(gene_list):
                    gene_names.append(_canonical_gene_symbol(gene_list[idx]))
                else:
                    gene_names.append(_canonical_gene_symbol(col))

        for sample_name in sample_names:
            data_sample = sample_dict.get(sample_name)
            if data_sample is None:
                continue

            cell_cols = None
            cell_df = _read_sample_df(cell_path, sample_name, columns=cell_cols)
            if cell_df.empty:
                continue

            gene_df = _read_sample_df(gene_path, sample_name)
            if gene_df.empty:
                continue

            _lazy_gene_init(gene_df)
            if not gene_cols:
                continue

            stage3_df = None
            if stage3_available:
                stage3_df = _read_sample_df(stage3_path, sample_name)

            cell_df = _ensure_proto_probs(cell_df, stage3_df)
            proto_col = 'prototype_id'
            if proto_col not in cell_df.columns:
                proto_col = 'prototype_id_merged' if 'prototype_id_merged' in cell_df.columns else proto_col
            cell_df[proto_col] = pd.to_numeric(cell_df[proto_col], errors='coerce')
            cell_df = cell_df.dropna(subset=[proto_col])
            if cell_df.empty:
                continue

            label_series = pd.to_numeric(cell_df.get('label'), errors='coerce')
            sample_label = int(label_series.mode(dropna=True).iat[0]) if not label_series.dropna().empty else -1

            gdf = gene_df.copy()
            gdf_cols = ['sample_name'] + gene_cols
            if 'node_idx' in gene_df.columns:
                gdf_cols.append('node_idx')
            gdf = gene_df[gdf_cols]

            sdf = cell_df.copy()
            for pid in focus_set:
                prob_col = f'P{pid}'
                if prob_col not in sdf.columns:
                    continue
                probs = pd.to_numeric(sdf[prob_col], errors='coerce').fillna(0.0).to_numpy()
                assigned = (pd.to_numeric(sdf[proto_col], errors='coerce').fillna(-1).astype(int).to_numpy() == int(pid))
                core_mask = (probs >= float(core_thr)) & assigned
                edge_mask = assigned & (~core_mask)
                if core_mask.sum() < max(min_cells_per_group, 5) or edge_mask.sum() < max(min_cells_per_group, 5):
                    continue

                sdf_core = sdf.loc[core_mask]
                sdf_edge = sdf.loc[edge_mask]

                X_core, _, used_core = _cg_weighted_cell_gene_matrix(sdf_core, gdf, gene_cols, data_sample)
                X_edge, _, used_edge = _cg_weighted_cell_gene_matrix(sdf_edge, gdf, gene_cols, data_sample)
                if X_core is None or X_edge is None or used_core == 0 or used_edge == 0:
                    continue

                core_vec = np.nanmean(X_core.astype(np.float64), axis=0)
                edge_vec = np.nanmean(X_edge.astype(np.float64), axis=0)
                if not np.isfinite(core_vec).any() or not np.isfinite(edge_vec).any():
                    continue
                        
                proto_stats[pid]['core'].append(core_vec)
                proto_stats[pid]['edge'].append(edge_vec)

                if sample_label == 1:
                    proto_stats[pid]['resp'].append(core_vec)
                elif sample_label == 0:
                    proto_stats[pid]['nonresp'].append(core_vec)

        if not gene_cols or not any(proto_stats[pid]['core'] for pid in focus_set):
            return None

        eps = 1e-9
        records: Dict[str, Dict[str, float]] = {}
        for pid in focus_set:
            core_sets = proto_stats[pid]['core']
            edge_sets = proto_stats[pid]['edge']
            if not core_sets or not edge_sets:
                continue

            core_arr = np.vstack(core_sets)
            edge_arr = np.vstack(edge_sets)
            core_mean = np.nanmean(core_arr, axis=0)
            edge_mean = np.nanmean(edge_arr, axis=0)

            resp_mean = np.full_like(core_mean, np.nan)
            non_mean = np.full_like(core_mean, np.nan)
            if proto_stats[pid]['resp']:
                resp_mean = np.nanmean(np.vstack(proto_stats[pid]['resp']), axis=0)
            if proto_stats[pid]['nonresp']:
                non_mean = np.nanmean(np.vstack(proto_stats[pid]['nonresp']), axis=0)

            for idx, gene in enumerate(gene_names):
                rec = records.setdefault(gene, {'gene': gene})
                rec[f'P{pid}_core_mean'] = float(core_mean[idx]) if np.isfinite(core_mean[idx]) else np.nan
                rec[f'P{pid}_edge_mean'] = float(edge_mean[idx]) if np.isfinite(edge_mean[idx]) else np.nan
                if np.isfinite(resp_mean[idx]):
                    rec[f'P{pid}_resp_mean'] = float(resp_mean[idx])
                if np.isfinite(non_mean[idx]):
                    rec[f'P{pid}_nonresp_mean'] = float(non_mean[idx])

        if not records:
            return None

        df = pd.DataFrame(records.values())
        df = df.groupby('gene', as_index=False).first()
        df = df.sort_values('gene')
        expected_cols = []
        for pid in focus_set:
            expected_cols.extend([
                f'P{pid}_core_mean',
                f'P{pid}_edge_mean',
                f'P{pid}_resp_mean',
                f'P{pid}_nonresp_mean',
            ])
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df

    proto_mean_df = cached_core_data

    def _safe_heatmap(matrix_df, title, fname, vcenter=0.0, cmap='coolwarm'):
        if matrix_df is None or matrix_df.empty:
            return
        plt.figure(figsize=(6.0, 5.2))
        sns.heatmap(matrix_df, cmap=cmap, center=vcenter, annot=True,
                    fmt=".2f", linewidths=0.5, square=True, cbar_kws={'label': title})
        plt.title(title, fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()

    # 1) Interface heatmaps (overall & responder difference)
    overall_csv = os.path.join(analysis_root, f"{ref_cohort}_only_interface_log2EN.csv")
    if os.path.exists(overall_csv):
        df_all = pd.read_csv(overall_csv, index_col=0)
        df_all = df_all.apply(pd.to_numeric, errors='coerce')
        df_all = df_all.loc[df_all.index.intersection(df_all.columns), df_all.columns.intersection(df_all.index)]
        _safe_heatmap(df_all, "Interface log2 enrichment", "interface_heatmap_overall.png")

    resp_csv = os.path.join(analysis_root, f"COHORT_{ref_cohort}", "responders_interface_log2EN.csv")
    non_csv = os.path.join(analysis_root, f"COHORT_{ref_cohort}", "non_responders_interface_log2EN.csv")
    if os.path.exists(resp_csv) and os.path.exists(non_csv):
        df_r = pd.read_csv(resp_csv, index_col=0).apply(pd.to_numeric, errors='coerce')
        df_n = pd.read_csv(non_csv, index_col=0).apply(pd.to_numeric, errors='coerce')
        common = sorted(set(df_r.index) & set(df_r.columns) & set(df_n.index) & set(df_n.columns))
        if common:
            diff = df_r.loc[common, common] - df_n.loc[common, common]
            vmax = np.nanmax(np.abs(diff.values)) if np.isfinite(diff.values).any() else 0.0
            vmax = max(vmax, 0.2)
            plt.figure(figsize=(6.0, 5.2))
            sns.heatmap(diff, cmap='coolwarm', center=0.0, linewidths=0.5,
                        vmax=vmax, vmin=-vmax, cbar_kws={'label': 'Responder - Non-responder'})
            plt.title('Interface  log2EN (responders - non)', fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'interface_heatmap_delta_pcr.png'), dpi=300)
            plt.close()

    # 2) Aggregate core-edge tables for selected prototypes
    proto_tables = {}
    for pid in focus_protos:
        pattern = os.path.join(analysis_root, "**", f"DE_core_vs_edge_P{int(pid)}.csv")
        tables = []
        for path in glob.glob(pattern, recursive=True):
            df = _load_core_edge_table(path, idx_to_gene)
            if df.empty:
                continue
            rename_map = {
                'log2fc_core_vs_edge': f'P{int(pid)}_core_vs_edge',
                'p_value': f'P{int(pid)}_core_vs_edge_pvalue',
                'core_mean': f'P{int(pid)}_core_mean',
                'edge_mean': f'P{int(pid)}_edge_mean',
                'core_sample_count': f'P{int(pid)}_core_sample_count',
                'edge_sample_count': f'P{int(pid)}_edge_sample_count',
            }
            keep_cols = [col for col in rename_map if col in df.columns]
            if not keep_cols:
                continue
            tables.append(df[keep_cols].rename(columns=rename_map))
        if tables:
            proto_tables[pid] = pd.concat(tables, axis=0).groupby(level=0).median()
        else:
            proto_tables[pid] = pd.DataFrame(columns=[f'P{int(pid)}_core_vs_edge'])

    if proto_tables:
        for pid, df in proto_tables.items():
            if summary is None:
                summary = df
            else:
                summary = summary.join(df, how='outer')
        if summary is not None and not summary.empty:
            for pid in focus_protos:
                col = f'P{int(pid)}_core_vs_edge'
                if col in summary.columns:
                    summary[f'P{int(pid)}_edge_vs_core'] = -summary[col]

    # 3) pCR differential tables (per prototype)
    pcr_tables = {}
    for pid in focus_protos:
        pattern = os.path.join(analysis_root, "**", "4_gene_functional_analysis", "1_intra_prototype_DE_by_phenotype", f"P{int(pid)}_DEGs.csv")
        rows = []
        for path in glob.glob(pattern, recursive=True):
            df = _load_pcr_table(path)
            if df.empty:
                continue
            rename_map = {
                'log2fc_Resp_vs_NonResp': f'P{int(pid)}_pCR_log2fc',
                'p_value': f'P{int(pid)}_pCR_pvalue',
                'resp_mean': f'P{int(pid)}_resp_mean',
                'nonresp_mean': f'P{int(pid)}_nonresp_mean',
                'resp_sample_count': f'P{int(pid)}_resp_sample_count',
                'nonresp_sample_count': f'P{int(pid)}_nonresp_sample_count',
            }
            keep_cols = [col for col in rename_map if col in df.columns]
            if not keep_cols:
                continue
            rows.append(df[keep_cols].rename(columns=rename_map))
        if rows:
            merged = pd.concat(rows, axis=0).groupby(level=0).median()
        else:
            merged = pd.DataFrame()
        pcr_tables[pid] = merged

    if proto_tables:
        for pid, df in pcr_tables.items():
            if summary is None:
                summary = df
            else:
                summary = summary.join(df, how='outer')

    if summary is not None and not summary.empty:
        cols_for_ranking = [c for c in summary.columns if c.endswith('_edge_vs_core') or c.endswith('_pCR_log2fc')]
        summary_rank = summary[cols_for_ranking].abs().max(axis=1).sort_values(ascending=False)
        keep = summary_rank.head(top_n).index
        plot_df = summary.loc[keep, cols_for_ranking].fillna(0.0)
        plot_df = plot_df.sort_values(cols_for_ranking, ascending=False)

        plt.figure(figsize=(max(6.4, 0.32 * len(cols_for_ranking) + 3), 0.35 * len(keep) + 2))
        sns.heatmap(plot_df, cmap='coolwarm', center=0.0, linewidths=0.4,
                    cbar_kws={'label': 'log2 fold-change'})
        plt.title('Interface/core & pCR signatures (median across cohorts)', fontsize=13)
        plt.xlabel('Contrast')
        plt.ylabel('Gene')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'gene_signature_heatmap.png'), dpi=300)
        plt.close()

        summary_out = os.path.join(out_dir, 'interface_core_signature_summary.csv')
        summary_export = summary.copy()
        summary_export.sort_index().to_csv(summary_out)

        mean_cols = [c for c in summary_export.columns
                     if c.endswith('_core_mean') or c.endswith('_edge_mean')
                     or c.endswith('_resp_mean') or c.endswith('_nonresp_mean')
                     or c.endswith('_core_vs_edge_log2fc') or c.endswith('_core_vs_edge_pvalue')
                     or c.endswith('_core_vs_edge_qvalue') or c.endswith('_pCR_log2fc')
                     or c.endswith('_pCR_pvalue') or c.endswith('_pCR_qvalue')
                     or '_pair_' in c]
        if mean_cols:
            means_df = summary_export.reset_index().rename(columns={'index': 'gene'})
            keep_cols = ['gene'] + mean_cols
            means_df = means_df[keep_cols]
            means_out = os.path.join(out_dir, 'interface_core_signature_means.csv')
            means_df.to_csv(means_out, index=False)

        long_rows = []
        for col in cols_for_ranking:
            if col not in summary.columns:
                continue
            for gene, val in summary[col].dropna().items():
                direction = 'edge>core' if ('edge_vs_core' in col and val > 0) else (
                    'core>edge' if ('edge_vs_core' in col and val < 0) else (
                    'pCR>non' if ('pCR' in col and val > 0) else (
                    'non>pCR' if ('pCR' in col and val < 0) else 'NA')))
                long_rows.append({'gene': gene, 'contrast': col, 'log2fc': float(val), 'direction': direction})
        if long_rows:
            long_df = pd.DataFrame(long_rows)
            long_df = long_df.sort_values(['contrast','direction','log2fc'], ascending=[True, True, False])
            long_df.to_csv(os.path.join(out_dir, 'interface_core_signature_long.csv'), index=False)

        for pid in focus_protos:
            col = f'P{int(pid)}_edge_vs_core'
            if col not in summary.columns:
                continue
            col_df = summary[[col]].dropna()
            if col_df.empty:
                continue
            col_df = col_df.sort_values(col, ascending=False)
            out_prefix = os.path.join(out_dir, f'P{int(pid)}_edge_vs_core')
            top_edge = col_df[col_df[col] > 0].head(min(25, len(col_df)))
            if not top_edge.empty:
                top_edge.assign(direction='edge>core').to_csv(f'{out_prefix}_edge_high.csv')
            top_core = col_df[col_df[col] < 0].tail(min(25, len(col_df)))
            if not top_core.empty:
                top_core.assign(direction='core>edge').sort_values(col, ascending=True).to_csv(f'{out_prefix}_core_high.csv')

        # scatter: P2 vs P4 interface enrichment
        col_a = f'P{focus_protos[0]}_edge_vs_core'
        col_b = f'P{focus_protos[1]}_edge_vs_core'
        if col_a in summary.columns and col_b in summary.columns:
            scatter_df = summary[[col_a, col_b]].dropna()
            if not scatter_df.empty:
                plt.figure(figsize=(6.0, 6.0))
                sns.scatterplot(x=col_a, y=col_b, data=scatter_df, s=40, edgecolor='k')
                lim = np.max(np.abs(scatter_df.values))
                lim = max(lim, 0.2)
                plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
                plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                plt.plot([-lim, lim], [-lim, lim], color='grey', linestyle=':', linewidth=0.8)
                plt.xlim(-lim, lim)
                plt.ylim(-lim, lim)
                labels = scatter_df.abs().sum(axis=1).sort_values(ascending=False).head(min(10, len(scatter_df)))
                for gene in labels.index:
                    x, y = scatter_df.loc[gene, col_a], scatter_df.loc[gene, col_b]
                    plt.text(x, y, gene, fontsize=8, ha='left', va='center')
                plt.xlabel(f'P{focus_protos[0]} interface vs core (log2)')
                plt.ylabel(f'P{focus_protos[1]} interface vs core (log2)')
                plt.title('Interface enrichment comparison')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'interface_scatter_P{}_P{}.png'.format(focus_protos[0], focus_protos[1])), dpi=300)
                plt.close()

    # 4) Interface DE (IF vs CORE) aggregated signatures
    interface_files = glob.glob(os.path.join(analysis_root, '**', '*_interface_DEGs.csv'), recursive=True)
    interface_records = []
    for path in interface_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or 'log2fc_IF_vs_CORE' not in df.columns:
            continue
        base = os.path.basename(path)
        import re
        m = re.search(r'(P\d+_P\d+)', base)
        pair_tok = m.group(1) if m else base.replace('_interface_DEGs.csv','')
        df = df.copy()
        mapped_genes = df['gene'].apply(lambda g: _map_gene_token(g, idx_to_gene))
        mask_unmapped = ~mapped_genes.isin(gene_canon_set)
        if mask_unmapped.any():
            unmapped_records.extend([
                {'pair': pair_tok, 'raw_gene': df['gene'].iloc[idx], 'mapped': mapped_genes.iloc[idx]}
                for idx in np.where(mask_unmapped.to_numpy())[0]
            ])
        df['gene'] = mapped_genes
        df['pair'] = pair_tok
        if 'q_value' not in df.columns:
            df['q_value'] = np.nan
        cols_keep = ['pair','gene','log2fc_IF_vs_CORE','p_value','q_value']
        if 'fallback' in df.columns:
            cols_keep.append('fallback')
        interface_records.append(df[cols_keep])

    if interface_records:
        IF = pd.concat(interface_records, axis=0, ignore_index=True)
        IF = IF.dropna(subset=['gene']).drop_duplicates()
        if 'fallback' in IF.columns:
            IF['fallback'] = IF['fallback'].astype(bool)
        else:
            IF['fallback'] = False

        def _pivot_interface(df):
            if df.empty:
                return pd.DataFrame()
            return (df.groupby(['pair','gene'])['log2fc_IF_vs_CORE'].median()
                      .unstack('pair').sort_index())

        IF_main = IF[~IF['fallback']]
        IF_fb = IF[IF['fallback']]
        IF_summary_main = _pivot_interface(IF_main)
        IF_summary_fb = _pivot_interface(IF_fb)
        if not IF_summary_main.empty:
            IF_summary = IF_summary_main
            if not IF_summary_fb.empty:
                IF_summary = IF_summary.combine_first(IF_summary_fb)
        else:
            IF_summary = IF_summary_fb
        IF_summary.to_csv(os.path.join(out_dir, 'interface_deg_summary.csv'))

        IF_long = IF.sort_values(['pair','fallback','log2fc_IF_vs_CORE'], ascending=[True, True, False])
        IF_long.to_csv(os.path.join(out_dir, 'interface_deg_long.csv'), index=False)

        if not IF_summary.empty:
            gene_scores = IF_summary.abs().max(axis=1).sort_values(ascending=False)
            keep = gene_scores.head(min(top_n, len(gene_scores))).index
            heat_df = IF_summary.loc[keep]
            plt.figure(figsize=(max(6.0, 0.35 * heat_df.shape[1] + 3), 0.35 * heat_df.shape[0] + 2))
            sns.heatmap(heat_df, cmap='coolwarm', center=0.0, linewidths=0.4,
                        cbar_kws={'label': 'log2FC (IF vs CORE)'})
            plt.title('Interface DE signatures (IF vs CORE)', fontsize=13)
            plt.xlabel('Interface pair')
            plt.ylabel('Gene')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'interface_deg_heatmap.png'), dpi=300)
            plt.close()

        for pair in sorted(IF['pair'].unique()):
            sub_all = IF[IF['pair']==pair]
            if sub_all.empty:
                continue
            sub = sub_all[~sub_all['fallback']]
            if sub.empty:
                sub = sub_all
            pos = sub[sub['log2fc_IF_vs_CORE']>0].sort_values('log2fc_IF_vs_CORE', ascending=False).head(min(25, len(sub)))
            neg = sub[sub['log2fc_IF_vs_CORE']<0].sort_values('log2fc_IF_vs_CORE').head(min(25, len(sub)))
            if not pos.empty:
                pos.to_csv(os.path.join(out_dir, f'{pair}_interface_high.csv'), index=False)
            if not neg.empty:
                neg.to_csv(os.path.join(out_dir, f'{pair}_interface_low.csv'), index=False)

    if unmapped_records:
        pd.DataFrame(unmapped_records).to_csv(
            os.path.join(out_dir, 'interface_deg_unmapped.csv'),
            index=False
        )


def overlay_all_interfaces_on_wsi(cell_df, pairs, svs_root_map, cellgeojson_root_map,
                                  out_root, target_level=1, k_vote=3,
                                  alpha_outline=80):
    import os
    os.makedirs(out_root, exist_ok=True)
    for sname, sdf in cell_df.groupby('sample_name'):
        cohort_raw = sdf['cohort'].iat[0] if 'cohort' in sdf.columns else None
        cohort_label = str(cohort_raw).strip() if cohort_raw is not None else "NA"
        svs_root = _resolve_cohort_path(svs_root_map, cohort_raw)
        gj_root  = _resolve_cohort_path(cellgeojson_root_map, cohort_raw)
        for (i,j) in pairs:
            col = f'IF_P{i}_P{j}'
            if col not in cell_df.columns: 
                continue
            out_png = os.path.join(out_root, cohort_label, f"{sname}__IF_P{i}_P{j}.png")
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            try:
                overlay_wsi_gate_for_sample_mod(
                    sample_name=sname, cell_df=cell_df,
                    svs_root=svs_root, cellgeojson_root=gj_root,
                    out_png=out_png,
                    scalar_col=col,
                    target_level=target_level, alpha_outline=alpha_outline, k_vote=k_vote
                )
            except Exception as ee:
                print(f"[WARN] overlay IF_P{i}_P{j} failed on {sname}: {ee}")




def plot_interface_contours(cell_df, pairs, out_dir,
                            t_if=0.45,
                            balance_delta=None,
                            balance_pairs=None,
                            interface_quantile=None,
                            min_cells=30,
                            grid_size=200,
                            line_width=2.0,
                            add_legend=False,
                            legend_loc='upper right',
                            legend_title='Interface pairs',
                            debug=False):
    """Render pos_x/pos_y heatmaps with contour overlays for interface masks."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd
    import seaborn as sns

    required = {'pos_x', 'pos_y'}
    if not required.issubset(cell_df.columns):
        print("[IF-CTR] pos_x/pos_y missing; skip interface contour plotting.")
        return

    os.makedirs(out_dir, exist_ok=True)

    balance_pairs_norm = set()
    if balance_pairs:
        for pair in balance_pairs:
            try:
                a, b = pair
                balance_pairs_norm.add(tuple(sorted((int(a), int(b)))))
            except Exception:
                continue

    if pairs:
        palette = sns.color_palette('colorblind', n_colors=max(3, len(pairs)))
    else:
        palette = [(0.2, 0.4, 0.8)]

    for sample_name, sdf in cell_df.groupby('sample_name'):
        sdf = sdf.dropna(subset=['pos_x', 'pos_y']).copy()
        if sdf.empty:
            continue

        x = sdf['pos_x'].to_numpy(np.float32)
        y = sdf['pos_y'].to_numpy(np.float32)
        if len(x) == 0:
            continue

        xs = np.linspace(x.min(), x.max(), grid_size + 1)
        ys = np.linspace(y.min(), y.max(), grid_size + 1)
        x_centers = 0.5 * (xs[:-1] + xs[1:])
        y_centers = 0.5 * (ys[:-1] + ys[1:])
        Xc, Yc = np.meshgrid(x_centers, y_centers)

        fig, ax = plt.subplots(figsize=(7.0, 6.2))
        base_drawn = False

        legend_entries = []

        for idx, (i, j) in enumerate(pairs):
            col = f'IF_P{i}_P{j}'
            if col not in sdf.columns:
                continue

            Pi = pd.to_numeric(sdf.get(f'P{i}', 0.0), errors='coerce').fillna(0.0).to_numpy()
            Pj = pd.to_numeric(sdf.get(f'P{j}', 0.0), errors='coerce').fillna(0.0).to_numpy()
            score = np.minimum(Pi, Pj)
            pair_key = tuple(sorted((int(i), int(j))))

            thresh_eff = float(t_if)
            if interface_quantile is not None and pair_key in balance_pairs_norm:
                finite_scores = score[np.isfinite(score)]
                if finite_scores.size:
                    try:
                        q_thr = float(np.quantile(finite_scores, interface_quantile))
                        thresh_eff = max(thresh_eff, q_thr)
                    except Exception:
                        pass

            mask = score >= thresh_eff
            if balance_delta is not None and pair_key in balance_pairs_norm:
                mask &= (np.abs(Pi - Pj) <= float(balance_delta))

            if mask.sum() < max(int(min_cells), 3):
                continue

            # Grid statistics
            score_sum, _, _ = np.histogram2d(x, y, bins=[xs, ys], weights=score)
            count, _, _ = np.histogram2d(x, y, bins=[xs, ys])
            avg_score = np.divide(score_sum, count, out=np.zeros_like(score_sum), where=count > 0)

            mask_sum, _, _ = np.histogram2d(x, y, bins=[xs, ys], weights=mask.astype(float))
            mask_frac = np.divide(mask_sum, count, out=np.zeros_like(mask_sum), where=count > 0)

            # Draw base heatmap once
            if not base_drawn:
                im = ax.imshow(avg_score.T, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]],
                               cmap='inferno', alpha=0.85)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='min(Pi, Pj)')
                base_drawn = True

            color = palette[idx % len(palette)]
            try:
                cs = ax.contour(Xc, Yc, mask_frac.T, levels=[0.5], colors=[color], linewidths=float(line_width))
            except Exception:
                continue
            if add_legend:
                legend_entries.append(Line2D([0], [0], color=color, linewidth=float(line_width), label=f'P{i}-P{j}'))

        if not base_drawn:
            plt.close(fig)
            continue

        ax.set_title(f'{sample_name}  Interface contours')
        ax.set_xlabel('pos_x')
        ax.set_ylabel('pos_y')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        if add_legend and legend_entries:
            try:
                ax.legend(handles=legend_entries,
                          loc=legend_loc,
                          title=legend_title,
                          fontsize=8,
                          title_fontsize=9,
                          framealpha=0.85)
            except Exception:
                pass

        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{sample_name}_interface_contours.png')
        plt.savefig(out_path, dpi=220)
        plt.close(fig)
        if debug:
            print(f"[IF-CTR] saved {out_path}")

def plot_sci_visuals(SCI, out_dir, title="SCI (prototypeprototype)", vmax=None):
    """
    Robust visualizer for SCI matrix.
    - Accepts pandas DataFrame (preferred) or numpy array.
    - Derives labels from DataFrame index/columns when available.
    - Guarantees labels length == M.shape[0] == M.shape[1].
    - Saves a heatmap and an edge list (for).
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ---- 1)  (M, labels) ----
    if isinstance(SCI, pd.DataFrame):
        df = SCI.copy()
        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            raise ValueError("[SCI] empty DataFrame provided")
        if n_rows != n_cols:
            n = min(n_rows, n_cols)
            df = df.iloc[:n, :n]
            n_rows = n_cols = n

        # --- / ---
        col_labels = [str(c) for c in df.columns]
        row_labels = [str(r) for r in df.index]
        seq_rows = [str(i) for i in range(n_rows)]

        labels = []
        if row_labels == col_labels and len(set(col_labels)) == n_cols:
            labels = col_labels
        elif row_labels == seq_rows and len(col_labels) == n_cols:
            labels = col_labels
        else:
            import re
            def _norm_label(val: str) -> str:
                s = str(val).strip()
                match = re.search(r"-?\d+", s)
                return match.group(0) if match else s

            row_norm = [_norm_label(r) for r in row_labels]
            col_norm = [_norm_label(c) for c in col_labels]
            row_pos = {key: idx for idx, key in enumerate(row_norm) if key not in row_norm[:idx]}

            if set(col_norm).issubset(set(row_norm)) and len(set(col_labels)) == n_cols:
                order = [row_pos.get(key) for key in col_norm]
                if any(pos is None for pos in order):
                    labels = col_labels if len(set(col_labels)) == n_cols else [f"P{i}" for i in range(n_cols)]
                else:
                    df = df.take(order, axis=0)
                    labels = col_labels
            else:
                labels = col_labels if len(set(col_labels)) == n_cols else [f"P{i}" for i in range(n_cols)]

        M = df.to_numpy(dtype=np.float32)
    else:
        M = np.asarray(SCI, dtype=np.float32)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"[SCI] expect square matrix, got shape={M.shape}")
        n = M.shape[0]
        labels = [f"P{i}" for i in range(n)]

    n = M.shape[0]
    assert len(labels) == n, "[SCI] labels length must match matrix size"

    # /
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = 0.5 * (M + M.T)

    # ---- 2) Heatmap ----
    if vmax is None:
        vmax = float(np.percentile(np.abs(M), 98)) if np.any(M) else 1.0
        vmax = max(vmax, 1.0)

    H = max(6, 0.35 * n + 2)
    W = max(6, 0.35 * n + 2)
    plt.figure(figsize=(W, H))
    ax = plt.gca()
    im = ax.imshow(M, vmin=-vmax, vmax=+vmax, cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=0, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=14, pad=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label("SCI (symmetrized)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "SCI_heatmap.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "SCI_heatmap.svg"))
    plt.close()

    # ---- 3) Edge list----
    flat = []
    for i in range(n):
        for j in range(i+1, n):
            flat.append((labels[i], labels[j], float(M[i, j])))
    edge_df = pd.DataFrame(flat, columns=["src", "dst", "weight"])
    edge_df.to_csv(os.path.join(out_dir, "SCI_edges.csv"), index=False)

    print(f"[SCI] visuals saved to: {out_dir} (heatmap + edges)")


def overlay_wsi_hubs_for_sample(
    sample_name,
    cell_df,
    svs_root,
    cellgeojson_root,
    out_png,
    target_level=1,
    k_vote=3,
    alpha_fill=120,
    alpha_outline=80,
    draw_outline=True
):
    """
     WSI  hub_id /
    _find_svs, _find_geojson, load_cell_polygons, polygon_centroids, _majority_vote_knn_classes
    """
    import os, numpy as np
    from pathlib import Path

    if (OpenSlide is None) or (Image is None):
        print("[WSI-HUB] OpenSlide/PIL not available; skip."); return

    svs_path = _find_svs(svs_root, sample_name)
    gj_path  = _find_geojson(cellgeojson_root, sample_name)
    if not (svs_path and os.path.exists(svs_path) and gj_path and os.path.exists(gj_path)):
        print(f"[WSI-HUB] skip {sample_name}: missing svs/geojson (svs_root={svs_root}, geojson_root={cellgeojson_root})")
        return

    sdf = cell_df[(cell_df['sample_name'] == sample_name)].dropna(subset=['pos_x', 'pos_y']).copy()
    if sdf.empty or 'hub_id' not in sdf.columns:
        print(f"[WSI-HUB] {sample_name}: missing hub_id")
        return
    sdf['hub_id'] = pd.to_numeric(sdf['hub_id'], errors='coerce').astype('Int64')
    sdf = sdf.dropna(subset=['hub_id'])
    if sdf.empty:
        print(f"[WSI-HUB] {sample_name}: hub_id all NA")
        return

    hub_ids = sorted(sdf['hub_id'].dropna().unique().astype(int).tolist())
    if not hub_ids:
        print(f"[WSI-HUB] {sample_name}: no hubs to plot")
        return

    cell_xy_lvl0 = sdf[['pos_x', 'pos_y']].to_numpy(np.float32)
    slide = OpenSlide(str(svs_path))
    polys_lvl0 = load_cell_polygons(gj_path, slide=slide)
    slide.close()
    if not polys_lvl0:
        print(f"[WSI-HUB] {sample_name}: polygons empty.")
        return

    base_path = Path(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    seq_cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'YlOrRd', 'YlGn', 'PuBu', 'BuPu', 'GnBu', 'OrRd']
    outputs = []

    for idx, hub in enumerate(hub_ids):
        cell_scalar = (sdf['hub_id'].astype(int).to_numpy() == int(hub)).astype(np.float32)
        scores, conf = _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, cell_scalar, k=k_vote)
        poly_scores = np.clip(scores * np.power(np.clip(conf, 0.0, 1.0), 0.5), 0, 1)
        if not np.any(poly_scores > 1e-6):
            continue
        cmap_name = seq_cmaps[idx % len(seq_cmaps)]
        out_hub = base_path.with_name(f"{base_path.stem}_hub{hub}.png")
        overlay_mask_heatmap_on_wsi_mod(
            svs_path=svs_path,
            polys_lvl0=polys_lvl0,
            poly_scores01=poly_scores,
            out_png=str(out_hub),
            target_level=target_level,
            blend='sum',
            sigma_lvl0=32.0,
            cmap_name=cmap_name,
            cmap_use_full=True,
            alpha_gamma=0.5,
            alpha_max=240,
            alpha_floor=150,
            brighten_bg=0.08,
            draw_all_outlines=False,
            low_clip_q=8.0,
            high_clip_q=99.2,
            colorbar_h_ratio=0.45,
            colorbar_w_px=90,
            colorbar_pad_px=28,
            colorbar_title=f"hub {hub}",
            fallback_disk_px=4,
            grow_cover_px=4,
            post_smooth_frac=0.0,
            alpha_smooth_frac=0.0,
        )
        outputs.append((hub, poly_scores.sum(), out_hub))

    if not outputs:
        print(f"[WSI-HUB] {sample_name}: no hub heatmaps generated.")
        return

    dominant = max(outputs, key=lambda x: x[1])
    try:
        shutil.copyfile(dominant[2], base_path)
    except Exception as e:
        print(f"[WARN] failed to write legacy hub overlay: {e}")
    print(f"[WSI-HUB] saved: {[str(o[2]) for o in outputs]}")


def analyze_archetypes_no_umap(df, centers_tensor, entity_type='cell'):
    """
     UMAP prototype_id / prototype_prob
    
      1)  z_* emb_* prototype_id
      2)  cos  W_shared argmax
    centers_tensor = (K, D)
    """
    import numpy as np, pandas as pd
    df = df.copy()

    # 1)  embedding 
    cand = [c for c in df.columns if c.startswith('z_')]
    if not cand:
        cand = [c for c in df.columns if c.startswith('emb_') or c.startswith('feat_')]
    # 2)  prototype_id
    if not cand:
        if 'prototype_id' in df.columns:
            if 'prototype_prob' not in df.columns:
                df['prototype_prob'] = 1.0
            return df
        #  0 
        df['prototype_id'] = 0
        df['prototype_prob'] = 1.0
        return df

    X = df[cand].to_numpy(np.float32)
    W = np.asarray(centers_tensor, dtype=np.float32)
    #  cos
    def _l2n(a, eps=1e-9): 
        n = np.linalg.norm(a, axis=1, keepdims=True); n[n<eps]=1.0; 
        return a/n
    Xn, Wn = _l2n(X), _l2n(W)
    S = Xn @ Wn.T                                     # [N, K]
    pid = np.argmax(S, axis=1)
    pmax = S[np.arange(S.shape[0]), pid]
    df['prototype_id'] = pid.astype(int)
    df['prototype_prob'] = pmax.astype(np.float32)
    # 
    if 'cohort' not in df.columns: df['cohort'] = 'NA'
    if 'sample_name' not in df.columns: df['sample_name'] = 'UNKNOWN'
    return df



# Spatial metrics helpers moved to archetype_analysis.spatial_metrics

# ======================================================================
# __main__ FIX  minimal-invasive edits to your pipeline main sequence
# ======================================================================
if __name__ == '__main__':
    try:
        INTERFACE_DEBUG = False
        parser = argparse.ArgumentParser(description='Archetype spatial analysis')
        parser.add_argument('--cohort-config', type=Path, default=None,
                            help='JSON file describing cohort paths/labels/model/output settings.')
        parser.add_argument('--model-path', type=str, default=None,
                            help='Override path to trained MHGL-ST checkpoint (takes precedence over config/env).')
        parser.add_argument('--output-dir', type=str, default=None,
                            help='Override base output directory (takes precedence over config/env).')
        parser.add_argument('--cohort-allowlist', type=str, default=None,
                            help='Comma-separated list of cohort names to analyze (overrides config/env).')
        parser.add_argument('--core-thr', type=float, default=0.4,
                            help='Threshold for defining prototype core fraction (default: 0.40)')
        parser.add_argument('--excl-thr', type=float, default=0.12,
                            help='Exclusivity margin for core definition (default: 0.12)')
        parser.add_argument('--interface-thr', type=float, default=0.40,
                            help='Threshold for interface occupancy metrics (default: 0.40)')
        parser.add_argument('--pair-exist-thr', type=float, default=0.05,
                            help='Per-sample mean occupancy required for a prototype pair to be tested (default: 0.05)')
        parser.add_argument('--interface-min-cells', type=int, default=60,
                            help='Minimum cells per group (interface/core) per sample to keep a DE contrast (default: 60)')
        parser.add_argument('--if-balance-delta', type=float, default=0.10,
                            help='Maximum |Pi-Pj| allowed for interface cells (set <=0 to disable, default: 0.10)')
        parser.add_argument('--if-balance-pairs', type=str, default='2-4',
                            help='Comma separated prototype pairs (e.g. 2-4,0-2) to apply balance/quantile rules to (default: 2-4)')
        parser.add_argument('--interface-quantile', type=float, default=0.0,
                            help='If >0, per-sample quantile on min(Pi,Pj) used in addition to interface-thr (default: 0.0=off)')
        parser.add_argument('--agg-method', type=str, default='median', choices=['mean','median'],
                            help='Aggregation method for pseudo-bulk per sample (default: median)')
        parser.add_argument('--agg-subsample-size', type=int, default=0,
                            help='If >0, subsample this many cells per group before aggregation (default: 0, no subsample)')
        parser.add_argument('--agg-winsor-quantile', type=float, default=0.0,
                            help='If >0, winsorize cell-level expression at this two-sided quantile before aggregation (default: 0.0)')
        parser.add_argument('--test-method', type=str, default='wilcoxon', choices=['ttest','wilcoxon'],
                            help='Paired statistical test for interface vs core comparison (default: wilcoxon)')
        parser.add_argument('--enable-gene-sets', action='store_true',
                            help='Run paired gene-set scoring for curated interface signatures')
        parser.add_argument('--plot-interface-contours', action='store_true',
                            help='Generate 2D heatmap + contour overlays for interface masks')
        parser.add_argument('--top-n-figure', type=int, default=30,
                            help='Top N genes to display in publication heatmaps (default: 30)')
        parser.add_argument('--interface-debug', action='store_true',
                            help='Enable verbose diagnostics for interface DE filtering')
        parser.add_argument('--cell-feature-template', type=Path, default=None,
                            help='Representative CSV used to reconstruct cell feature names (optional).')
        parser.add_argument('--gene-id-path', type=Path, default=None,
                            help='Text/CSV file listing gene IDs in expression order (optional).')
        cli_args, unknown_args = parser.parse_known_args()
        if unknown_args:
            print(f"[WARN] Unknown CLI arguments ignored: {unknown_args}")
        COHORT_SPECS, cfg_model_path, cfg_output_dir, cfg_allowlist = load_cohort_specs(
            cli_args.cohort_config.expanduser() if getattr(cli_args, 'cohort_config', None) else None
        )
        if cli_args.model_path:
            MODEL_PATH = str(Path(cli_args.model_path).expanduser())
        elif cfg_model_path:
            MODEL_PATH = str(Path(cfg_model_path).expanduser())
        if cli_args.output_dir:
            ANALYSIS_OUTPUT_DIR = str(Path(cli_args.output_dir).expanduser())
        elif cfg_output_dir:
            ANALYSIS_OUTPUT_DIR = str(Path(cfg_output_dir).expanduser())
        os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
        if cli_args.cohort_allowlist:
            COHORT_ALLOWLIST = {c.strip() for c in cli_args.cohort_allowlist.split(',') if c.strip()}
        elif cfg_allowlist is not None:
            COHORT_ALLOWLIST = set(cfg_allowlist)
        CORE_THRESHOLD = float(cli_args.core_thr)
        EXCL_THRESHOLD = float(cli_args.excl_thr)
        INTERFACE_THRESHOLD = float(cli_args.interface_thr)
        PAIR_EXIST_THRESHOLD = float(cli_args.pair_exist_thr)
        INTERFACE_MINCELLS = int(cli_args.interface_min_cells)
        INTERFACE_DEBUG = bool(cli_args.interface_debug)
        IF_BALANCE_DELTA = float(cli_args.if_balance_delta)
        if IF_BALANCE_DELTA <= 0:
            IF_BALANCE_DELTA = None
        IF_BALANCE_PAIRS_RAW = cli_args.if_balance_pairs.strip()
        IF_BALANCE_PAIRS = set()
        if IF_BALANCE_PAIRS_RAW:
            for token in IF_BALANCE_PAIRS_RAW.split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    a, b = token.split('-')
                    IF_BALANCE_PAIRS.add(tuple(sorted((int(a), int(b)))))
                except Exception:
                    print(f"[WARN] could not parse if-balance pair token '{token}', expected format like 2-4")
        INTERFACE_QUANTILE = float(cli_args.interface_quantile)
        if INTERFACE_QUANTILE <= 0 or INTERFACE_QUANTILE >= 1:
            INTERFACE_QUANTILE = None
        AGG_METHOD = str(cli_args.agg_method)
        AGG_SUBSAMPLE = max(0, int(cli_args.agg_subsample_size))
        AGG_WINSOR_Q = max(0.0, float(cli_args.agg_winsor_quantile))
        TEST_METHOD = str(cli_args.test_method)
        ENABLE_GENE_SETS = bool(cli_args.enable_gene_sets)
        PLOT_INTERFACE_CONTOURS = bool(cli_args.plot_interface_contours)
        publication_top_n = int(cli_args.top_n_figure)
        cell_feature_template = Path(cli_args.cell_feature_template).expanduser() if cli_args.cell_feature_template else None
        gene_id_path = Path(cli_args.gene_id_path).expanduser() if cli_args.gene_id_path else None

        # ===========================
        # Configs
        # ===========================
        FAST_SKIP_UMAP = True  #  speed-up; set False for paper figures

        # ========== Stage 1: Load ==========
        model = initialize_model(
            CELL_IN_CHANNELS,
            GENE_IN_CHANNELS,
            HIDDEN_CHANNELS,
            EMBEDDING_DIM,
            OUT_CHANNELS,
            NUM_SHARED_CLUSTERS,
            GNN_TYPE,
            NUM_ATTENTION_HEADS,
            DROPOUT_RATE,
            NUM_INTRA_MODAL_LAYERS,
            NUM_INTER_MODAL_LAYERS,
        )
        loader, sample_index, cohort_svs_map, cohort_geojson_map, cohort_pos_levels = load_all_cohorts_mod(
            cohort_specs=COHORT_SPECS,
            allowlist=COHORT_ALLOWLIST,
        )
        if cohort_svs_map:
            COHORT_SVS.clear()
            COHORT_SVS.update(cohort_svs_map)
        if cohort_geojson_map:
            COHORT_CELLGJSON.clear()
            COHORT_CELLGJSON.update(cohort_geojson_map)
        if cohort_pos_levels:
            COHORT_POS_LEVEL.clear()
            COHORT_POS_LEVEL.update(cohort_pos_levels)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
        else:
            print(f"[WARN] MODEL_PATH not found: {MODEL_PATH} (using random init weights)")
        model.to(DEVICE); model.eval()

        # ========== Stage 2: Extract ==========
        extract_cell_cache = CACHE_DIR / "stage2_cell_df_raw.parquet"
        extract_gene_cache = CACHE_DIR / "stage2_gene_df_raw.parquet"
        extract_sample_cache = CACHE_DIR / "stage2_sample_dict.pt"
        extract_meta_required = {"stage": "stage2_extract"}
        use_extract_cache = False
        meta_extract = _cache_meta_matches("stage2_extract", extract_meta_required)
        if CACHE_ENABLED and meta_extract and extract_cell_cache.exists() and extract_gene_cache.exists() and extract_sample_cache.exists():
            try:
                cell_df_raw = pd.read_parquet(extract_cell_cache)
                gene_df_raw = pd.read_parquet(extract_gene_cache)
                sample_dict = torch.load(extract_sample_cache, map_location='cpu')
                sample_hash_current = _json_hash(sorted(cell_df_raw['sample_name'].astype(str).unique()))
                if meta_extract.get('sample_hash') == sample_hash_current:
                    use_extract_cache = True
                    print("[CACHE] Stage2: loaded raw cell/gene data from cache.")
            except Exception as e:
                print(f"[CACHE] Stage2 cache invalid: {e}; recomputing.")
                use_extract_cache = False

        if not use_extract_cache:
            cell_df_raw, gene_df_raw = extract_all_data_for_analysis_mod(model, loader, DEVICE)
            sample_dict = build_sample_dict_mod(loader)
            if CACHE_ENABLED:
                _ensure_cache_dir()
                try:
                    cell_df_raw.to_parquet(extract_cell_cache)
                    gene_df_raw.to_parquet(extract_gene_cache)
                except Exception as e:
                    print(f"[WARN] failed to cache raw dataframes: {e}")
                try:
                    torch.save(sample_dict, extract_sample_cache)
                except Exception as e:
                    print(f"[WARN] failed to cache sample_dict: {e}")
                sample_hash = _json_hash(sorted(cell_df_raw['sample_name'].astype(str).unique()))
                _cache_write_meta("stage2_extract", {**extract_meta_required, "sample_hash": sample_hash})

        # (2.1) Shared centers (prefer) or bootstrap
        W_shared = _get_any_centers_shared(model, DEVICE) if '_get_any_centers_shared' in globals() else None
        if W_shared is None:
            print("[INFO] no shared centers; bootstrap from embeddings ...")
            guess_k = (getattr(model, "k_cell", None)
                       or getattr(model, "num_shared_clusters", None)
                       or globals().get("NUM_SHARED_CLUSTERS", None)
                       or 6)
            W_shared = _bootstrap_centers_from_embeddings(
                cell_df=cell_df_raw, gene_df=gene_df_raw, k=int(guess_k), device=DEVICE
            )
            print(f"[INFO] bootstrapped centers: {tuple(W_shared.shape)}")
        Wc_set = Wg_set = W_shared

        # (2.2) Feature names & gene list (user-configurable with fallbacks)
        cell_features = None
        if cell_feature_template:
            try:
                cell_features = get_cell_feature_names(str(cell_feature_template))
            except Exception as e:
                print(f"[WARN] get_cell_feature_names failed on {cell_feature_template}: {e}")
        if not cell_features:
            nf = len([c for c in cell_df_raw.columns if c.startswith("raw_feat_")])
            cell_features = [f"raw_feat_{i}" for i in range(max(0, nf))]

        gene_list = []
        if gene_id_path:
            try:
                raw_gene_list = pd.read_csv(gene_id_path, header=None)[0].tolist()
                gene_list = clean_gene_list(raw_gene_list)
            except Exception as e:
                print(f"[WARN] load gene list failed on {gene_id_path}: {e}")
        if not gene_list:
            gene_cols = [c for c in gene_df_raw.columns if c.startswith("gene_expr_")]
            gene_list = [f"G{i}" for i in range(len(gene_cols))]

        # ========== Stage 3: Prototypes (optionally skip UMAP) ==========
        stage3_cell_cache = CACHE_DIR / "stage3_cell_df.parquet"
        stage3_gene_cache = CACHE_DIR / "stage3_gene_df.parquet"
        centers_sha = _tensor_hash(W_shared)
        stage3_meta_required = {
            "stage": "stage3",
            "fast_skip_umap": bool(FAST_SKIP_UMAP),
            "centers_sha": centers_sha,
            "tau_cell": 0.3,
            "tau_gene": 0.4,
        }
        meta_stage3 = _cache_meta_matches("stage3", stage3_meta_required)
        use_stage3_cache = False
        if CACHE_ENABLED and meta_stage3 and stage3_cell_cache.exists() and stage3_gene_cache.exists():
            try:
                cell_df = pd.read_parquet(stage3_cell_cache)
                gene_df = pd.read_parquet(stage3_gene_cache)
                did_umap = bool(meta_stage3.get('did_umap', not FAST_SKIP_UMAP))
                sample_hash_current = _json_hash(sorted(cell_df['sample_name'].astype(str).unique()))
                if meta_stage3.get('sample_hash') == sample_hash_current:
                    use_stage3_cache = True
                    print("[CACHE] Stage3: loaded prototypes/UMAP from cache.")
            except Exception as e:
                print(f"[CACHE] Stage3 cache invalid: {e}; recomputing.")
                use_stage3_cache = False

        if not use_stage3_cache:
            if FAST_SKIP_UMAP and 'analyze_archetypes_no_umap' in globals():
                cell_df = analyze_archetypes_no_umap(cell_df_raw, centers_tensor=Wc_set, entity_type='cell')
                gene_df = analyze_archetypes_no_umap(gene_df_raw, centers_tensor=Wg_set, entity_type='gene')
                did_umap = False
                print("[FAST] Used analyze_archetypes_no_umap (skip UMAP).")
            else:
                cell_df = analyze_archetypes_and_umap(cell_df_raw, centers_tensor=Wc_set, entity_type='cell', tau=0.3)
                gene_df = analyze_archetypes_and_umap(gene_df_raw, centers_tensor=Wg_set, entity_type='gene', tau=0.4)
                did_umap = True
                print("[INFO] UMAP computed for cell/gene.")
                
            if CACHE_ENABLED:
                try:
                    _ensure_cache_dir()
                    cell_df.to_parquet(stage3_cell_cache)
                    gene_df.to_parquet(stage3_gene_cache)
                    sample_hash = _json_hash(sorted(cell_df['sample_name'].astype(str).unique()))
                    _cache_write_meta("stage3", {**stage3_meta_required, "did_umap": did_umap, "sample_hash": sample_hash})
                except Exception as e:
                    print(f"[WARN] failed to cache stage3 outputs: {e}")

        del cell_df_raw, gene_df_raw
        gc.collect()

        if COHORT_ALLOWLIST:
            allow_norm = {_normalize_cohort_key(c) for c in COHORT_ALLOWLIST}
            if 'cohort' in cell_df.columns:
                cell_df = cell_df[cell_df['cohort'].map(_normalize_cohort_key).isin(allow_norm)]
            if 'cohort' in gene_df.columns:
                gene_df = gene_df[gene_df['cohort'].map(_normalize_cohort_key).isin(allow_norm)]
            else:
                allowed_samples = set(cell_df['sample_name'].astype(str).unique())
                if allowed_samples:
                    gene_df = gene_df[gene_df['sample_name'].astype(str).isin(allowed_samples)]
            def _cohort_of_data(di):
                coh = getattr(di, 'cohort', None)
                if isinstance(coh, list) and coh:
                    coh = coh[0]
                return coh
            sample_dict = {
                k: v for k, v in sample_dict.items()
                if _normalize_cohort_key(_cohort_of_data(v)) in allow_norm
            }

        # (optional) core-cell selection
        if 'select_core_cells' in globals():
            cell_df = select_core_cells(
                cell_df, by=('cohort','prototype_id'),
                top_quantile=0.80, abs_floor=0.03, min_keep_per_group=2000
            )

        # ========== Stage 4: Merge map from reference & apply ==========
        stage4_cell_cache = CACHE_DIR / "stage4_cell_df.parquet"
        stage4_gene_cache = CACHE_DIR / "stage4_gene_df.parquet"
        cohort_candidates = [c for c in cell_df['cohort'].dropna().unique().tolist()]
        if not cohort_candidates:
            raise RuntimeError("No cohort labels available to derive reference cohort.")
        if COHORT_ALLOWLIST:
            allow_norm = {_normalize_cohort_key(c) for c in COHORT_ALLOWLIST}
            ref = next((c for c in cohort_candidates if _normalize_cohort_key(c) in allow_norm), cohort_candidates[0])
        else:
            ref = cohort_candidates[0]
        REF_COHORT = str(ref)
        merge_map_json = Path(ANALYSIS_OUTPUT_DIR) / f"merge_map_from_{REF_COHORT}.json"
        stage4_meta_required = {
            "stage": "stage4_merge",
            "min_center_cos": 0.7,
            "ref_cohort": REF_COHORT,
            "centers_sha": centers_sha,
        }
        meta_stage4 = _cache_meta_matches("stage4_merge", stage4_meta_required)
        use_stage4_cache = False
        if CACHE_ENABLED and meta_stage4 and stage4_cell_cache.exists() and merge_map_json.exists():
            try:
                cell_df = pd.read_parquet(stage4_cell_cache)
                gene_df = pd.read_parquet(stage4_gene_cache) if stage4_gene_cache.exists() else gene_df
                with open(merge_map_json, "r", encoding="utf-8") as f:
                    merge_map = {int(k): int(v) for k, v in json.load(f).items()}
                sample_hash_current = _json_hash(sorted(cell_df['sample_name'].astype(str).unique()))
                if meta_stage4.get('sample_hash') == sample_hash_current and meta_stage4.get('merge_map_sha') == _json_hash(merge_map):
                    use_stage4_cache = True
                    print("[CACHE] Stage4: loaded merged data from cache.")
            except Exception as e:
                print(f"[CACHE] Stage4 cache invalid: {e}; recomputing.")
                use_stage4_cache = False

        if not use_stage4_cache:
            cell_df_ref = cell_df[cell_df['cohort'] == REF_COHORT].copy()
            if cell_df_ref.empty:
                raise RuntimeError(f"No cells for reference cohort {REF_COHORT}.")
            log2EN_ref = run_archetype_interface_analysis(cell_df_ref, ANALYSIS_OUTPUT_DIR, file_prefix=f"{REF_COHORT}_only")

            merge_map = build_merge_map_multi_signal_mod(
                model, cell_df=cell_df, gene_bulk_df=None, log2EN=log2EN_ref,
                p_min=0.05, n_min=1500, keep_top_m=3, w=(0.35,0.25,0.25,0.10,0.05),
                min_center_cos=0.7, centers_tensor=W_shared
            )
            with open(merge_map_json, "w", encoding="utf-8") as f:
                json.dump({int(k): int(v) for k, v in merge_map.items()}, f, indent=2)

            cell_df = apply_merge_to_cell_df_mod(cell_df, merge_map, overwrite_prototype_id=True)
            if 'prototype_id' in gene_df.columns:
                gene_df = apply_merge_to_cell_df_mod(gene_df, merge_map, overwrite_prototype_id=False)

            if CACHE_ENABLED:
                try:
                    _ensure_cache_dir()
                    cell_df.to_parquet(stage4_cell_cache)
                    gene_df.to_parquet(stage4_gene_cache)
                    sample_hash = _json_hash(sorted(cell_df['sample_name'].astype(str).unique()))
                    _cache_write_meta(
                        "stage4_merge",
                        {
                            **stage4_meta_required,
                            "sample_hash": sample_hash,
                            "merge_map_sha": _json_hash(merge_map),
                        },
                    )
                    for done_flag in Path(ANALYSIS_OUTPUT_DIR).glob("COHORT_*/.done"):
                        try:
                            done_flag.unlink()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[WARN] failed to cache stage4 data: {e}")
        else:
            if stage4_gene_cache.exists():
                gene_df = pd.read_parquet(stage4_gene_cache)
            elif 'prototype_id' in gene_df.columns:
                gene_df = apply_merge_to_cell_df_mod(gene_df, merge_map, overwrite_prototype_id=False)

        # ==========  PREP: sanitize cell_type BEFORE any kNN ecology ==========
        if 'cell_type' in cell_df.columns:
            cell_df['cell_type'] = (cell_df['cell_type']
                                    .astype('string')
                                    .str.strip()
                                    .replace({'NA': pd.NA, 'N/A': pd.NA, 'NULL': pd.NA, 'None': pd.NA, '': pd.NA, 'nan': pd.NA})
                                    .fillna('Unknown'))

        # ==========  Gate + Local ecology ==========
        if 'attach_cell_gate_strength' in globals():
            cell_df = attach_cell_gate_strength(cell_df, sample_dict, agg="mean")
        else:
            print("[WARN] attach_cell_gate_strength not found; gate overlay will be skipped.")

        cell_df = ensure_celltype_from_onehot(cell_df, feature_names=cell_features)

        cell_df = gate_strength_within_proto_by_type(
            cell_df,
            gate_col='gate_strength_norm',
            min_cells_per_type=50,
            out_dir=os.path.join(ANALYSIS_OUTPUT_DIR, "gate_by_type")
        )

        # Local ecology (kNN)
        neigh_proto_prop, neigh_type_prop = compute_local_ecology_kNN(cell_df, k=15)
        cell_df = cell_df.join(neigh_proto_prop, how='left')

        # UMAP advanced panels
        if did_umap:
            try:
                plot_umap_advanced_panels_mod(
                    cell_df, gene_df,
                    out_dir=os.path.join(ANALYSIS_OUTPUT_DIR, "1_umap_visualizations_advanced"),
                    max_per_proto=50_000
                )
            except Exception as e:
                print(f"[WARN] advanced UMAP panels failed: {e}")
        else:
            print("[FAST] Skip UMAP advanced panels (FAST_SKIP_UMAP=True)")

        # ========== Stage 5: Hub / SCI / LR / trajectory ==========
        # (5.1) optional zones
        zones_per_sample = {}
        if 'make_histologic_zones' in globals():
            for sname in sorted(cell_df['sample_name'].unique()):
                zones_per_sample[sname] = make_histologic_zones(cell_df, sname, grid=512, thr=0.55, buffer_px=800)

        # (5.2) Hub (disabled  prototypes already merged upstream)
        proto2hub = {}
        hub_centers = None
        S_all = None

        # (5.3) SCI-lite ( robust index fix before plotting)
        if 'sci_lite' in globals():
            SCI = sci_lite(cell_df, k=8)

            # --- 1)  Inf  NaN
            SCI = SCI.replace([np.inf, -np.inf], np.nan)
                                   
            # --- 2)  NaN 
            num_cols = SCI.select_dtypes(include=[np.number]).columns
            if len(num_cols):
              numeric = SCI[num_cols].apply(pd.to_numeric, errors='coerce')
              mask = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
              keep = pd.Series(mask.any(axis=1), index=SCI.index)
              SCI = SCI.loc[keep].copy()

            # --- 3) / 0
            # 
            prob_like = [c for c in SCI.columns if c.lower().endswith(('_prop','_prob','_rate'))]
            for c in prob_like:
                SCI[c] = SCI[c].fillna(0.0)

            # --- 4)  0..n-1 indexint 
            SCI = SCI.reset_index(drop=True)

            # 
            sci_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "SCI_lite.csv")
            SCI.to_csv(sci_csv, index=False)

            # --- 5)  index plot_sci_visuals  int  index
            if 'plot_sci_visuals' in globals():
                try:
                    plot_sci_visuals(SCI, out_dir=os.path.join(ANALYSIS_OUTPUT_DIR, "3_sci_visuals"))
                except Exception as e:
                    print(f"[WARN] plot_sci_visuals failed (after sanitize): {e}")


        # (5.4) LR-lite (pseudo-bulk)
        bulk = _pseudobulk_gene_mean(gene_df, gene_list)
        if not bulk.empty and 'lr_lite_scores' in globals():
            if 'hub_id' in cell_df.columns:
                cnt = cell_df.groupby(['sample_name','hub_id']).size().unstack(fill_value=0)
                dom_hub = cnt.div(cnt.sum(axis=1), axis=0).idxmax(axis=1)
                bulk2 = bulk.set_index('sample_name').join(dom_hub.rename('dom_hub'))
            else:
                bulk2 = bulk.set_index('sample_name')
            gcols = [c for c in bulk2.columns if c not in ('label','dom_hub')]
            LR = lr_lite_scores(bulk2[gcols], set(gcols))
            LR.to_csv(os.path.join(ANALYSIS_OUTPUT_DIR, "LR_lite_scores_by_sample.csv"))

        # (5.5) Trajectory via diffusion
        anchor_proto_mix, anchor_type_mix = aggregate_ecology_by_anchor(cell_df, neigh_proto_prop, neigh_type_prop, min_cells=200)
        anchor_geneproto_mix = compute_anchor_geneproto_mix(cell_df, gene_df, sample_dict, min_cells=200)
        def _to_hub(mat):
            if mat is None or mat.empty or not proto2hub: return mat
            df = mat.copy(); df['hub_id'] = [proto2hub.get(int(i), -1) for i in df.index]
            return df.groupby('hub_id').mean().sort_index()
        state_parts = [x for x in (_to_hub(anchor_proto_mix), _to_hub(anchor_type_mix), _to_hub(anchor_geneproto_mix)) if (x is not None and not x.empty)]
        if state_parts and 'diffusion_pseudotime' in globals():
            from functools import reduce
            all_cols = sorted(set().union(*[set(x.columns) for x in state_parts]))
            state_parts = [x.reindex(columns=all_cols, fill_value=0) for x in state_parts]
            state_df = reduce(lambda a,b: (a.add(b, fill_value=0)/2.0), state_parts)
            max_eig = max(1, min(3, state_df.shape[0]-1, state_df.shape[1]))
            U, pt = diffusion_pseudotime(state_df, n_components=max_eig, root_index=0)
            if U.size and U.shape[1] > 0:
                ucols = [f'd{i+1}' for i in range(U.shape[1])]
                pd.DataFrame(U, index=state_df.index, columns=ucols).to_csv(
                    os.path.join(ANALYSIS_OUTPUT_DIR, "hub_diffusion_coords.csv")
                )
            else:
                print("[WARN] hub diffusion: insufficient eigenvectors, skip coordinate export")
            pd.Series(pt, index=state_df.index, name='pseudotime').to_csv(
                os.path.join(ANALYSIS_OUTPUT_DIR, "hub_pseudotime.csv")
            )

        # ==========  NEW: Interface DE & WSI overlays remain unchanged ==========
        cell_df, interface_pairs = add_interface_strength_columns(cell_df, thr_exist=PAIR_EXIST_THRESHOLD)
        if not interface_pairs:
            print(f"[IF-DE] no prototype pairs passed --pair-exist-thr={PAIR_EXIST_THRESHOLD:.3f}; "
                  "try lowering this threshold if interfaces seem sparse.")
        if INTERFACE_DEBUG:
            print(f"[IF-DE][debug] candidate pairs: {interface_pairs}")
            print(f"[IF-DE][debug] cell_df index head: {cell_df.index[:5].tolist() if len(cell_df.index) else []}")
            print(f"[IF-DE][debug] gene_df index head: {gene_df.index[:5].tolist() if len(gene_df.index) else []}")
            gcols_dbg = [c for c in gene_df.columns if c.startswith('gene_expr_')]
            print(f"[IF-DE][debug] gene_df gene_expr cols: {len(gcols_dbg)} (showing up to 5) {gcols_dbg[:5]}")
            print(f"[IF-DE][debug] balance_delta={IF_BALANCE_DELTA}, balance_pairs={sorted(list(IF_BALANCE_PAIRS)) if IF_BALANCE_PAIRS else []}, interface_quantile={INTERFACE_QUANTILE}")
            print(f"[IF-DE][debug] agg_method={AGG_METHOD}, agg_subsample={AGG_SUBSAMPLE}, agg_winsor={AGG_WINSOR_Q}, test_method={TEST_METHOD}, gene_sets={ENABLE_GENE_SETS}")
        IF_OUT = os.path.join(ANALYSIS_OUTPUT_DIR, "5_spatial_interaction_analysis", "A3_core_edge_DE")
        pair_stats_all = run_interface_DE_all_pairs(
            cell_df=cell_df, gene_df=gene_df, gene_list=gene_list,
            out_root=IF_OUT, pairs=interface_pairs,
            t_if=INTERFACE_THRESHOLD, t_core=CORE_THRESHOLD,
            min_cells_per_group=max(10, INTERFACE_MINCELLS), p_thr=0.01, log2fc_thr=0.0,
            tag_suffix='', fallback_flag=False,
            debug=INTERFACE_DEBUG, debug_label='default',
            sample_dict=sample_dict,
            balance_delta=IF_BALANCE_DELTA,
            balance_pairs=IF_BALANCE_PAIRS,
            interface_quantile=INTERFACE_QUANTILE,
            agg_method=AGG_METHOD,
            agg_subsample=AGG_SUBSAMPLE,
            agg_winsor=AGG_WINSOR_Q,
            test_method=TEST_METHOD,
            enable_gene_sets=ENABLE_GENE_SETS
        )
        fallback_used = False
        try:
            import glob as _glob
            deg_files = _glob.glob(os.path.join(IF_OUT, '*_interface_DEGs.csv'))
            if not deg_files:
                print("[IF-DE] no significant interface genes at default thresholds; retrying with relaxed parameters.")
                relaxed_core = max(0.30, CORE_THRESHOLD - 0.10)
                relaxed_if = max(0.15, INTERFACE_THRESHOLD - 0.10)
                pair_stats_fb = run_interface_DE_all_pairs(
                    cell_df=cell_df, gene_df=gene_df, gene_list=gene_list,
                    out_root=IF_OUT, pairs=interface_pairs,
                    t_if=relaxed_if, t_core=relaxed_core,
                    min_cells_per_group=max(5, INTERFACE_MINCELLS // 2), p_thr=0.10, log2fc_thr=0.05,
                    tag_suffix='_fallback', fallback_flag=True,
                    debug=INTERFACE_DEBUG, debug_label='fallback',
                    sample_dict=sample_dict,
                    balance_delta=IF_BALANCE_DELTA,
                    balance_pairs=IF_BALANCE_PAIRS,
                    interface_quantile=INTERFACE_QUANTILE,
                    agg_method=AGG_METHOD,
                    agg_subsample=AGG_SUBSAMPLE,
                    agg_winsor=AGG_WINSOR_Q,
                    test_method=TEST_METHOD,
                    enable_gene_sets=ENABLE_GENE_SETS
                )
                fallback_used = True
                pair_stats_all.extend(pair_stats_fb)
        except Exception as e:
            print(f"[IF-DE] fallback check failed: {e}")
        pair_counts_path = os.path.join(IF_OUT, 'interface_pair_sample_counts.csv')
        pair_stats_df = pd.DataFrame(pair_stats_all)
        if pair_stats_df.empty:
            pair_stats_df = pd.DataFrame(columns=['pair','pair_tag','fallback','n_samples_used','interface_total_cells','core_total_cells'])
        pair_stats_df.to_csv(pair_counts_path, index=False)

        if PLOT_INTERFACE_CONTOURS and interface_pairs:
            contour_dir = os.path.join(ANALYSIS_OUTPUT_DIR, "interface_contour_overlays")
            plot_interface_contours(
                cell_df=cell_df,
                pairs=interface_pairs,
                out_dir=contour_dir,
                t_if=INTERFACE_THRESHOLD,
                balance_delta=IF_BALANCE_DELTA,
                balance_pairs=IF_BALANCE_PAIRS,
                interface_quantile=INTERFACE_QUANTILE,
                min_cells=max(10, INTERFACE_MINCELLS // 2),
                debug=INTERFACE_DEBUG
            )

        if not COHORT_SVS or not COHORT_CELLGJSON:
            print("[WSI] Missing cohort SVS/GeoJSON mappings; skip asset status report.")
        else:
            report_wsi_asset_status_mod(
                cell_df,
                COHORT_SVS,
                COHORT_CELLGJSON,
                out_csv=os.path.join(ANALYSIS_OUTPUT_DIR, "wsi_asset_status.csv"),
                allow_cohorts=COHORT_ALLOWLIST
            )
        # overlay_all_interfaces_on_wsi(
        #     cell_df=cell_df, pairs=interface_pairs,
        #     svs_root_map=COHORT_SVS, cellgeojson_root_map=COHORT_CELLGJSON,
        #     out_root=os.path.join(ANALYSIS_OUTPUT_DIR, "WSI_overlays_interfaces"),
        #     target_level=1, k_vote=3, alpha_outline=80
        # )

        # ==========  NEW: Sample-level spatial metrics + MWU test ==========
        # (If coordinates are in pixels, set microns_per_pixel accordingly)
        metrics = assemble_spatial_metrics_mod(cell_df, k=15, radii=(100,150,200),
                                           focus_proto=2, pairs=((2,0),(2,4)),
                                           microns_per_pixel=1.0,
                                           core_thr=CORE_THRESHOLD,
                                           excl_thr=EXCL_THRESHOLD,
                                           interface_thr=INTERFACE_THRESHOLD)
        metrics_path = os.path.join(ANALYSIS_OUTPUT_DIR, "spatial_metrics_summary.csv")
        metrics.to_csv(metrics_path)

        if 'pcr_response' in metrics.columns:
            from scipy.stats import mannwhitneyu
            def mwu(x, y):
                x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
                if (len(x)>=3) and (len(y)>=3):
                    return mannwhitneyu(x, y, alternative='two-sided').pvalue
                return np.nan
            grp = metrics['pcr_response']
            base_cols = ['ID_2_0','ID_2_4','P2_occ_r100','P2_occ_r150','P2_occ_r200']
            extra_cols = [c for c in metrics.columns
                          if c.startswith('core_frac_P') or c.startswith('interface_frac_P')]
            test_cols = []
            for col in base_cols + extra_cols:
                if col in metrics.columns and col not in test_cols:
                    test_cols.append(col)

            mwu_rows = []
            for col in test_cols:
                series = pd.to_numeric(metrics[col], errors='coerce')
                if series.notna().sum() < 6:
                    continue
                p = mwu(series[grp=='pCR'].values, series[grp!='pCR'].values)
                print(f"[MWU] {col}: p={p:.3g}")
                mwu_rows.append({
                    'cohort': 'ALL',
                    'metric': col,
                    'p_value': float(p) if np.isfinite(p) else np.nan,
                    'n_pCR': int((grp=='pCR').sum()),
                    'n_non_pCR': int((grp!='pCR').sum()),
                    'mean_pCR': float(series[grp=='pCR'].mean()),
                    'mean_non_pCR': float(series[grp!='pCR'].mean())
                })
            if 'cohort' in metrics.columns:
                for coh_value, sub_metrics in metrics.groupby('cohort'):
                    coh_mask = metrics['cohort'] == coh_value
                    grp_coh = grp[coh_mask]
                    if grp_coh.shape[0] < 6:
                        continue
                    for col in test_cols:
                        if col not in sub_metrics.columns:
                            continue
                        series = pd.to_numeric(sub_metrics[col], errors='coerce')
                        if series.notna().sum() < 6:
                            continue
                        p = mwu(series[grp_coh=='pCR'].values, series[grp_coh!='pCR'].values)
                        mwu_rows.append({
                            'cohort': str(coh_value),
                            'metric': col,
                            'p_value': float(p) if np.isfinite(p) else np.nan,
                            'n_pCR': int((grp_coh=='pCR').sum()),
                            'n_non_pCR': int((grp_coh!='pCR').sum()),
                            'mean_pCR': float(series[grp_coh=='pCR'].mean()),
                            'mean_non_pCR': float(series[grp_coh!='pCR'].mean())
                        })
            if mwu_rows:
                mwu_df = pd.DataFrame(mwu_rows)
                try:
                    from statsmodels.stats.multitest import multipletests as _mwu_multipletests
                    pvals = mwu_df['p_value'].to_numpy()
                    mask = np.isfinite(pvals)
                    if mask.any():
                        qvals = np.full_like(pvals, np.nan, dtype=float)
                        qvals[mask] = _mwu_multipletests(pvals[mask], method='fdr_bh')[1]
                        mwu_df['q_value'] = qvals
                except Exception:
                    pass
                mwu_df.to_csv(
                    os.path.join(ANALYSIS_OUTPUT_DIR, 'spatial_metrics_mwu_summary.csv'),
                    index=False
                )

        if 'generate_publication_interface_figures' in globals():
            try:
                generate_publication_interface_figures(
                    analysis_root=ANALYSIS_OUTPUT_DIR,
                    gene_list=gene_list,
                    ref_cohort=REF_COHORT,
                    focus_protos=(0, 2, 4),
                    top_n=publication_top_n
                )
            except Exception as e:
                print(f"[WARN] publication figure generation failed: {e}")

        # ========== WSI overlays ==========
        WSI_OUT = os.path.join(ANALYSIS_OUTPUT_DIR, "WSI_overlays"); os.makedirs(WSI_OUT, exist_ok=True)
        if not COHORT_SVS or not COHORT_CELLGJSON:
            print("[WSI] Missing cohort SVS/GeoJSON mappings; skip WSI overlays.")
        else:
            default_anchor = int(cell_df['prototype_id'].value_counts().idxmax()) if 'prototype_id' in cell_df.columns else 0

            for sname, sdf in cell_df.groupby('sample_name'):
                cohort_raw = sdf['cohort'].iat[0] if 'cohort' in sdf.columns else None
                cohort_label = str(cohort_raw).strip() if cohort_raw is not None else "NA"
                svs_root = _resolve_cohort_path(COHORT_SVS, cohort_raw)
                gj_root  = _resolve_cohort_path(COHORT_CELLGJSON, cohort_raw)

                if not (svs_root and gj_root):
                    print(f"[WSI] skip {sname}: cohort '{cohort_label}' missing svs/geojson root mapping")
                    continue

                cohort_out_dir = os.path.join(WSI_OUT, cohort_label)
                os.makedirs(cohort_out_dir, exist_ok=True)

                overlay_wsi_prototypes_for_sample_mod(
                    sample_name=sname, cell_df=cell_df,
                    svs_root=svs_root, cellgeojson_root=gj_root,
                    out_png=os.path.join(cohort_out_dir, f"{sname}__prototypes.png"),
                    use_merged=True, target_level=1, alpha_fill=120, alpha_outline=90, k_vote=3, draw_outline=True,
                    core_threshold=CORE_THRESHOLD
                )

            if 'gate_strength_norm' in cell_df.columns:
                overlay_wsi_gate_for_sample_mod(
                    sample_name=sname, cell_df=cell_df,
                    svs_root=svs_root, cellgeojson_root=gj_root,
                    out_png=os.path.join(cohort_out_dir, f"{sname}__gate.png"),
                    scalar_col='gate_strength_norm',
                    target_level=1, alpha_outline=80, k_vote=3
                )

                available_types = sdf['cell_type'].astype(str).value_counts().index.tolist()
                for ct in available_types:
                    out_ct = os.path.join(cohort_out_dir, f"{sname}__gate_{ct}_z.png")
                    try:
                        overlay_wsi_gate_by_celltype(
                            sample_name=sname,
                            cell_df=cell_df,
                            svs_root=svs_root,
                            cellgeojson_root=gj_root,
                            out_png=out_ct,
                            target_cell_type=str(ct),
                            scalar_col='gate_type_z_within_proto',
                            target_level=1,
                            k_vote=3
                        )
                    except Exception as ee:
                        print(f"[WARN] overlay_wsi_gate_by_celltype failed on {sname} [{ct}]: {ee}")

            need_col = f'P{int(default_anchor)}'
            if need_col in cell_df.columns:
                overlay_wsi_core_edge_for_sample_mod(
                    sample_name=sname, cell_df=cell_df,
                    svs_root=svs_root, cellgeojson_root=gj_root,
                    out_png=os.path.join(cohort_out_dir, f"{sname}__core_edge_P{default_anchor}.png"),
                    anchor_pid=default_anchor, prop_thr=CORE_THRESHOLD, k_vote=3, target_level=1
                )

        try:
            WSI_ALIGN_OUT = os.path.join(ANALYSIS_OUTPUT_DIR, "WSI_overlays_alignment"); os.makedirs(WSI_ALIGN_OUT, exist_ok=True)
            if not COHORT_SVS or not COHORT_CELLGJSON:
                print("[WSI-ALIGN] Missing cohort maps; skip alignment overlays.")
            else:
                SVS_ROOT_MAP = COHORT_SVS
                CELLGJSON_ROOT_MAP = COHORT_CELLGJSON
                for d in loader:
                    name   = d.patient_id[0] if isinstance(d.patient_id, list) else d.patient_id
                    cohort_raw = d.cohort[0] if isinstance(d.cohort, list) else d.cohort
                    if COHORT_ALLOWLIST:
                        allow_norm = {_normalize_cohort_key(c) for c in COHORT_ALLOWLIST}
                        if _normalize_cohort_key(cohort_raw) not in allow_norm:
                            continue
                    svs_root = _resolve_cohort_path(SVS_ROOT_MAP, cohort_raw)
                    cellvit_root = _resolve_cohort_path(CELLGJSON_ROOT_MAP, cohort_raw)
                    if not (svs_root and cellvit_root):
                        continue
                    run_alignment_wsi_overlay_for_batch_mod(
                        model=model,
                        data_obj=d,
                        svs_root=svs_root,
                        cellgeojson_root=cellvit_root,
                        cohort_pos_level_map=COHORT_POS_LEVEL,
                        target_level=None, soften_temp=3.0, out_root=WSI_ALIGN_OUT,
                        nn_k=3, nn_sigma_lvl0=24.0,
                        blend='sum', sigma_lvl0=16.0,
                        cmap_name='coolwarm', cmap_use_full=None,
                        low_clip_q=10.0, high_clip_q=98.0,
                        alpha_gamma=0.7, alpha_max=220, alpha_floor=24,
                        brighten_bg=0.10, grow_cover_px=3,
                        draw_all_outlines=True, fallback_disk_px=1
                    )
        except Exception as e:
            print(f"[WARN] alignment WSI overlay failed: {e}")

        # ========== Stage 6: Per-cohort pipelines ==========
        available = [str(x) for x in cell_df['cohort'].dropna().unique().tolist()]
        allowed   = sorted(set(COHORT_SVS.keys()) | set(COHORT_CELLGJSON.keys()))
        allowed_map = {_normalize_cohort_key(k): k for k in allowed}
        cohorts = []
        seen_norm = set()
        for val in available:
            norm = _normalize_cohort_key(val)
            if not norm or norm in seen_norm:
                continue
            if allowed_map:
                if norm in allowed_map:
                    cohorts.append(allowed_map[norm])
                    seen_norm.add(norm)
            else:
                cohorts.append(val)
                seen_norm.add(norm)
        if not cohorts:
            cohorts = available if available else ["ALL"]
        for coh in cohorts:
            run_per_cohort_pipeline(
                cohort_name=coh,
                model=model,
                cell_df_all=cell_df,
                gene_df_all=gene_df,
                sample_dict=sample_dict,
                gene_list=gene_list,
                cell_features=cell_features,
                svs_root_map=COHORT_SVS,
                cellgeojson_root_map=COHORT_CELLGJSON,
                base_out_dir=ANALYSIS_OUTPUT_DIR,
                core_thr=CORE_THRESHOLD,
            )

        log_gate_stats(sample_dict, ANALYSIS_OUTPUT_DIR)

        analysis_meta = {
            'core_threshold': CORE_THRESHOLD,
            'core_exclusivity_margin': EXCL_THRESHOLD,
            'interface_threshold': INTERFACE_THRESHOLD,
            'interface_min_cells': INTERFACE_MINCELLS,
            'pair_exist_threshold': PAIR_EXIST_THRESHOLD,
            'interface_balance_delta': IF_BALANCE_DELTA,
            'interface_balance_pairs': sorted(list(IF_BALANCE_PAIRS)) if IF_BALANCE_PAIRS else [],
            'interface_quantile': INTERFACE_QUANTILE,
            'interface_agg_method': AGG_METHOD,
            'interface_agg_subsample': AGG_SUBSAMPLE,
            'interface_agg_winsor_quantile': AGG_WINSOR_Q,
            'interface_test_method': TEST_METHOD,
            'interface_gene_sets_enabled': bool(ENABLE_GENE_SETS),
            'interface_plot_contours': bool(PLOT_INTERFACE_CONTOURS),
            'interface_debug_enabled': bool(INTERFACE_DEBUG),
            'publication_top_n': publication_top_n,
            'interface_fallback_used': bool(fallback_used),
            'interface_pair_sample_counts_csv': pair_counts_path,
            'spatial_metrics_summary_csv': metrics_path,
            'spatial_metrics_mwu_summary_csv': os.path.join(ANALYSIS_OUTPUT_DIR, 'spatial_metrics_mwu_summary.csv'),
            'interface_deg_summary_csv': os.path.join(ANALYSIS_OUTPUT_DIR, 'figures_publication', 'interface_deg_summary.csv')
        }
        try:
            with open(os.path.join(ANALYSIS_OUTPUT_DIR, 'analysis_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(analysis_meta, f, indent=2)
        except Exception as e:
            print(f"[WARN] failed to write analysis_meta.json: {e}")

        print("\n" + "="*70)
        print(" Deep Dive Analysis Pipeline  Completed! ")
        print(f"Outputs: {ANALYSIS_OUTPUT_DIR}")
        print("--> UMAP panels in: 1_umap_visualizations_advanced/")
        print("--> SCI visuals in: 3_sci_visuals/")
        print("--> WSI overlays in: WSI_overlays/ (and WSI_overlays_alignment/)")
        print("--> WSI interface overlays in: WSI_overlays_interfaces/")
        print("--> Gene programs in: 4_gene_functional_analysis/")
        print("--> Spatial diffs in: 5_spatial_interaction_analysis/ (A3_core_edge_DE)")
        print("--> Spatial metrics summary: spatial_metrics_summary.csv")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n Error in main pipeline: {e}")
        import traceback; traceback.print_exc()

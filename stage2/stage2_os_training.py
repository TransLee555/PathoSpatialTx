import os
import glob
import logging
import traceback
import random
import gc
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax
from scipy.spatial import Delaunay
from openslide import OpenSlide
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from MHGL_ST_model import HierarchicalMultiModalGNN, GraphAugmentations

from typing import List, Optional, Tuple
import math


def _set_global_seed(seed: int = 233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the stage 2 overall-survival predictor on processed multi-modal graphs."
    )
    parser.add_argument("--root-dir", type=Path, required=True, help="Directory used for processed dataset artifacts and checkpoints.")
    parser.add_argument("--cell-output-dir", type=Path, required=True, help="Root directory containing CellViT outputs (per-sample subdirectories).")
    parser.add_argument("--gene-dir", type=Path, required=True, help="Directory containing gene expression prediction CSV files.")
    parser.add_argument("--svs-dir", type=Path, required=True, help="Directory containing the raw SVS files.")
    parser.add_argument("--label-file", type=Path, required=True, help="CSV file with survival annotations (must include PatientID, T, E columns).")
    parser.add_argument("--graph-viz-dir", type=Path, default=None, help="Optional directory to store graph visualizations (defaults to ROOT/graph_visualizations).")

    parser.add_argument("--patch-size-level0", type=int, default=512, help="Patch size in level-0 pixels when building graphs.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction for StratifiedShuffleSplit.")
    parser.add_argument("--num-time-bins", type=int, default=5, help="Number of discrete time bins for survival modeling.")
    parser.add_argument("--time-bin-method", choices=["quantile", "linear"], default="quantile", help="Strategy for computing time-bin edges.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for DataLoader instances.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")

    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden channel size inside the GNN.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension passed to the readout head.")
    parser.add_argument("--num-shared-clusters", type=int, default=6, help="Number of shared archetype clusters.")
    parser.add_argument("--gnn-type", choices=["Transformer", "GAT", "GCN"], default="Transformer", help="Intra-modal GNN backbone type.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads for Transformer/GAT layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate applied inside the model.")
    parser.add_argument("--num-intra-layers", type=int, default=3, help="Number of intra-modal GNN layers.")
    parser.add_argument("--num-inter-layers", type=int, default=1, help="Number of inter-modal fusion layers.")

    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum number of training epochs.")
    parser.add_argument("--scheduler-patience", type=int, default=15, help="ReduceLROnPlateau patience (epochs).")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau decay factor.")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Minimum learning rate for ReduceLROnPlateau.")
    parser.add_argument("--early-stop-patience", type=int, default=500, help="Stop training if no validation improvement for this many epochs.")

    parser.add_argument("--cluster-warmup-epochs", type=int, default=1, help="Epochs before enabling center updates / clustering loss.")
    parser.add_argument("--cluster-lambda-max", type=float, default=0.02, help="Maximum clustering loss weight.")
    parser.add_argument("--cluster-lambda-ramp", type=int, default=100, help="Epochs used for cosine ramp-up of clustering weight.")
    parser.add_argument("--base-cluster-temperature", type=float, default=0.4, help="Temperature for base archetype assignments.")
    parser.add_argument("--base-confidence-weight", type=float, default=3e-2, help="Confidence regularizer weight for shared clusters.")
    parser.add_argument("--base-margin-weight", type=float, default=5e-3, help="Margin loss weight for shared clusters.")
    parser.add_argument("--base-margin-delta", type=float, default=0.20, help="Margin delta for shared cluster separation.")
    parser.add_argument("--center-repulsion", type=float, default=1e-2, help="Center repulsion weight.")
    parser.add_argument("--center-separation", type=float, default=0.75, help="Desired separation scale between centers.")
    parser.add_argument("--assignment-balance", type=float, default=0.0, help="Assignment balance weight.")
    parser.add_argument("--cell-cluster-temperature", type=float, default=0.20, help="Temperature for cell-specific clustering loss.")
    parser.add_argument("--cell-cluster-confidence", type=float, default=5e-2, help="Confidence weight for cell-specific clustering loss.")
    parser.add_argument("--gene-cluster-temperature", type=float, default=0.30, help="Temperature for gene-specific clustering loss.")
    parser.add_argument("--gene-cluster-confidence", type=float, default=2e-2, help="Confidence weight for gene-specific clustering loss.")

    parser.add_argument("--graph-augment-rotation-prob", type=float, default=0.2, help="Probability of applying a random rotation when augmenting graphs.")
    parser.add_argument("--graph-augment-flip-prob", type=float, default=0.2, help="Probability of applying a horizontal flip when augmenting graphs.")

    parser.add_argument("--device", default="auto", help="Torch device spec (e.g., 'cpu', 'cuda:0', or 'auto').")
    parser.add_argument("--preferred-gpu", type=int, default=1, help="Preferred CUDA index when --device=auto and multiple GPUs are available.")
    parser.add_argument("--seed", type=int, default=233, help="Random seed used for dataset splitting and initialization.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity.")
    return parser.parse_args()


def resolve_device(device_str: str, preferred_index: int) -> torch.device:
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            if torch.cuda.device_count() > preferred_index:
                return torch.device(f"cuda:{preferred_index}")
            return torch.device("cuda:0")
        return torch.device("cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def ensure_logdata(data: HeteroData) -> HeteroData:
    """Ensure node features are floating-point tensors and only check once."""
    if getattr(data, "_logdata_checked", False):
        return data
    for node_type in ('cell', 'gene'):
        if node_type in data.node_types and hasattr(data[node_type], 'x') and data[node_type].x is not None:
            if not torch.is_floating_point(data[node_type].x):
                data[node_type].x = data[node_type].x.float()
    data._logdata_checked = True
    return data


@torch.no_grad()
def log_archetype_usage(tag: str,
                        embeddings: Optional[torch.Tensor],
                        centers: Optional[torch.Tensor],
                        temperature: float,
                        sample_prob: float = 0.2):
    if embeddings is None or centers is None or embeddings.numel() == 0 or centers.numel() == 0:
        return
    if random.random() > sample_prob:
        return
    embed = embeddings.detach().float()
    cent = centers.detach().float()
    embed = embed / embed.norm(dim=1, keepdim=True).clamp_min(1e-6)
    cent = cent / cent.norm(dim=1, keepdim=True).clamp_min(1e-6)
    dist = torch.cdist(embed, cent, p=2)
    logits = -dist / max(temperature, 1e-6)
    logits -= logits.max(dim=1, keepdim=True).values
    Q = torch.softmax(logits, dim=1)
    usage = Q.mean(dim=0)
    qmax = Q.max(dim=1).values
    usage_dict = {int(i): float(u) for i, u in enumerate(usage.cpu())}
    logging.info(f"[{tag}] usage={usage_dict}")
    kth = max(1, int(math.ceil(0.9 * qmax.numel())))
    logging.info(f"[{tag}] Qmax mean={qmax.mean().item():.3f} med={qmax.median().item():.3f} p90={qmax.kthvalue(kth)[0]:.3f}")
    emb_norms = embed.norm(dim=1)
    center_norms = cent.norm(dim=1)
    info = (
        f"||emb|| mean={emb_norms.mean().item():.3f} min={emb_norms.min().item():.3f} max={emb_norms.max().item():.3f} | "
        f"||cent|| mean={center_norms.mean().item():.3f} min={center_norms.min().item():.3f} max={center_norms.max().item():.3f}"
    )
    if cent.size(0) > 1:
        pairwise = torch.pdist(cent)
        info += f" | pairwise_mean={pairwise.mean().item():.3f}"
    logging.info(f"[{tag}] {info}")


def _extract_sample_names(data) -> Optional[list]:
    if data is None:
        return None
    names_attr = getattr(data, 'sample_name', None)
    candidates = []
    if names_attr is not None:
        if isinstance(names_attr, (list, tuple)):
            candidates.extend([str(n) for n in names_attr if n is not None])
        else:
            candidates.append(str(names_attr))
    elif hasattr(data, 'misc_info'):
        info = data.misc_info
        if isinstance(info, dict):
            name = info.get('sample_name')
            if name:
                candidates.append(str(name))
        elif isinstance(info, (list, tuple)):
            for item in info:
                if isinstance(item, dict):
                    nm = item.get('sample_name')
                    if nm:
                        candidates.append(str(nm))
                elif item is not None:
                    candidates.append(str(item))
    return candidates if candidates else None


def _log_nonfinite_tensor(stage: str,
                          tensor_name: str,
                          tensor: Optional[torch.Tensor],
                          data_ref=None,
                          extra: Optional[str] = None) -> bool:
    if tensor is None:
        return False
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return False
    check = tensor.detach()
    if not torch.is_floating_point(check):
        check = check.float()
    mask = ~torch.isfinite(check)
    if mask.any():
        idxs = mask.nonzero(as_tuple=False).view(-1).tolist()
        sample_names = _extract_sample_names(data_ref)
        values = check.flatten()[mask.flatten()].cpu().tolist()
        logging.warning(
            f"[{stage}] Non-finite {tensor_name} detected at indices {idxs}; "
            f"values={values[:5]}{'...' if len(values) > 5 else ''}; samples={sample_names}; {extra or ''}"
        )
        return True
    return False


# def cox_negative_log_likelihood(risk_scores: torch.Tensor,
#                                 durations: torch.Tensor,
#                                 events: torch.Tensor) -> torch.Tensor:
#     risk = risk_scores.view(-1)
#     time = durations.view(-1)
#     event = events.view(-1)
#     if risk.numel() == 0:
#         return torch.tensor(0.0, device=risk_scores.device, requires_grad=risk_scores.requires_grad)
#     order = torch.argsort(time, descending=True)
#     sorted_risk = risk[order]
#     sorted_event = event[order]
#     if sorted_event.sum() == 0:
#         return torch.tensor(0.0, device=risk_scores.device, requires_grad=risk_scores.requires_grad)
#     shift = torch.max(sorted_risk)
#     exp_risk = torch.exp(sorted_risk - shift)
#     cumsum_exp = torch.cumsum(exp_risk, dim=0)
#     log_cumsum = torch.log(cumsum_exp) + shift
#     losses = -(sorted_risk - log_cumsum) * sorted_event
#     return losses.sum() / (sorted_event.sum() + 1e-8)



def compute_c_index(risk_scores: torch.Tensor,
                    durations: torch.Tensor,
                    events: torch.Tensor) -> float:
    if risk_scores.numel() == 0:
        return 0.0
    try:
        preds = risk_scores.detach().cpu().numpy().reshape(-1)
        times = durations.detach().cpu().numpy().reshape(-1)
        ev    = events.detach().cpu().numpy().reshape(-1)
        mask  = np.isfinite(preds) & np.isfinite(times) & np.isfinite(ev)
        if mask.sum() < 2:
            return 0.0
        # concordance_index expects higher values to indicate longer survival, hence the negation
        return float(concordance_index(times[mask], -preds[mask], ev[mask].astype(bool)))
    except Exception as exc:
        logging.warning(f"C-index computation failed: {exc}")
        return 0.0


def plot_km_curves(times: np.ndarray,
                   events: np.ndarray,
                   risks: np.ndarray,
                   output_path: str,
                   threshold: float = None,
                   strategy: str = 'cindex',   # 'cindex' or 'logrank'
                   search_qrange: tuple = (0.1, 0.9),  # quantile range for searching thresholds
                   n_grid: int = 100,          # number of grid points when searching thresholds
                   min_group_size: int = 10    # minimum samples per group after splitting
                   ) -> Optional[tuple]:
    """Plot Kaplan-Meier curves using a risk threshold discovered via C-index/log-rank search.

    Returns (output_path, best_threshold, best_cindex, p_value, n_high, n_low) if plotting succeeds,
    otherwise None.
    """
    # 1) Basic checks and cleaning
    if times.size == 0 or np.nan_to_num(events, nan=0).sum() == 0:
        logging.warning("KM plot skipped: insufficient events.")
        return None

    times  = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    risks  = np.asarray(risks, dtype=float)

    mask = np.isfinite(times) & np.isfinite(events) & np.isfinite(risks)
    if mask.sum() < 2:
        logging.warning("KM plot skipped: not enough finite samples after masking.")
        return None
    times, events, risks = times[mask], events[mask], risks[mask]

    # 2) Determine the threshold
    best_thr = threshold
    best_cidx = None
    best_lr_stat = None
    best_p = None

    from lifelines.utils import concordance_index
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt
    import os

    def _try_thr(t):
        high = risks >= t
        low  = ~high
        if high.sum() < min_group_size or low.sum() < min_group_size:
            return None
        # Use binary high/low labels to compute a C-index for comparison
        cidx = concordance_index(times,  high.astype(float), events.astype(bool))
        # Also compute the log-rank statistic for the logrank strategy
        try:
            lr = logrank_test(times[high], times[low],
                              event_observed_A=events[high].astype(bool),
                              event_observed_B=events[low].astype(bool))
            lr_stat, p = float(lr.test_statistic), float(lr.p_value)
        except Exception:
            lr_stat, p = np.nan, np.nan
        return cidx, lr_stat, p, int(high.sum()), int(low.sum())

    if best_thr is None:
        ql, qh = search_qrange
        ql = max(0.0, min(ql, 1.0))
        qh = max(0.0, min(qh, 1.0))
        if qh <= ql:
            ql, qh = 0.1, 0.9
        lo, hi = np.quantile(risks, ql), np.quantile(risks, qh)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.min(risks), np.max(risks)

        # Build candidate thresholds within the quantile range
        cand = np.unique(np.quantile(risks, np.linspace(ql, qh, n_grid)))
        cand = cand[np.isfinite(cand)]
        cand = cand[(cand > lo) & (cand < hi)]
        if cand.size == 0:
            cand = np.unique(risks)
            cand = cand[(cand > lo) & (cand < hi)]

        # Evaluate each candidate threshold
        for t in cand:
            res = _try_thr(t)
            if res is None:
                continue
            cidx, lr_stat, p, nh, nl = res
            if strategy == 'cindex':
                better = (best_cidx is None) or (cidx > best_cidx + 1e-12) or \
                         (abs(cidx - (best_cidx or 0)) <= 1e-12 and abs(nh - nl) < abs((res[3]) - (res[4])))
                if better:
                    best_thr, best_cidx, best_lr_stat, best_p, best_nh, best_nl = t, cidx, lr_stat, p, nh, nl
            elif strategy == 'logrank':
                # Select the threshold with the strongest log-rank statistic (smallest p)
                better = (best_lr_stat is None) or (lr_stat > best_lr_stat + 1e-12) or \
                         (abs(lr_stat - (best_lr_stat or 0)) <= 1e-12 and abs(nh - nl) < abs((res[3]) - (res[4])))
                if better:
                    best_thr, best_cidx, best_lr_stat, best_p, best_nh, best_nl = t, cidx, lr_stat, p, nh, nl
            else:
                raise ValueError("strategy must be 'cindex' or 'logrank'")
    else:
        # Use the externally supplied threshold (e.g., from the training set)
        res = _try_thr(best_thr)
        if res is None:
            logging.warning("Given threshold yields empty/small groups. KM skipped.")
            return None
        best_cidx, best_lr_stat, best_p, best_nh, best_nl = res

    if best_thr is None:
        logging.warning("No valid threshold found for KM.")
        return None

    # 3) Plot KM curves with the best threshold
    high_mask = risks >= best_thr
    low_mask  = ~high_mask

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        logging.warning("KM plot skipped: one risk group empty.")
        return None

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    kmf_high.fit(times[high_mask], events[high_mask].astype(bool), label=f'High risk (n={int(high_mask.sum())})').plot(ax=ax)
    kmf_low.fit(times[low_mask], events[low_mask].astype(bool),  label=f'Low risk (n={int(low_mask.sum())})').plot(ax=ax)

    # 4) Estimate hazard ratio (optional)
    hr_txt = ""
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
        df = pd.DataFrame({
            "time": times,
            "event": events.astype(int),
            "group": high_mask.astype(int)
        })
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time", event_col="event")
        hr = float(np.exp(cph.params_["group"]))
        hr_ci = cph.confidence_intervals_.loc["group"].values  # log(HR) interval
        hr_ci = np.exp(hr_ci)
        hr_txt = f" | HR={hr:.2f} [{hr_ci[0]:.2f}, {hr_ci[1]:.2f}]"
    except Exception:
        pass  # lifelines CoxPHFitter missing or insufficient data

    # 5) Title/grid/save
    p_value = best_p if (best_p is not None and np.isfinite(best_p)) else np.nan
    title = (f"KM by best {strategy} threshold\n"
             f"thr={best_thr:.6g} | C-index={best_cidx:.3f} | p={p_value:.3g}{hr_txt}")
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival probability')
    ax.grid(True, linestyle='--', alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path, float(best_thr), float(best_cidx), float(p_value) if np.isfinite(p_value) else np.nan, int(high_mask.sum()), int(low_mask.sum())



# === Discrete-time survival: time binning & loss ===
def make_time_bins(times: np.ndarray, K: int = 20, method: str = "quantile"):
    times = np.asarray(times, float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        # Fallback to a dummy range when no finite values are available
        return np.linspace(0.0, 1.0, K+1)
    if method == "quantile":
        edges = np.unique(np.quantile(times, np.linspace(0, 1, K+1)))
        if edges.size <= 2:
            edges = np.linspace(times.min(), times.max() + 1e-8, K+1)
    else:
        edges = np.linspace(times.min(), times.max() + 1e-8, K+1)
    return edges

@torch.no_grad()
def search_bin_id(durations: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Return the time-bin index in [0, K-1] for each duration."""
    # Equivalent to np.searchsorted(edges, t, 'right') - 1, then clipped to [0, K-1]
    K = edges.numel() - 1
    t = durations.view(-1, 1)  # [B,1]
    e = edges.view(1, -1)      # [1,K+1]
    # Count how many edge values are strictly less than t
    bin_id = (t > e[:, :-1]).sum(dim=1) - 1
    return bin_id.clamp_(0, K-1)

def discrete_time_nll(logits: torch.Tensor,
                      durations: torch.Tensor,
                      events: torch.Tensor,
                      edges: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for discrete-time survival hazards."""
    B, K = logits.shape
    hazards = torch.sigmoid(logits)
    bin_id = search_bin_id(durations, edges)

    arangeK = torch.arange(K, device=logits.device).view(1, K)
    pre_mask = (arangeK < bin_id.view(-1,1)).float()
    at_mask  = torch.nn.functional.one_hot(bin_id, K).float()
    le_mask  = (arangeK <= bin_id.view(-1,1)).float()

    eps = 1e-7
    log1m = torch.log(1. - hazards + eps)
    logh  = torch.log(hazards + eps)

    ev = events.view(-1,1).float()
    loglik_event = (pre_mask * log1m).sum(dim=1) + (at_mask * logh).sum(dim=1)
    loglik_cens  = (le_mask * log1m).sum(dim=1)

    loglik = ev.squeeze()*loglik_event + (1 - ev.squeeze())*loglik_cens
    return -(loglik.mean())

def hazards_to_risk(logits: torch.Tensor, mode: str = "cumhaz") -> torch.Tensor:
    hazards = torch.sigmoid(logits)  # [B,K]
    eps = 1e-6
    if mode == "cuminc":
        S = torch.cumprod(1. - hazards, dim=1)
        return 1. - S[:, -1]
    elif mode == "cumhaz":  # Recommended: more resolution, monotonic equivalent to -log(S(T_end))
        return torch.sum(-torch.log(1. - hazards + eps), dim=1)
    elif mode == "exp_time":
        K = hazards.size(1)
        S = torch.cumprod(1. - hazards, dim=1)
        f = hazards * torch.cat([torch.ones_like(hazards[:, :1]), S[:, :-1]], dim=1)
        t_idx = torch.arange(K, device=logits.device, dtype=logits.dtype).view(1, K)
        return -(f * t_idx).sum(dim=1)
    else:
        raise ValueError("mode must be 'cuminc', 'cumhaz', or 'exp_time'")





class ProcessedImageDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 nucle_root,
                 gene_root,
                 svs_root,
                 label_root=None,
                 original_patch_size_level0=512,
                 transform=None,
                 pre_transform=None):
        self.nucle_raw_root = nucle_root
        self.gene_raw_root = gene_root
        self.svs_raw_root = svs_root
        self.original_patch_size_level0 = original_patch_size_level0
        self.labels = None
        if label_root is not None:
            try:
                self.labels = pd.read_csv(label_root)
            except FileNotFoundError:
                logging.warning("Label file %s not found; proceeding without survival annotations.", label_root)
            except Exception as exc:
                logging.warning("Failed to read label file %s: %s. Proceeding without survival annotations.", label_root, exc)
        self._global_seed = 233
        self._cg_epoch = 0

        type_categories = [['Connective', 'Neoplastic', 'Dead', 'Epithelial', 'Inflammatory']]
        try:
            self.type_encoder = OneHotEncoder(sparse_output=False, categories=type_categories)
        except TypeError:
            self.type_encoder = OneHotEncoder(sparse=False, categories=type_categories)

        # === Cross-modality (cell-gene) sharding / lazy-loading parameters ===
        self.cg_radius = self.original_patch_size_level0 * 2.5   # cross-modal radius
        self.cg_cells_per_shard = 20_000                         # cells processed per shard
        self.cg_shard_root = os.path.join(root, "cg_shards")
        os.makedirs(self.cg_shard_root, exist_ok=True)

        self.cg_max_edges_per_sample = 800_000  # cap CG edges per sample (down-sample beyond this)
        self.cg_load_frac = 0.8                 # randomly subsample CG edges if <1.0
        self.load_edges_on_get = True          # lazily load CG edges in __getitem__
        self._cg_force_full = False            # force all edges during validation/inference

        

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

        self._ensure_survival_attributes()
        self._ensure_sample_name_attribute()


    def _lookup_survival(self, sample_name: Optional[str]) -> Optional[Tuple[float, int]]:
        if sample_name is None or self.labels is None:
            return None
        try:
            row = self.labels.loc[self.labels["PatientID"] == sample_name[:12]]
            if row.empty:
                return None
            T_raw = pd.to_numeric(row["T"].iloc[0], errors='coerce')
            E_raw = pd.to_numeric(row["E"].iloc[0], errors='coerce')
            if pd.isna(T_raw) or pd.isna(E_raw):
                return None
            time = float(T_raw) / 365.0
            event = int(E_raw)
            if not np.isfinite(time) or not np.isfinite(event):
                return None
            return time, event
        except Exception:
            return None

    def _ensure_sample_name_attribute(self):
        if not hasattr(self, 'data_list') or not self.data_list:
            return
        for data in self.data_list:
            if data is None:
                continue
            if hasattr(data, 'sample_name') and data.sample_name is not None:
                continue
            if hasattr(data, 'misc_info') and isinstance(data.misc_info, dict):
                name = data.misc_info.get('sample_name')
                if name:
                    data.sample_name = name

    def _build_cg_edges_to_shards(self, pos_cells: torch.Tensor, pos_genes: torch.Tensor,
                              radius: float, cells_per_shard: int, shard_dir: str,
                              shard_prefix: str = "cg"):
        """Build cell→gene radius edges by sharding cells and writing raw-distance .pt files."""
        os.makedirs(shard_dir, exist_ok=True)
        cells_np = pos_cells.detach().cpu().numpy()
        genes_np = pos_genes.detach().cpu().numpy()
        tree = cKDTree(genes_np)

        N = cells_np.shape[0]
        total_edges = 0
        step = int(cells_per_shard) if (cells_per_shard is not None and int(cells_per_shard) > 0) else N

        for start in range(0, N, step):
            end = min(start + step, N)
            block = cells_np[start:end]
            neigh_lists = tree.query_ball_point(block, r=radius)

            src_idx, dst_idx, dlist = [], [], []
            for i, neigh in enumerate(neigh_lists):
                if not neigh: 
                    continue
                src = start + i
                genes = np.asarray(neigh, dtype=np.int64)
                diff = genes_np[genes] - cells_np[src]
                dist = np.sqrt((diff * diff).sum(axis=1)).astype(np.float32)
                src_idx.append(np.full(genes.shape[0], src, dtype=np.int64))
                dst_idx.append(genes)
                dlist.append(dist)

            if src_idx:
                src_idx = np.concatenate(src_idx)
                dst_idx = np.concatenate(dst_idx)
                dlist = np.concatenate(dlist)[:, None]
                edge_index = torch.from_numpy(np.stack([src_idx, dst_idx], axis=0)).long().contiguous()
                payload = {
                    "edge_index": edge_index,
                    "edge_attr_raw": torch.from_numpy(dlist).contiguous(),
                    "meta": {"block_start": start, "block_end": end, "radius": radius,
                            "N_cells": int(N), "N_genes": int(genes_np.shape[0])}
                }
                shard_path = os.path.join(shard_dir, f"{shard_prefix}_{start:09d}.pt")
                torch.save(payload, shard_path)
                total_edges += edge_index.shape[1]
        logging.info(f"[CG-SHARD] total edges={total_edges:,} dir={shard_dir}")



    def _normalize_cg_shards_inplace(self, shard_dir: str, shard_prefix: str = "cg"):
        files = sorted(glob.glob(os.path.join(shard_dir, f"{shard_prefix}_*.pt")))
        if not files:
            logging.warning(f"[CG-NORM] no shards in {shard_dir}")
            return

        sample_min = float("inf")
        sample_max = float("-inf")

        # First pass: record per-sample min/max raw distances across shards
        for f in files:
            obj = torch.load(f, map_location="cpu")
            d = obj.get("edge_attr_raw", None)
            if d is None or d.numel() == 0:
                continue
            dm, dM = float(d.min()), float(d.max())
            if np.isfinite(dm): sample_min = min(sample_min, dm)
            if np.isfinite(dM): sample_max = max(sample_max, dM)

        # Fallback: if distances are invalid or degenerate, set all edge_attr to 1.0
        if not np.isfinite(sample_min) or not np.isfinite(sample_max) or sample_max <= sample_min:
            for f in files:
                obj = torch.load(f, map_location="cpu")
                d = obj.get("edge_attr_raw", None)
                if d is None or d.numel() == 0:
                    continue
                obj["edge_attr"] = torch.ones_like(d)  # degenerate fallback: distance ignored
                torch.save(obj, f)
            logging.warning(f"[CG-NORM] invalid per-sample min/max in {shard_dir}: min={sample_min}, max={sample_max}. set all edge_attr=1.0")
            return

        # Second pass: normalize distances per sample so near edges approach 1 and far edges 0
        scale = sample_max - sample_min
        for f in files:
            obj = torch.load(f, map_location="cpu")
            d = obj.get("edge_attr_raw", None)
            if d is None or d.numel() == 0:
                continue
            obj["edge_attr"] = 1.0 - ((d - sample_min) / scale)
            torch.save(obj, f)

        # Optionally record metadata describing the per-sample normalization range
        try:
            import json
            with open(os.path.join(shard_dir, "meta.json"), "w", encoding="utf-8") as fw:
                json.dump({"norm_scope": "per-sample", "min": sample_min, "max": sample_max}, fw)
        except Exception:
            pass

        logging.info(f"[CG-NORM] normalized {len(files)} shards (per-sample) with min={sample_min:.4f}, max={sample_max:.4f}")




    def _load_cg_edges_from_shards(self, shard_dir: str,
                               max_edges: int = None, frac: float = 1.0, seed: Optional[int] = None):
        files = sorted(glob.glob(os.path.join(shard_dir, "cg_*.pt")))
        if not files:
            empty_ei = torch.empty((2,0),dtype=torch.long)
            empty_ea = torch.empty((0,1),dtype=torch.float)
            return empty_ei, empty_ea, empty_ei, empty_ea
        rng = np.random.default_rng(seed)
        ei_list, ea_list = [], []
        for f in files:
            obj = torch.load(f, map_location="cpu")
            ei = obj["edge_index"]
            ea = obj.get("edge_attr", obj.get("edge_attr_raw", None))
            if ea is None: 
                continue
            E = ei.shape[1]
            if frac < 1.0 and E > 0:
                keep = rng.random(E) < frac
                keep = torch.from_numpy(keep.astype(np.bool_))
                ei = ei[:, keep]
                ea = ea[keep]
            ei_list.append(ei); ea_list.append(ea)

        if not ei_list:
            empty_ei = torch.empty((2,0),dtype=torch.long)
            empty_ea = torch.empty((0,1),dtype=torch.float)
            return empty_ei, empty_ea, empty_ei, empty_ea

        ei = torch.cat(ei_list, dim=1)
        ea = torch.cat(ea_list, dim=0)

        if max_edges is not None and ei.shape[1] > max_edges:
            perm = torch.randperm(ei.shape[1])[:max_edges]
            ei = ei[:, perm]; ea = ea[perm]

        ei_gc = torch.stack([ei[1], ei[0]], dim=0).contiguous()  # gene→cell
        ea_gc = ea.clone()
        return ei.contiguous(), ea.contiguous(), ei_gc, ea_gc

    def set_cg_sampling_mode(self, mode: str):
        if mode == 'train':
            self._cg_force_full = False
        elif mode in {'eval', 'val', 'test'}:
            self._cg_force_full = True
        else:
            raise ValueError(f"Unknown CG sampling mode: {mode}")

    def get_cg_sampling_mode(self) -> str:
        return 'eval' if self._cg_force_full else 'train'


    def _ensure_survival_attributes(self):
        if not hasattr(self, 'data_list') or self.data_list is None:
            return
        for data in self.data_list:
            sample_name = None
            if hasattr(data, 'misc_info') and isinstance(data.misc_info, dict):
                sample_name = data.misc_info.get('sample_name')
            elif hasattr(data, 'sample_name'):
                sample_name = data.sample_name

            t_tensor = None
            e_tensor = None
            if hasattr(data, 't') and data.t is not None:
                t_tensor = data.t if torch.is_tensor(data.t) else torch.tensor([float(data.t)], dtype=torch.float)
            if hasattr(data, 'e') and data.e is not None:
                e_tensor = data.e if torch.is_tensor(data.e) else torch.tensor([float(data.e)], dtype=torch.float)

            needs_lookup = (
                t_tensor is None or e_tensor is None or
                (t_tensor is not None and not torch.isfinite(t_tensor).all()) or
                (e_tensor is not None and not torch.isfinite(e_tensor).all())
            )
            if needs_lookup:
                lookup = self._lookup_survival(sample_name)
                if lookup is None:
                    continue
                time, event = lookup
                t_tensor = torch.tensor([time], dtype=torch.float)
                e_tensor = torch.tensor([event], dtype=torch.float)

            data.t = t_tensor
            data.e = e_tensor


    @property
    def raw_file_names(self):
        svs_files_found = glob.glob(os.path.join(self.svs_raw_root, "*.svs"))
        sample_names = [os.path.basename(f).replace(".svs", "") for f in svs_files_found]
        # print(f"DEBUG(raw_file_names): SVS files found by glob: {svs_files_found}")
        # print(f"DEBUG(raw_file_names): Derived sample names: {sample_names}")
        return sample_names

    @property
    def processed_file_names(self):
        return ['processed_graphs_final_vis.pt']  # processed dataset filename

    def process(self):
        print(f"DEBUG(process): Entering process method.")

        raw_names = self.raw_file_names
        print(f"DEBUG(process): Raw file names for processing: {raw_names}")

        if not raw_names:
            print("WARNING(process): No raw files. Saving empty data.")
            data, slices = self.collate([])
            torch.save((data, slices), self.processed_paths[0])
            return

        all_graph_data_list = []
        for sample_idx, sample_name in enumerate(tqdm(raw_names, desc="Processing Samples")):
            svs_path   = os.path.join(self.svs_raw_root,   sample_name + ".svs")
            nucle_path = os.path.join(self.nucle_raw_root, sample_name, "full_slide_nuclei_features_sliding_window_debug", "all_nuclei_features_full_slide_sliding_window_robust_scaled.csv")
            gene_path  = os.path.join(self.gene_raw_root,  sample_name + ".csv")

            if not (os.path.exists(svs_path) and os.path.exists(nucle_path) and os.path.exists(gene_path)):
                print(f"WARNING: Missing files for {sample_name}. Skipping.")
                continue

            try:
                # --- Read WSI metadata to obtain level downsample ---
                slide = OpenSlide(svs_path)
                level = 1
                downsample_factor = float(slide.level_downsamples[level])
                slide.close()

                # === Cells ===
                nucle_data = pd.read_csv(nucle_path)
                # print(nucle_data.columns)

                x_cells_global = nucle_data["Identifier.CentoidX_Global"].values
                y_cells_global = nucle_data["Identifier.CentoidY_Global"].values
                pos_cells = torch.tensor(np.vstack((x_cells_global, y_cells_global)).T, dtype=torch.float)

                if 'type' in nucle_data.columns:
                    cell_types_encoded = self.type_encoder.fit_transform(nucle_data[['type']])
                    cell_types_tensor = torch.tensor(cell_types_encoded, dtype=torch.float)
                else:
                    cell_types_tensor = torch.empty((len(nucle_data), 0), dtype=torch.float)

                columns_to_drop_cells = [
                    "Identifier.CentoidX_Global",
                    "Identifier.CentoidY_Global",
                    "Global_Nuclei_ID", "type",
                    'Shape.HuMoments4','Shape.WeightedHuMoments4',
                    'Shape.HuMoments5','Shape.HuMoments6','Shape.HuMoments7',
                    'Shape.WeightedHuMoments6','Shape.WeightedHuMoments3',
                    'Shape.HuMoments3','Shape.HuMoments2','Shape.WeightedHuMoments2'
                ]
                cell_features_df = nucle_data.drop(columns=columns_to_drop_cells, errors='ignore')
                for col in cell_features_df.columns:
                    cell_features_df[col] = pd.to_numeric(cell_features_df[col], errors='coerce').astype(float)
                cell_features_df = cell_features_df.fillna(0.0)

                x_cells_base = torch.tensor(cell_features_df.values, dtype=torch.float)
                x_cells = torch.cat([x_cells_base, cell_types_tensor], dim=1)

                # === Genes ===
                gene_data = pd.read_csv(gene_path)
                gene_coord_col = gene_data.columns[0]
                gene_coords_parsed = gene_data[gene_coord_col].str.split('_', expand=True)\
                                            .apply(pd.to_numeric, errors='coerce').fillna(0.0)
                x_gene = gene_coords_parsed.iloc[:,0].values if gene_coords_parsed.shape[1] >= 1 else np.zeros(len(gene_data))
                y_gene = gene_coords_parsed.iloc[:,1].values if gene_coords_parsed.shape[1] >= 2 else np.zeros(len(gene_data))

                # Align tile coordinates with global cell coordinates via the downsample factor
                pos_genes = torch.tensor(np.vstack((x_gene * downsample_factor, y_gene * downsample_factor)).T,
                                        dtype=torch.float)

                gene_features_df = gene_data.drop(columns=[gene_coord_col])
                for col in gene_features_df.columns:
                    gene_features_df[col] = pd.to_numeric(gene_features_df[col], errors='coerce').astype(float)
                gene_features_df = gene_features_df.fillna(0.0)
                x_genes = torch.tensor(gene_features_df.values, dtype=torch.float)

                # --- Intra-modal Delaunay edges (convert distances so nearer edges approach 1) ---
                edge_index_cells, edge_attr_cells = self._create_delaunay_edges(pos_cells)
                edge_index_genes, edge_attr_genes = self._create_delaunay_edges(pos_genes)

                # --- Cross-modal CG edges: write shards to disk and normalize globally ---
                sample_shard_dir = os.path.join(self.cg_shard_root, sample_name)
                self._build_cg_edges_to_shards(
                    pos_cells=pos_cells,
                    pos_genes=pos_genes,
                    radius=self.cg_radius,
                    cells_per_shard=self.cg_cells_per_shard,
                    shard_dir=sample_shard_dir,
                    shard_prefix="cg"
                )
                self._normalize_cg_shards_inplace(sample_shard_dir, "cg")

                # --- Assemble hetero graph (CC/GG in-memory, CG edges loaded lazily) ---
                hetero_data = HeteroData()
                if x_cells.numel() > 0 and pos_cells.shape[0] > 0:
                    hetero_data['cell'].x = x_cells
                    hetero_data['cell'].pos = pos_cells
                if x_genes.numel() > 0 and pos_genes.shape[0] > 0:
                    hetero_data['gene'].x = x_genes
                    hetero_data['gene'].pos = pos_genes

                if edge_index_cells.numel() > 0:
                    hetero_data['cell', 'c_c', 'cell'].edge_index = edge_index_cells
                if edge_attr_cells.numel() > 0:
                    hetero_data['cell', 'c_c', 'cell'].edge_attr  = edge_attr_cells

                if edge_index_genes.numel() > 0:
                    hetero_data['gene', 'g_g', 'gene'].edge_index = edge_index_genes
                if edge_attr_genes.numel() > 0:
                    hetero_data['gene', 'g_g', 'gene'].edge_attr  = edge_attr_genes

                # Survival information
                if self.labels is not None:
                    try:
                        surv_row = self.labels.loc[self.labels["PatientID"] == sample_name[:12]].iloc[0]
                        T_raw = pd.to_numeric(surv_row["T"], errors='coerce')
                        E_raw = pd.to_numeric(surv_row["E"], errors='coerce')
                        if pd.isna(T_raw) or pd.isna(E_raw):
                            logging.warning(f"Survival NaN for {sample_name}. Sample skipped.")
                            continue
                        time  = float(T_raw) / 365.0
                        event = int(E_raw)
                    except Exception:
                        logging.warning(f"Survival info missing for {sample_name}. Sample skipped.")
                        continue
                    if not np.isfinite(time) or not np.isfinite(event):
                        logging.warning(f"Invalid survival info for {sample_name}: T={time}, E={event}. Sample skipped.")
                        continue
                    hetero_data.t = torch.tensor([time], dtype=torch.float)
                    hetero_data.e = torch.tensor([event], dtype=torch.float)

                # Record the CG shard directory for lazy loading
                hetero_data.misc_info = {
                    'radius_threshold': self.cg_radius,
                    'sample_name': sample_name,
                    'cg_shard_dir': sample_shard_dir
                }
                hetero_data.sample_name = sample_name

                valid_nodes_exist = any(
                    hasattr(hetero_data[nt], 'num_nodes') and hetero_data[nt].num_nodes > 0
                    for nt in hetero_data.node_types
                )
                if not valid_nodes_exist:
                    print(f"WARNING: No valid nodes for {sample_name}. Skipping.")
                    continue

                all_graph_data_list.append(hetero_data)

            except Exception as e:
                print(f"ERROR processing {sample_name}: {e}")
                traceback.print_exc()
                continue

        if self.pre_filter is not None:
            all_graph_data_list = [d for d in all_graph_data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            all_graph_data_list = [self.pre_transform(d) for d in all_graph_data_list]

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
                    if max_dist > min_dist:
                        edge_attr = 1.0 - ((edge_attr_raw - min_dist) / (max_dist - min_dist))
                    else:
                        edge_attr = torch.ones_like(edge_attr_raw)
            except Exception as e: print(f"ERROR: Delaunay failed for shape {pos_nodes.shape}: {e}")
        return edge_index, edge_attr


    # def _create_radius_edges(self, pos_cells, pos_genes, radius_threshold):
    #     edge_index_c_g, edge_attr_c_g, edge_index_g_c, edge_attr_g_c = torch.empty((2,0),dtype=torch.long), torch.empty((0,1),dtype=torch.float), torch.empty((2,0),dtype=torch.long), torch.empty((0,1),dtype=torch.float)
    #     if len(pos_cells) > 0 and len(pos_genes) > 0:
    #         dist_matrix = torch.cdist(pos_cells, pos_genes, p=2)
    #         adj = dist_matrix <= radius_threshold
    #         cell_indices, gene_indices = adj.nonzero(as_tuple=True)
    #         if cell_indices.numel() > 0:
    #             edge_index_c_g = torch.stack([cell_indices, gene_indices], dim=0)
    #             edge_attr_c_g_raw = dist_matrix[cell_indices, gene_indices].unsqueeze(1)
    #             if edge_attr_c_g_raw.numel() > 0:
    #                 min_d, max_d = edge_attr_c_g_raw.min(), edge_attr_c_g_raw.max()
    #                 if max_d > min_d:
    #                     edge_attr_c_g = 1.0 - ((edge_attr_c_g_raw - min_d) / (max_d - min_d))
    #                 else:
    #                     edge_attr_c_g = torch.ones_like(edge_attr_c_g_raw)
    #             edge_index_g_c = torch.stack([gene_indices, cell_indices], dim=0)
    #             edge_attr_g_c = edge_attr_c_g
    #     return edge_index_c_g, edge_attr_c_g, edge_index_g_c, edge_attr_g_c


    # Standard InMemoryDataset len and get
    def len(self):
        # Correct way to get length for InMemoryDataset
        if hasattr(self, 'slices') and self.slices is not None:
            if 't' in self.slices:
                return len(self.slices['t']) - 1
            if 'e' in self.slices:
                return len(self.slices['e']) - 1
            # Fallback: iterate through slices to find a representative one
            for key in self.slices:
                 if isinstance(self.slices[key], torch.Tensor) and self.slices[key].ndim == 1:
                     return len(self.slices[key]) -1
        return 0 # Should not happen if data is loaded/processed correctly

    def get(self, idx):
        data = super().get(idx)

        if not self.load_edges_on_get:
            return data

        try:
            shard_dir = None
            if hasattr(data, 'misc_info') and isinstance(data.misc_info, dict):
                shard_dir = data.misc_info.get('cg_shard_dir')
            if shard_dir and os.path.isdir(shard_dir):
                if self._cg_force_full:
                    frac = 1.0
                    max_edges = None
                    seed_for_this_get = None
                else:
                    frac = self.cg_load_frac
                    max_edges = self.cg_max_edges_per_sample
                    # Deterministically sample edges based on (seed, epoch, sample index)
                    base = getattr(self, '_global_seed', 233)
                    ep   = getattr(self, '_cg_epoch', 0)
                    seed_for_this_get = (hash((int(base), int(ep), int(idx))) & 0xFFFFFFFF)

                ei_cg, ea_cg, ei_gc, ea_gc = self._load_cg_edges_from_shards(
                    shard_dir,
                    max_edges=max_edges,
                    frac=frac,
                    seed=seed_for_this_get
                )
                if ei_cg.numel() > 0:
                    data['cell','c_g','gene'].edge_index = ei_cg
                    data['cell','c_g','gene'].edge_attr  = ea_cg
                if ei_gc.numel() > 0:
                    data['gene','g_c','cell'].edge_index = ei_gc
                    data['gene','g_c','cell'].edge_attr  = ea_gc
        except Exception as e:
            logging.warning(f"[GET] lazy-load CG shards failed at idx={idx}: {e}")

        return data





# --- 3. Visualization helper ---
def visualize_hetero_graph(data, sample_name=None, save_dir="./graph_visualizations", show_plot=False):
    """Visualize a single heterogenous graph (used for debugging)."""
    data = ensure_logdata(data)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    name = sample_name or (data.misc_info.get('sample_name') if hasattr(data, 'misc_info') and isinstance(data.misc_info, dict) else 'unknown')

    pos_cells_np = None
    if 'cell' in data.node_types and getattr(data['cell'], 'pos', None) is not None and data['cell'].num_nodes > 0:
        pos_cells_np = data['cell'].pos.cpu().numpy()
        ax.scatter(pos_cells_np[:, 0], pos_cells_np[:, 1], s=1, c='#4daf4a', label=f'Cells ({data["cell"].num_nodes})', alpha=0.6, edgecolors='k', linewidths=0.3)
    else:
        logging.info(f"Visualization: no cell nodes for {name}")

    pos_genes_np = None
    if 'gene' in data.node_types and getattr(data['gene'], 'pos', None) is not None and data['gene'].num_nodes > 0:
        pos_genes_np = data['gene'].pos.cpu().numpy()
        ax.scatter(pos_genes_np[:, 0], pos_genes_np[:, 1], s=40, c='#377eb8', marker='^', label=f'Genes ({data["gene"].num_nodes})', alpha=0.7, edgecolors='k', linewidths=0.3)
    else:
        logging.info(f"Visualization: no gene nodes for {name}")

    if ('cell', 'c_c', 'cell') in data.edge_types and pos_cells_np is not None:
        edges = data['cell', 'c_c', 'cell'].edge_index.cpu().numpy()
        for src, dst in edges.T:
            if src < dst:
                ax.plot([pos_cells_np[src, 0], pos_cells_np[dst, 0]],
                        [pos_cells_np[src, 1], pos_cells_np[dst, 1]],
                        color='gray', alpha=0.1, linewidth=0.3)

    if ('gene', 'g_g', 'gene') in data.edge_types and pos_genes_np is not None:
        edges = data['gene', 'g_g', 'gene'].edge_index.cpu().numpy()
        for src, dst in edges.T:
            if src < dst:
                ax.plot([pos_genes_np[src, 0], pos_genes_np[dst, 0]],
                        [pos_genes_np[src, 1], pos_genes_np[dst, 1]],
                        color='#f781bf', alpha=0.2, linewidth=0.4)

    if ('cell', 'c_g', 'gene') in data.edge_types and pos_cells_np is not None and pos_genes_np is not None:
        edges = data['cell', 'c_g', 'gene'].edge_index.cpu().numpy()
        for src, dst in edges.T:
            ax.plot([pos_cells_np[src, 0], pos_genes_np[dst, 0]],
                    [pos_cells_np[src, 1], pos_genes_np[dst, 1]],
                    color='#ff7f00', alpha=0.05, linewidth=0.2)

    ax.set_title(f"Heterogeneous Graph: {name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(loc='best')

    vis_output_path = os.path.join(save_dir, f'graph_vis_{name.replace(os.sep, "_")}.png')
    plt.tight_layout()
    plt.savefig(vis_output_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()
    logging.info(f"Graph visualization saved to {vis_output_path}")


# --- 4. Training and evaluation routines ---
def train_epoch(model, loader, optimizer, criterion_primary, aug, lambda_clust, device, epoch,
                cluster_params=None, enable_center_updates=True):
    model.train()
    base_dataset = getattr(loader.dataset, 'dataset', loader.dataset)
    if hasattr(base_dataset, 'set_cg_sampling_mode'):
        base_dataset.set_cg_sampling_mode('train')
    setattr(base_dataset, '_cg_epoch', int(epoch))

    total_loss = total_primary_loss = total_cluster_loss = 0.0
    risk_collector, time_collector, event_collector = [], [], []

    for data in tqdm(loader, desc="Training"):
        if aug is not None:
            data = aug(data)
        data = ensure_logdata(data).to(device)
        if not hasattr(data, 't') or not hasattr(data, 'e'):
            continue

        optimizer.zero_grad()
        hazard_logits, cluster_loss = model(
            data,
            compute_clustering_loss=True,
            clustering_loss_params=cluster_params,
            update_centers=enable_center_updates
        )
        log_archetype_usage(
            tag=f"train/cell/epoch{epoch:03d}",
            embeddings=model._last_cell_embeddings,
            centers=model.shared_cluster_centers,
            temperature=model.cluster_temperature,
            sample_prob=0.1
        )
        log_archetype_usage(
            tag=f"train/gene/epoch{epoch:03d}",
            embeddings=model._last_gene_embeddings,
            centers=model.shared_cluster_centers,
            temperature=model.cluster_temperature,
            sample_prob=0.1
        )
        stats = getattr(model, '_last_cluster_stats', None)
        if stats:
            def _fmt(x):
                import math
                try:
                    x = float(x)
                    return "-" if not math.isfinite(x) else f"{x:.4f}"
                except Exception:
                    return "-"
            ent = stats.get('entropy')
            bal = stats.get('balance_kl')
            mar = stats.get('margin')
            logits_std = stats.get('logits_std')
            center_delta = stats.get('center_delta')
            assign_conf = stats.get('assign_conf')
            logging.info(
                f"[epoch{epoch:03d}] entropy={_fmt(ent)} | balanceKL={_fmt(bal)} | margin={_fmt(mar)} "
                f"| logits_std={_fmt(logits_std)} | assign_conf={_fmt(assign_conf)} | center_delta={_fmt(center_delta)}"
            )

        times = data.t.to(device).view(-1)
        events = data.e.to(device).view(-1)
        extra = f"epoch={epoch}"

        _log_nonfinite_tensor('Train', 'hazard_logits', hazard_logits, data, extra)

        # Compute discrete-time NLL loss
        primary_loss = criterion_primary(hazard_logits, times, events)

        cluster_loss_value = cluster_loss if cluster_loss is not None else torch.tensor(0.0, device=device)
        loss = primary_loss + lambda_clust * cluster_loss_value
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        num_graphs = times.numel()
        total_loss += loss.item() * num_graphs
        total_primary_loss += primary_loss.item() * num_graphs
        total_cluster_loss += cluster_loss_value.item() * num_graphs

        # Collect scalar risks for C-index/threshold search (using cumulative incidence)
        risk_scalar = hazards_to_risk(hazard_logits.detach(), mode="cumhaz").view(-1)
        risk_collector.append(risk_scalar)
        time_collector.append(times.detach().cpu())
        event_collector.append(events.detach().cpu())


    if not risk_collector:
        return 0.0, 0.0, 0.0, 0.0

    dataset_size = len(loader.dataset) if hasattr(loader, 'dataset') else sum(t.numel() for t in time_collector)
    avg_loss = total_loss / max(dataset_size, 1)
    avg_primary_loss = total_primary_loss / max(dataset_size, 1)
    avg_cluster_loss = total_cluster_loss / max(dataset_size, 1)

    all_risk = torch.cat(risk_collector).view(-1)  # cumulative incidence risk; larger implies higher risk
    all_time = torch.cat(time_collector).view(-1)
    all_event = torch.cat(event_collector).view(-1)
    _log_nonfinite_tensor('Train', 'all_risk_scalar', all_risk, None, extra=f"epoch={epoch}")
    c_index = compute_c_index(all_risk, all_time, all_event)
    
    # Determine the best training threshold (by C-index) for later evaluation
    best_thr, _, nh, nl = choose_best_threshold_by_cindex(
        times=all_time.cpu().numpy(),
        events=all_event.cpu().numpy(),
        risks=all_risk.cpu().numpy(),
        search_qrange=(0.1, 0.9),
        n_grid=101,
        min_group_size=10
    )
    if best_thr is None:
        # Fallback to the median risk if threshold search failed
        best_thr = float(np.median(all_risk.cpu().numpy()))
        logging.warning(f"[Train][epoch {epoch}] No valid threshold found; fallback to median={best_thr:.6g}")
    else:
        logging.info(f"[Train][epoch {epoch}] Best threshold (C-index split) = {best_thr:.6g} | groups: high={nh}, low={nl}")

    return avg_loss, avg_primary_loss, avg_cluster_loss, c_index, best_thr


@torch.no_grad()
def evaluate(model, loader, criterion_primary, lambda_clust, device, epoch):
    model.eval()
    base_dataset = getattr(loader.dataset, 'dataset', loader.dataset)
    prev_mode = None
    if hasattr(base_dataset, 'set_cg_sampling_mode'):
        if hasattr(base_dataset, 'get_cg_sampling_mode'):
            prev_mode = base_dataset.get_cg_sampling_mode()
        base_dataset.set_cg_sampling_mode('eval')
    logits_list, times_list, events_list = [], [], []
    try:
        for data in tqdm(loader, desc="Evaluating"):
            data = ensure_logdata(data).to(device)
            if not hasattr(data, 't') or not hasattr(data, 'e'):
                continue

            logits, _ = model(data, compute_clustering_loss=False, update_centers=False)
            # Collect logits/time/event on CPU; concatenate once after the loop
            logits_list.append(logits.detach().cpu())
            times_list.append(data.t.view(-1).detach().cpu())
            events_list.append(data.e.view(-1).detach().cpu())
    finally:
        if hasattr(base_dataset, 'set_cg_sampling_mode'):
            if prev_mode is not None:
                base_dataset.set_cg_sampling_mode(prev_mode)
            else:
                base_dataset.set_cg_sampling_mode('train')

    if not logits_list:
        return 0.0, 0.0, 0.0, 0.0

    # Concatenate tensors once
    logits_all = torch.cat(logits_list, dim=0).to(device)     # [N, K]
    times_all  = torch.cat(times_list,  dim=0).to(device)     # [N]
    events_all = torch.cat(events_list, dim=0).to(device)     # [N]

    # Primary loss (discrete-time NLL); criterion already encodes the time bins
    avg_loss = criterion_primary(logits_all, times_all, events_all).item()

    # Use cumulative hazard risk for the C-index (larger => higher risk)
    risk_scalar = hazards_to_risk(logits_all, mode="cumhaz").view(-1)
    c_index = compute_c_index(risk_scalar, times_all, events_all)
    logging.info(f"[Eval][epoch{epoch:03d}] C-index={c_index:.4f}")

    # No clustering loss during evaluation
    return avg_loss, avg_loss, 0.0, c_index






def choose_best_threshold_by_cindex(times: np.ndarray,
                                    events: np.ndarray,
                                    risks: np.ndarray,
                                    search_qrange: tuple = (0.1, 0.9),
                                    n_grid: int = 100,
                                    min_group_size: int = 10):
    """Search for the risk threshold that maximizes the C-index when splitting into high/low groups."""
    from lifelines.utils import concordance_index

    # Clean inputs
    times  = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    risks  = np.asarray(risks, dtype=float)
    mask = np.isfinite(times) & np.isfinite(events) & np.isfinite(risks)
    if mask.sum() < max(2, 2*min_group_size) or np.nan_to_num(events, nan=0).sum() == 0:
        return None, None, 0, 0
    times, events, risks = times[mask], events[mask], risks[mask]

    # Build candidate thresholds from the quantile range
    ql, qh = search_qrange
    ql, qh = max(0.0, min(ql, 1.0)), max(0.0, min(qh, 1.0))
    if qh <= ql:
        ql, qh = 0.1, 0.9
    lo, hi = np.quantile(risks, ql), np.quantile(risks, qh)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.min(risks), np.max(risks)

    cand = np.unique(np.quantile(risks, np.linspace(ql, qh, n_grid)))
    cand = cand[np.isfinite(cand)]
    cand = cand[(cand > lo) & (cand < hi)]
    if cand.size == 0:
        # Fallback: use the middle portion of unique risk values
        uniq = np.unique(risks)
        cand = uniq[(uniq > lo) & (uniq < hi)]
        if cand.size == 0:
            return None, None, 0, 0

    best_thr, best_cidx, best_nh, best_nl = None, None, 0, 0
    # Dynamic minimum group size: at least `min_group_size` or 5% of samples
    dyn_min = max(min_group_size, int(0.05 * len(times)))

    for t in cand:
        high = risks >= t
        low  = ~high
        nh, nl = int(high.sum()), int(low.sum())
        if nh < dyn_min or nl < dyn_min:
            continue
        # Compute C-index using binary high/low predictions
        cidx = concordance_index(times,  high.astype(float), events.astype(bool))
        if (best_cidx is None) or (cidx > best_cidx + 1e-12) or \
           (abs(cidx - (best_cidx or 0)) <= 1e-12 and abs(nh - nl) < abs(best_nh - best_nl)):
            best_thr, best_cidx, best_nh, best_nl = float(t), float(cidx), nh, nl

    if best_thr is None:
        return None, None, 0, 0
    return best_thr, best_cidx, best_nh, best_nl







def build_discrete_time_criterion(edges_tensor: torch.Tensor):
    def _criterion(hazard_logits: torch.Tensor, durations: torch.Tensor, events: torch.Tensor):
        return discrete_time_nll(hazard_logits, durations, events, edges_tensor)
    return _criterion

def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser()
    cell_dir = args.cell_output_dir.expanduser()
    gene_dir = args.gene_dir.expanduser()
    svs_dir = args.svs_dir.expanduser()
    label_file = args.label_file.expanduser()
    graph_viz_dir = args.graph_viz_dir.expanduser() if args.graph_viz_dir else (root_dir / "graph_visualizations")

    root_dir.mkdir(parents=True, exist_ok=True)
    graph_viz_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='[%(asctime)s] %(levelname)s - %(message)s')
    _set_global_seed(args.seed)
    device = resolve_device(args.device, args.preferred_gpu)
    logging.info("Using device: %s", device)

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
        raise SystemExit(1)

    # Disable CG lazy loading before splitting so we can examine in-memory graphs only
    dataset.load_edges_on_get = False
    all_items = dataset.data_list
    valid_idx = []
    for i, g in enumerate(all_items):
        if not (hasattr(g, 't') and hasattr(g, 'e')):
            continue
        t_tensor = g.t if torch.is_tensor(g.t) else torch.tensor([g.t], dtype=torch.float)
        e_tensor = g.e if torch.is_tensor(g.e) else torch.tensor([g.e], dtype=torch.float)
        if torch.isfinite(t_tensor).all() and torch.isfinite(e_tensor).all():
            valid_idx.append(i)
        else:
            sample_name = getattr(g, 'sample_name', None)
            if sample_name is None and hasattr(g, 'misc_info') and isinstance(g.misc_info, dict):
                sample_name = g.misc_info.get('sample_name')
            logging.warning(
                f"Dropping sample {sample_name or i} due to non-finite survival values: "
                f"t={t_tensor.detach().cpu().tolist()}, e={e_tensor.detach().cpu().tolist()}"
            )
    if not valid_idx:
        logging.error("No samples with valid survival information. Exiting.")
        raise SystemExit(1)

    # Prepare labels for stratified splitting
    times  = np.asarray([float(all_items[i].t.item() if torch.is_tensor(all_items[i].t) else all_items[i].t)
                         for i in valid_idx], dtype=np.float32)
    events = np.asarray([int(all_items[i].e.item() if torch.is_tensor(all_items[i].e) else all_items[i].e)
                         for i in valid_idx], dtype=np.int32)


    # Stratified split (absolute indices relative to dataset)
    q = pd.qcut(times, q=4, duplicates='drop')     # quartile stratification
    y_strat = np.array([f"{int(e)}_{c}" for e, c in zip(events, q.codes)])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    rel_tr, rel_va = next(sss.split(times, y_strat))

    K_TIME_BINS = args.num_time_bins
    TIME_BIN_EDGES = make_time_bins(times[rel_tr], K=K_TIME_BINS, method=args.time_bin_method)
    # Move bin edges to device tensor
    time_bin_edges_t = torch.tensor(TIME_BIN_EDGES, dtype=torch.float, device=device)


    train_ids = np.asarray(valid_idx)[rel_tr]            # absolute dataset indices
    val_ids   = np.asarray(valid_idx)[rel_va]

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_ids)
    val_dataset   = Subset(dataset, val_ids)
    logging.info(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")

    # Re-enable CG lazy loading for training
    dataset.load_edges_on_get = True

    # Determine input channel counts using one training sample
    sample_graph = dataset.data_list[int(train_ids[0])]
    CELL_IN_CHANNELS = sample_graph['cell'].x.shape[1]
    GENE_IN_CHANNELS = sample_graph['gene'].x.shape[1]

    # DataLoaders (get() will lazily load CG edges)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if len(val_ids) else None

    # Optional: visualize one sample graph (would include CG edges once loaded)
    # sample_idx = int(train_ids[0] if len(train_ids) else val_ids[0])
    # sample_for_vis = dataset[sample_idx]
    # sample_name = sample_for_vis.misc_info.get('sample_name') if hasattr(sample_for_vis, 'misc_info') else None
    # visualize_hetero_graph(sample_for_vis, sample_name=sample_name, save_dir=str(graph_viz_dir), show_plot=False)

    aug = GraphAugmentations(rotation_prob=args.graph_augment_rotation_prob, flip_prob=args.graph_augment_flip_prob)

    # Model and training configuration
    HIDDEN_CHANNELS, EMBEDDING_DIM, OUT_CHANNELS = args.hidden_dim, args.embedding_dim, K_TIME_BINS
    num_shared_clusters = args.num_shared_clusters
    GNN_TYPE, NUM_ATTENTION_HEADS, DROPOUT_RATE = args.gnn_type, args.num_heads, args.dropout
    NUM_INTRA_MODAL_LAYERS, NUM_INTER_MODAL_LAYERS = args.num_intra_layers, args.num_inter_layers

    CLUSTER_TEMPERATURE = args.base_cluster_temperature
    CENTER_REPULSION_WEIGHT = args.center_repulsion
    CENTER_SEPARATION_SCALE = args.center_separation
    ASSIGNMENT_BALANCE_WEIGHT = args.assignment_balance

    model = HierarchicalMultiModalGNN(
        CELL_IN_CHANNELS,
        GENE_IN_CHANNELS,
        HIDDEN_CHANNELS,
        EMBEDDING_DIM,
        OUT_CHANNELS,
        num_shared_clusters,
        GNN_TYPE,
        NUM_ATTENTION_HEADS,
        DROPOUT_RATE,
        NUM_INTRA_MODAL_LAYERS,
        NUM_INTER_MODAL_LAYERS,
        num_time_bins=K_TIME_BINS,
        cluster_temperature=CLUSTER_TEMPERATURE,
        center_repulsion_weight=CENTER_REPULSION_WEIGHT,
        center_separation_scale=CENTER_SEPARATION_SCALE,
        assignment_balance_weight=ASSIGNMENT_BALANCE_WEIGHT
    ).to(device)

    LEARNING_RATE, NUM_EPOCHS = args.learning_rate, args.epochs
    base_params, center_params, readout_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('shared_cluster_centers'):
            center_params.append(param)
        elif name.startswith('readout_mlp'):
            readout_params.append(param)
        else:
            base_params.append(param)
    optim_groups = []
    if base_params:
        optim_groups.append({'params': base_params, 'lr': LEARNING_RATE})
    if center_params:
        optim_groups.append({'params': center_params, 'lr': LEARNING_RATE * 0.5})
    if readout_params:
        optim_groups.append({'params': readout_params, 'lr': LEARNING_RATE * 2.0})
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        min_lr=args.scheduler_min_lr,
        verbose=True,
    )
    criterion_primary = build_discrete_time_criterion(time_bin_edges_t)
    CLUSTER_WARMUP_EPOCHS = args.cluster_warmup_epochs
    CLUSTER_LAMBDA_MAX = args.cluster_lambda_max
    CLUSTER_LAMBDA_RAMP = args.cluster_lambda_ramp

    BASE_CLUSTER_PARAMS = {
        'temperature': CLUSTER_TEMPERATURE,
        'confidence_weight': args.base_confidence_weight,
        'balance_weight': ASSIGNMENT_BALANCE_WEIGHT,
        'margin_weight': args.base_margin_weight,
        'margin_delta': args.base_margin_delta,
        'alpha': 0.3,
    }
    CELL_CLUSTER_PARAMS = {
        'temperature': args.cell_cluster_temperature,
        'confidence_weight': args.cell_cluster_confidence,
    }
    GENE_CLUSTER_PARAMS = {
        'temperature': args.gene_cluster_temperature,
        'confidence_weight': args.gene_cluster_confidence,
    }
    CLUSTERING_LOSS_PARAMS = {
        'base': BASE_CLUSTER_PARAMS,
        'cell': CELL_CLUSTER_PARAMS,
        'gene': GENE_CLUSTER_PARAMS,
        'gene_update_stride': 2
    }
    
    best_val_c = -float('inf')
    best_model_path: Optional[Path] = None
    epochs_no_improve = 0
    early_stop_patience = args.early_stop_patience
    best_train_thr_for_best_model = None

    # Training loop
    logging.info(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        enable_center_updates = epoch > CLUSTER_WARMUP_EPOCHS
        base_alpha = 0.7 if enable_center_updates else 0.3
        CLUSTERING_LOSS_PARAMS['base']['alpha'] = base_alpha
        model._cluster_epoch = epoch

        if epoch <= CLUSTER_WARMUP_EPOCHS:
            lambda_clust = 0.0
        else:
            t = min(epoch - CLUSTER_WARMUP_EPOCHS, CLUSTER_LAMBDA_RAMP)
            lambda_clust = CLUSTER_LAMBDA_MAX * 0.5 * (1 - math.cos(math.pi * t / CLUSTER_LAMBDA_RAMP))

        train_loss, train_p_loss, train_c_loss, train_c_index, best_thr = train_epoch(
            model, train_loader, optimizer, criterion_primary, aug, lambda_clust, device, epoch,
            cluster_params=CLUSTERING_LOSS_PARAMS,
            enable_center_updates=enable_center_updates
        )
        logging.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} (Cox:{train_p_loss:.4f} Cluster:{train_c_loss:.4f})"
            f" | C-index: {train_c_index:.4f} | lambda_clust={lambda_clust:.3f}"
        )

        if val_loader is not None:
            val_loss, val_p_loss, val_c_loss, val_c_index = evaluate(
                model, val_loader, criterion_primary, lambda_clust, device, epoch
            )

            logging.info(
                f"Epoch {epoch:03d} | Val Loss: {val_loss:.4f} (Cox:{val_p_loss:.4f} Cluster:{val_c_loss:.4f})"
                f" | C-index: {val_c_index:.4f}"
            )
            scheduler.step(val_c_index)

            if val_c_index > best_val_c + 1e-4:
                best_val_c = val_c_index
                epochs_no_improve = 0
                best_model_path = root_dir / f"best_model_cox_{best_val_c:.4f}.pth"
                torch.save(model.state_dict(), best_model_path)
                best_train_thr_for_best_model = float(best_thr)
                logging.info("New best model saved with C-index %.4f -> %s", best_val_c, best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    logging.info("Early stopping triggered.")
                    break
        else:
            # Without validation data, drive LR scheduling via the training C-index
            scheduler.step(train_c_index)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(f"Training finished. Best Val C-index: {best_val_c if val_loader is not None else 'N/A'}")

    # Plot KM curves on the validation set using the best checkpoint
    if val_loader is not None and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        val_risks, val_times, val_events = [], [], []
        with torch.no_grad():
            for data in val_loader:
                data = ensure_logdata(data).to(device)
                if not hasattr(data, 't') or not hasattr(data, 'e'):
                    continue
                risk_scores, _ = model(data, compute_clustering_loss=False, update_centers=False)
                val_risks.append(risk_scores.detach().cpu())
                val_times.append(data.t.detach().cpu())
                val_events.append(data.e.detach().cpu())

        if val_risks:
            val_logits = torch.cat(val_risks)     # [N, K] logits
            risk_np  = hazards_to_risk(val_logits, mode="cumhaz").cpu().numpy().reshape(-1)
            time_np  = torch.cat(val_times).numpy().reshape(-1)
            event_np = torch.cat(val_events).numpy().reshape(-1)
            km_path = root_dir / "km_curve_validation.png"
            result_path = plot_km_curves(
                time_np,
                event_np.astype(int),
                risk_np,
                str(km_path),
                threshold=best_train_thr_for_best_model,
            )
            logging.info("KM curves saved to %s", result_path)
    elif val_loader is not None:
        logging.warning("Validation loader was available but no checkpoint was saved; skipping KM plot.")


if __name__ == '__main__':
    main()

"""Prototype analysis utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

__all__ = [
    "assign_prototypes_with_conf_dual",
    "apply_merge_to_cell_df",
    "build_merge_map_multi_signal",
]


def assign_prototypes_with_conf_dual(df, centers_tensor, tau: float = 0.4, store_soft: bool = False):
    embed_cols = [c for c in df.columns if str(c).startswith("embed_")]
    if df.empty or not embed_cols:
        return df
    X = df[embed_cols].to_numpy(dtype=np.float32, copy=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    C = centers_tensor.detach().float().cpu().numpy()
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    S = X @ C.T
    P = np.exp(S / max(1e-6, float(tau)))
    P /= P.sum(axis=1, keepdims=True) + 1e-8
    hard = P.argmax(axis=1).astype(int)
    entropy = -(P * np.log(P + 1e-12)).sum(axis=1) / np.log(P.shape[1])
    conf = 1.0 - entropy
    out = df.copy()
    out["prototype_id"] = hard.astype(np.int16)
    out["proto_conf"] = conf.astype(np.float32)
    if store_soft:
        for k in range(P.shape[1]):
            out[f"proto_p{k}"] = P[:, k].astype(np.float32)
    return out


def apply_merge_to_cell_df(cell_df, merge_map, overwrite_prototype_id: bool = False):
    """Attach a merged prototype column based on a mapping."""

    if cell_df.empty or "prototype_id" not in cell_df.columns:
        return cell_df

    out = cell_df.copy()
    pid_series = pd.to_numeric(out["prototype_id"], errors="coerce").astype("Int64")
    mapping = pd.Series(merge_map, name="merged", dtype="Int64")
    merged = pid_series.map(mapping)
    need_fill = merged.isna() & pid_series.notna()
    merged.loc[need_fill] = pid_series.loc[need_fill]
    out["prototype_id_merged"] = merged.astype("Int64")

    if overwrite_prototype_id:
        if "prototype_id_raw" not in out.columns:
            out["prototype_id_raw"] = out["prototype_id"]
        out["prototype_id"] = out["prototype_id_merged"]

    return out


def build_merge_map_multi_signal(
    cell_df: pd.DataFrame,
    centers_tensor: Optional[torch.Tensor] = None,
    gene_bulk_df: Optional[pd.DataFrame] = None,
    log2EN: Optional[pd.DataFrame] = None,
    p_min: float = 0.01,
    n_min: int = 5000,
    keep_top_m: int = 2,
    prefer_targets: Optional[Iterable[int]] = None,
    w: Sequence[float] = (0.35, 0.25, 0.25, 0.10, 0.05),
    min_center_cos: float = 0.0,
) -> Dict[int, int]:
    """Compute a merge map using phenotype/spatial/gene/center cues."""

    if cell_df.empty or "prototype_id" not in cell_df.columns:
        return {}

    proto_series = pd.to_numeric(cell_df["prototype_id"], errors="coerce").dropna().astype(int)
    counts = proto_series.value_counts().to_dict()
    K_from_centers = centers_tensor.shape[0] if centers_tensor is not None else 0
    K_from_df = int(proto_series.max() + 1) if not proto_series.empty else 0
    K = max(K_from_centers, K_from_df)
    if K == 0:
        return {}

    counts_arr = np.array([counts.get(k, 0) for k in range(K)], dtype=np.int64)
    props = counts_arr / (counts_arr.sum() + 1e-9)

    keep = set(np.argsort(-props)[: max(1, int(keep_top_m))].tolist())
    if prefer_targets:
        keep |= {int(x) for x in prefer_targets}
    targets = sorted([t for t in keep if 0 <= t < K])
    if not targets:
        targets = [0]

    def _proto_resp_prop(df, K):
        if "label" not in df.columns:
            return pd.Series([0.5] * K, index=range(K), dtype="float64")
        tab = (
            df[df["label"].isin([0, 1])]
            .groupby(["prototype_id", "label"]).size()
            .unstack(fill_value=0)
        )
        prop = (tab.get(1, 0) / (tab.sum(axis=1) + 1e-9)).reindex(range(K), fill_value=np.nan)
        return prop.fillna(0.5)

    resp_prop = _proto_resp_prop(cell_df, K)
    A_spatial = _coerce_proto_index(log2EN, K=K) if isinstance(log2EN, pd.DataFrame) else None
    A_knn = _proto_knn_transition(cell_df, K, k=8)
    A_gene = None  # placeholder for future gene similarity
    if centers_tensor is not None and centers_tensor.shape[0] == K:
        C_np = centers_tensor.detach().float().cpu().numpy()
        C_np /= np.linalg.norm(C_np, axis=1, keepdims=True) + 1e-8
        A_center = C_np @ C_np.T
        A_center_01 = (A_center + 1.0) / 2.0
    else:
        A_center_01 = np.full((K, K), 0.5, dtype=np.float32)

    def _safe_lookup(mat, i, j, default=0.0):
        try:
            return float(mat.loc[i, j])
        except Exception:
            return default

    mapping: Dict[int, int] = {}
    merged_pairs: List[Tuple[int, int, float, float]] = []

    for s in range(K):
        if (props[s] >= float(p_min) and counts_arr[s] >= int(n_min)) or (s in keep):
            mapping[s] = s
            continue

        scores = []
        for t in targets:
            a1 = 1.0 - abs(float(resp_prop.get(s, 0.5)) - float(resp_prop.get(t, 0.5)))
            a2 = _safe_lookup(A_spatial, s, t, 0.0) if A_spatial is not None else 0.0
            a2 = max(0.0, a2)
            a3 = _safe_lookup(A_knn, s, t, 0.0) if A_knn is not None else 0.0
            a4 = 0.0 if A_gene is None else _safe_lookup(A_gene, s, t, 0.0)
            a5 = float(A_center_01[s, t])
            score = w[0] * a1 + w[1] * a2 + w[2] * a3 + w[3] * a4 + w[4] * a5
            scores.append((t, score, a5))

        if not scores:
            mapping[s] = s
            continue

        t_star, best_score, best_center01 = max(scores, key=lambda x: x[1])
        if best_center01 >= float(min_center_cos):
            mapping[s] = int(t_star)
            merged_pairs.append((s, int(t_star), best_center01, best_score))
        else:
            mapping[s] = s

    try:
        n_merge = sum(1 for k, v in mapping.items() if k != v)
        print(
            f"[MERGE] multi-signal (center-cos>={min_center_cos}): merged {n_merge}/{K}."
        )
        if n_merge > 0:
            preview = ", ".join(
                [
                    f"{a}->{b}(cos={c:.2f})"
                    for a, b, c, _ in sorted(merged_pairs, key=lambda x: -x[2])[:10]
                ]
            )
            print(f"[MERGE] examples: {preview}")
    except Exception:
        pass

    return mapping


def _proto_knn_transition(df, K, k=8):
    need = {"pos_x", "pos_y", "prototype_id"}
    try:
        from sklearn.neighbors import KDTree
    except Exception:
        return None
    if (not need.issubset(df.columns)) or (len(df) < k + 1):
        return None
    d2 = df.dropna(subset=["pos_x", "pos_y"]).copy()
    if d2.empty or len(d2) < k + 1:
        return None
    coords = d2[["pos_x", "pos_y"]].to_numpy(np.float32)
    pids = pd.to_numeric(d2["prototype_id"], errors="coerce").astype(int).to_numpy()
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


def _coerce_proto_index(mat, K=None):
    if not isinstance(mat, pd.DataFrame):
        return None
    X = mat.copy()
    def _to_int_idx(idx):
        out = []
        for v in idx:
            s = str(v)
            if s.startswith("P") or s.startswith("p"):
                s = s[1:]
            try:
                out.append(int(s))
            except Exception:
                out.append(None)
        return out

    X.index = _to_int_idx(X.index)
    X.columns = _to_int_idx(X.columns)
    X = X.dropna(axis=0).dropna(axis=1)
    if K is not None:
        X = X.reindex(index=range(K), columns=range(K), fill_value=0.0)
    return X

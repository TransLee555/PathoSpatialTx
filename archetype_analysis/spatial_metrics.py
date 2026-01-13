"""Spatial metric helpers (interface density, proximity, etc.)."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

__all__ = [
    "compute_interface_density_per_sample",
    "compute_proximity_occupancy_to_P0",
    "compute_core_interface_metrics",
    "assemble_spatial_metrics",
]


def _build_knn_pairs_xy(xy: np.ndarray, k: int = 15):
    n = xy.shape[0]
    k_eff = min(k + 1, n)
    tree = KDTree(xy)
    ind = tree.query(xy, k=k_eff, return_distance=False)
    ind = ind[:, 1:]
    src = np.repeat(np.arange(n), ind.shape[1])
    dst = ind.ravel()
    d = np.linalg.norm(xy[src] - xy[dst], axis=1)
    return src, dst, d


def compute_interface_density_per_sample(
    cell_df: pd.DataFrame,
    k: int = 15,
    pairs: Sequence[Tuple[int, int]] = ((2, 0), (2, 4)),
):
    group_key = "sample_id" if "sample_id" in cell_df.columns else (
        "sample_name" if "sample_name" in cell_df.columns else None
    )
    if group_key is None:
        raise RuntimeError(
            "compute_interface_density_per_sample: neither sample_id nor sample_name in cell_df"
        )

    out = []
    for sid, sub in cell_df.groupby(group_key):
        sub = sub.dropna(subset=["pos_x", "pos_y", "prototype_id"]).copy()
        if len(sub) < max(10, k + 2):
            continue
        xy = sub[["pos_x", "pos_y"]].to_numpy(np.float32)
        proto = pd.to_numeric(sub["prototype_id"], errors="coerce").astype(int).to_numpy()

        u, v, dist = _build_knn_pairs_xy(xy, k=k)
        total_len = float(max(dist.sum(), 1e-6))
        rec = {group_key: sid, "total_edge_len": total_len, "n_cells": int(len(sub))}
        cross = proto[u] != proto[v]
        rec["edges_cross_rate"] = float(np.mean(cross))
        for (a, b) in pairs:
            sel = ((proto[u] == a) & (proto[v] == b)) | ((proto[u] == b) & (proto[v] == a))
            L_ab = float(dist[sel].sum())
            rec[f"ID_{a}_{b}"] = L_ab / total_len
        out.append(rec)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).set_index(group_key)
    return df


def compute_proximity_occupancy_to_P0(
    cell_df: pd.DataFrame,
    radii: Sequence[float] = (100.0, 150.0, 200.0),
    focus_proto: int = 2,
    microns_per_pixel: float = 1.0,
):
    group_key = "sample_id" if "sample_id" in cell_df.columns else (
        "sample_name" if "sample_name" in cell_df.columns else None
    )
    if group_key is None:
        raise RuntimeError(
            "compute_proximity_occupancy_to_P0: neither sample_id nor sample_name in cell_df"
        )

    out = []
    for sid, sub in cell_df.groupby(group_key):
        sub = sub.dropna(subset=["pos_x", "pos_y", "prototype_id"]).copy()
        if sub.empty:
            continue
        xy = sub[["pos_x", "pos_y"]].to_numpy(np.float32)
        proto = pd.to_numeric(sub["prototype_id"], errors="coerce").astype(int).to_numpy()
        p0_mask = proto == 0
        if not np.any(p0_mask):
            continue
        p0_xy = xy[p0_mask]
        tree_p0 = KDTree(p0_xy)
        dist_to_p0, _ = tree_p0.query(xy, k=1)
        dist_to_p0 = dist_to_p0.ravel() * float(microns_per_pixel)

        rec = {group_key: sid}
        for r in radii:
            in_near = dist_to_p0 <= float(r)
            denom = int(in_near.sum())
            key = f"P{focus_proto}_occ_r{int(r)}"
            if denom == 0:
                rec[key] = np.nan
            else:
                num = int(np.sum((proto == focus_proto) & in_near))
                rec[key] = float(num / denom)
        out.append(rec)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).set_index(group_key)
    return df


def compute_core_interface_metrics(
    cell_df: pd.DataFrame,
    pairs: Iterable[Tuple[int, int]] = (),
    core_thr: float = 0.7,
    excl_thr: float = 0.12,
    interface_thr: float = 0.40,
):
    if cell_df.empty:
        return pd.DataFrame()

    group_key = "sample_id" if "sample_id" in cell_df.columns else (
        "sample_name" if "sample_name" in cell_df.columns else None
    )
    if group_key is None:
        raise RuntimeError(
            "compute_core_interface_metrics: neither sample_id nor sample_name in cell_df"
        )

    Pcols = [c for c in cell_df.columns if c.startswith("P") and c[1:].isdigit()]
    if not Pcols:
        return pd.DataFrame()

    proto_ids = sorted({int(c[1:]) for c in Pcols})
    rows = []

    for sid, sdf in cell_df.groupby(group_key):
        total = float(len(sdf)) if len(sdf) else 1.0
        rec = {group_key: sid, "n_cells_total": int(len(sdf))}
        if len(sdf) == 0:
            rows.append(rec)
            continue
        for pid in proto_ids:
            col = f"P{pid}"
            if col not in sdf.columns:
                continue
            vals = sdf[col].to_numpy(dtype=float)
            if len(Pcols) > 1:
                others = np.vstack([sdf[c].to_numpy(dtype=float) for c in Pcols if c != col])
                other_max = np.max(others, axis=0)
            else:
                other_max = np.zeros_like(vals)
            margin = vals - other_max
            core_mask = (vals >= float(core_thr)) & (margin >= float(excl_thr))
            rec[f"core_frac_P{pid}"] = float(core_mask.sum() / total)
            rec[f"core_count_P{pid}"] = int(core_mask.sum())
            rec[f"core_margin_mean_P{pid}"] = (
                float(margin[core_mask].mean()) if core_mask.any() else np.nan
            )
            rec[f"core_prop_mean_P{pid}"] = float(vals.mean()) if len(vals) else np.nan

        for (i, j) in pairs or []:
            if_col = f"IF_P{i}_P{j}"
            if if_col not in sdf.columns:
                continue
            if_vals = sdf[if_col].to_numpy(dtype=float)
            iface_mask = if_vals >= float(interface_thr)
            rec[f"interface_frac_P{i}_P{j}"] = float(iface_mask.sum() / total)
            rec[f"interface_count_P{i}_P{j}"] = int(iface_mask.sum())
            rec[f"interface_mean_strength_P{i}_P{j}"] = (
                float(if_vals[iface_mask].mean()) if iface_mask.any() else np.nan
            )

        rows.append(rec)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(group_key)
    return df


def assemble_spatial_metrics(
    cell_df: pd.DataFrame,
    k: int = 15,
    radii: Sequence[int] = (100, 150, 200),
    focus_proto: int = 2,
    pairs: Sequence[Tuple[int, int]] = ((2, 0), (2, 4)),
    microns_per_pixel: float = 1.0,
    core_thr: float = 0.7,
    excl_thr: float = 0.12,
    interface_thr: float = 0.40,
):
    group_key = "sample_id" if "sample_id" in cell_df.columns else (
        "sample_name" if "sample_name" in cell_df.columns else None
    )
    if group_key is None:
        raise RuntimeError(
            "assemble_spatial_metrics: neither sample_id nor sample_name in cell_df"
        )

    idf = compute_interface_density_per_sample(cell_df, k=k, pairs=pairs)
    pdf = compute_proximity_occupancy_to_P0(
        cell_df,
        radii=radii,
        focus_proto=focus_proto,
        microns_per_pixel=microns_per_pixel,
    )
    res = idf.join(pdf, how="outer") if not idf.empty else pdf

    cim = compute_core_interface_metrics(
        cell_df,
        pairs=pairs,
        core_thr=core_thr,
        excl_thr=excl_thr,
        interface_thr=interface_thr,
    )
    if not cim.empty:
        res = res.join(cim, how="outer") if res is not None else cim

    if "prototype_id" in cell_df.columns:
        gp = (
            cell_df.dropna(subset=["prototype_id"])
            .assign(cnt=1)
            .pivot_table(
                index=group_key,
                columns="prototype_id",
                values="cnt",
                aggfunc="sum",
                fill_value=0,
            )
        )
        gp = gp.div(gp.sum(1), axis=0)
        gp.columns = [f"global_P{int(c)}" for c in gp.columns]
        res = res.join(gp, how="left") if res is not None else gp

    if "cohort" in cell_df.columns:
        cohort_series = cell_df.groupby(group_key)["cohort"].first()
        res = (
            res.join(cohort_series.rename("cohort"), how="left")
            if res is not None
            else cohort_series.to_frame(name="cohort")
        )

    if "pcr_response" in cell_df.columns:
        lab = cell_df.groupby(group_key)["pcr_response"].first()
        res = res.join(lab) if res is not None else lab.to_frame(name="pcr_response")

    return res

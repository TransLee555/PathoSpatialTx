"""Visualization helpers for stage3 analysis."""
from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

__all__ = [
    "plot_umap_advanced_panels",
    "plot_cell_prototypes_umap",
    "overlay_cell_prototypes_per_sample",
]


def _pick_pid_cols(df: pd.DataFrame):
    raw_col = "prototype_id_raw" if "prototype_id_raw" in df.columns else "prototype_id"
    merged_col = "prototype_id_merged" if "prototype_id_merged" in df.columns else "prototype_id"
    if raw_col == merged_col:
        print(
            f"[WARN] raw_col == merged_col == '{raw_col}' — "
            "no raw backup or merged column present."
        )
    return raw_col, merged_col


def _ensure_umap2d_inplace(
    df: pd.DataFrame,
    entity: str = "cell",
    max_points: int = 200_000,
    n_neighbors_cell: int = 20,
    n_neighbors_gene: int = 15,
):
    if df.empty:
        return df
    if "umap_x" in df.columns and df["umap_x"].notna().any():
        return df
    embed_cols = [c for c in df.columns if str(c).startswith("embed_")]
    if not embed_cols:
        return df
    samp = df
    if len(df) > max_points:
        samp = df.sample(n=max_points, random_state=42)
    X = samp[embed_cols].to_numpy(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    reducer = umap.UMAP(
        n_neighbors=(n_neighbors_cell if entity == "cell" else n_neighbors_gene),
        min_dist=(0.05 if entity == "cell" else 0.10),
        n_components=2,
        random_state=42,
        densmap=False,
    )
    XY = reducer.fit_transform(X)
    df["umap_x"], df["umap_y"] = np.nan, np.nan
    df.loc[samp.index, ["umap_x", "umap_y"]] = XY
    return df


def _make_proto_palette(unique_pids):
    pids = sorted([int(x) for x in unique_pids])
    n = max(1, len(pids))
    if n <= 20:
        colors = sns.color_palette("tab20", n)
    else:
        m = n - 20
        colors = list(sns.color_palette("tab20", 20)) + list(
            sns.husl_palette(m, s=0.75, l=0.55)
        )
    return {pid: tuple(colors[i]) for i, pid in enumerate(pids)}


def _panel_scatter(
    ax,
    df: pd.DataFrame,
    pid_col: str,
    title: str,
    max_per_proto: int = 20_000,
    point_size: int = 2,
    alpha: float = 0.35,
):
    valid = df.dropna(subset=["umap_x", "umap_y"]).copy()
    if valid.empty:
        ax.set_title(title)
        ax.axis("off")
        return
    uniq = pd.to_numeric(valid[pid_col], errors="coerce").dropna().unique()
    pal = _make_proto_palette(uniq)
    for pid in sorted(uniq):
        sub = valid[valid[pid_col] == pid]
        if len(sub) > max_per_proto:
            sub = sub.sample(n=max_per_proto, random_state=42)
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=point_size,
            alpha=alpha,
            color=pal.get(int(pid), (0.5, 0.5, 0.5)),
            label=f"P{int(pid)}",
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=5, fontsize=6, loc="best")


def plot_umap_advanced_panels(
    cell_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    out_dir: str = "./analysis_out/1_umap_visualizations_advanced",
    max_per_proto: int = 20_000,
):
    os.makedirs(out_dir, exist_ok=True)
    cdf = _ensure_umap2d_inplace(cell_df.copy(), entity="cell")
    gdf = _ensure_umap2d_inplace(gene_df.copy(), entity="gene")
    c_raw, c_mrg = _pick_pid_cols(cdf)
    g_raw, g_mrg = _pick_pid_cols(gdf)

    from matplotlib.gridspec import GridSpec

    plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.12, wspace=0.08)
    ax = plt.subplot(gs[0, 0])
    _panel_scatter(ax, cdf, c_raw, "Cells — Raw prototypes")
    ax = plt.subplot(gs[0, 1])
    _panel_scatter(ax, cdf, c_mrg, "Cells — Merged prototypes")
    ax = plt.subplot(gs[1, 0])
    if g_raw in gdf.columns and gdf[g_raw].notna().any():
        _panel_scatter(ax, gdf, g_raw, "Genes — Raw prototypes")
    else:
        ax.set_title("Genes — Raw prototypes (N/A)")
        ax.axis("off")
    ax = plt.subplot(gs[1, 1])
    if g_mrg in gdf.columns and gdf[g_mrg].notna().any():
        _panel_scatter(ax, gdf, g_mrg, "Genes — Merged prototypes")
    else:
        ax.set_title("Genes — Merged prototypes (N/A)")
        ax.axis("off")
    plt.suptitle(
        "Advanced 2D UMAP — Cells & Genes × Raw vs Merged",
        fontsize=14,
        weight="bold",
    )
    out_png = os.path.join(out_dir, "umap_advanced_panels.png")
    plt.tight_layout(rect=[0, 0.00, 1, 0.97])
    plt.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close()

    singles = [
        ("cells_raw.png", cdf, c_raw, "Cells — Raw prototypes"),
        ("cells_merged.png", cdf, c_mrg, "Cells — Merged prototypes"),
        ("genes_raw.png", gdf, g_raw, "Genes — Raw prototypes"),
        ("genes_merged.png", gdf, g_mrg, "Genes — Merged prototypes"),
    ]
    for fname, df_, pid_col, title in singles:
        if pid_col not in df_.columns:
            continue
        plt.figure(figsize=(7.5, 6.2))
        ax = plt.gca()
        _panel_scatter(ax, df_, pid_col, title, max_per_proto=max_per_proto)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=320, bbox_inches="tight")
        plt.close()
    print(f"[UMAP-ADV] saved: {out_png}")


def plot_cell_prototypes_umap(
    cell_df: pd.DataFrame,
    out_dir: str = "./analysis_out",
    use_merged: bool = False,
    max_cells: int = 200_000,
):
    os.makedirs(out_dir, exist_ok=True)
    pid_col = (
        "prototype_id_merged"
        if (use_merged and "prototype_id_merged" in cell_df.columns)
        else "prototype_id"
    )
    if cell_df.empty or pid_col not in cell_df.columns:
        print("[VIS] no cell prototypes; skip.")
        return

    if {"umap_x", "umap_y"}.issubset(cell_df.columns):
        df2d = cell_df.dropna(subset=["umap_x", "umap_y"])
        if not df2d.empty:
            if len(df2d) > max_cells:
                df2d = df2d.sample(n=max_cells, random_state=42)
            plt.figure(figsize=(10, 8))
            sns.kdeplot(
                data=df2d,
                x="umap_x",
                y="umap_y",
                hue=pid_col,
                fill=True,
                alpha=0.65,
                levels=70,
                thresh=0.05,
                common_norm=False,
            )
            plt.title(
                f"Cell prototypes — 2D density UMAP ({'merged' if pid_col != 'prototype_id' else 'raw'})"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"cell_proto_umap2d_density_{pid_col}.png"),
                dpi=260,
            )
            plt.close()
    else:
        print("[VIS] Missing 2D UMAP columns on cell_df; skip density panel.")

    if {"umap_3d_x", "umap_3d_y", "umap_3d_z"}.issubset(cell_df.columns):
        df3d = cell_df.dropna(subset=["umap_3d_x", "umap_3d_y", "umap_3d_z"])
        if not df3d.empty:
            if len(df3d) > max_cells:
                df3d = df3d.sample(n=max_cells, random_state=42)
            fig = plt.figure(figsize=(12, 11))
            ax = fig.add_subplot(111, projection="3d")
            for k in sorted(pd.to_numeric(df3d[pid_col], errors="coerce").dropna().unique()):
                sub = df3d[df3d[pid_col] == k]
                ax.scatter(
                    sub["umap_3d_x"],
                    sub["umap_3d_y"],
                    sub["umap_3d_z"],
                    s=7,
                    alpha=0.6,
                    label=f"P{k}",
                )
            ax.set_title(
                f"Cell prototypes — 3D UMAP ({'merged' if pid_col != 'prototype_id' else 'raw'})"
            )
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_zlabel("UMAP-3")
            ax.legend(markerscale=2.0)
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"cell_proto_umap3d_scatter_{pid_col}.png"),
                dpi=280,
            )
            plt.close()
    else:
        print("[VIS] Missing 3D UMAP columns on cell_df; skip 3D panel.")


def overlay_cell_prototypes_per_sample(
    cell_df: pd.DataFrame,
    sample_dict: Dict[str, object],
    svs_root_map: Dict[str, str],
    out_dir: str = "./analysis_out/overlays",
    use_merged: bool = False,
    min_cells: int = 50,
):
    os.makedirs(out_dir, exist_ok=True)
    pid_col = (
        "prototype_id_merged"
        if (use_merged and "prototype_id_merged" in cell_df.columns)
        else "prototype_id"
    )
    need = {"pos_x", "pos_y", pid_col, "sample_name"}
    if not need.issubset(cell_df.columns):
        print("[OVERLAY] missing columns; skip overlays.")
        return
    uniq = sorted(
        pd.to_numeric(cell_df[pid_col], errors="coerce").dropna().unique().tolist()
    )
    pal = sns.color_palette("tab10", n_colors=max(10, len(uniq)))
    cmap = {int(k): pal[i % len(pal)] for i, k in enumerate(uniq)}

    for sample, sdf in cell_df.groupby("sample_name"):
        if len(sdf) < min_cells:
            continue
        cohort = str(sdf["cohort"].iat[0]) if "cohort" in sdf.columns else "NA"
        plt.figure(figsize=(8, 7))
        for k, sub in sdf.groupby(pid_col):
            plt.scatter(
                sub["pos_x"],
                sub["pos_y"],
                s=1,
                alpha=0.6,
                color=cmap.get(int(k), (0.5, 0.5, 0.5)),
                label=f"P{k}",
            )
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.legend(markerscale=4, fontsize=8, loc="upper right")
        plt.title(f"{cohort} — {sample}  cell prototypes ({pid_col})")
        spath = os.path.join(out_dir, f"{cohort}__{sample}__{pid_col}.png")
        plt.tight_layout()
        plt.savefig(spath, dpi=260)
        plt.close()

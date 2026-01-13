"""WSI overlay and alignment helpers."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from scipy.ndimage import binary_dilation, gaussian_filter

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
    from shapely.geometry import MultiPolygon, Polygon, shape
    _HAS_SHAPELY = True
except Exception:
    MultiPolygon = Polygon = shape = None
    _HAS_SHAPELY = False
try:
    from skimage.draw import polygon as SKIMAGE_POLYGON
except Exception:
    SKIMAGE_POLYGON = None
try:
    from sklearn.neighbors import KDTree
except Exception:
    KDTree = None

__all__ = [
    'collect_cell_alignment_scores',
    'overlay_mask_heatmap_on_wsi',
    'run_alignment_wsi_overlay_for_batch',
    'overlay_wsi_gate_for_sample',
    'overlay_wsi_core_edge_for_sample',
    'overlay_wsi_prototypes_for_sample',
    'report_wsi_asset_status',
]

def _sample_lookup_variants(sample_name):
    s = str(sample_name).strip()
    if not s:
        return []
    variants = {s}
    variants.add(s.replace(" ", ""))
    variants.add(s.replace('_', '-'))
    variants.add(s.replace('-', '_'))
    variants = {v for v in variants if v}
    return list(variants)


_SVS_SEARCH_CACHE: Dict[Path, List[Tuple[str, Path]]] = {}
_GEOJSON_DIR_CACHE: Dict[Path, List[Tuple[str, Path]]] = {}


def _cached_svs_entries(root: Path) -> List[Tuple[str, Path]]:
    root = Path(root).resolve()
    cache = _SVS_SEARCH_CACHE.get(root)
    if cache is None:
        entries: List[Tuple[str, Path]] = []
        print(f"[CACHE] indexing SVS files under {root} ...")
        for svs_path in sorted(root.rglob('*.svs')):
            entries.append((svs_path.name.lower(), svs_path))
        _SVS_SEARCH_CACHE[root] = entries
        cache = entries
    return cache


def _cached_geojson_dirs(root: Path) -> List[Tuple[str, Path]]:
    root = Path(root).resolve()
    cache = _GEOJSON_DIR_CACHE.get(root)
    if cache is None:
        entries: List[Tuple[str, Path]] = []
        print(f"[CACHE] indexing GeoJSON directories under {root} ...")
        try:
            for entry in root.iterdir():
                if entry.is_dir():
                    entries.append((entry.name.lower(), entry))
        except FileNotFoundError:
            entries = []
        _GEOJSON_DIR_CACHE[root] = entries
        cache = entries
    return cache


def _find_svs(svs_root, sample_name):
    if not svs_root: return None
    root = Path(svs_root)
    if not root.exists():
        return None
    entries = _cached_svs_entries(root)
    if not entries:
        return None
    for key in _sample_lookup_variants(sample_name):
        key_low = key.lower()
        for name_low, path in entries:
            if key_low in name_low:
                return str(path)
    return None


def _find_geojson(cellgeojson_root, sample_name):
    if not cellgeojson_root:
        return None
    root = Path(cellgeojson_root)
    if not root.exists():
        return None
    entries = _cached_geojson_dirs(root)
    candidates = []
    for key in _sample_lookup_variants(sample_name):
        direct = root / key
        if direct.exists() and direct.is_dir():
            candidates.append(direct)
        key_low = key.lower()
        candidates.extend(p for name_low, p in entries if key_low in name_low)
    for base in candidates:
        for rel in [
            'cell_detection/cells.geojson',
            'cell_detection/cell_detection.geojson',
            'cells.geojson',
            f'{sample_name}.geojson',
            f'{sample_name}_cells.geojson',
        ]:
            target = base / rel
            if target.exists():
                return str(target)
    return None


def _normalize_cohort_key(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return re.sub(r"\s+", "", s).lower()


def _resolve_cohort_path(mapping, cohort_value):
    """Case/whitespace agnostic lookup of cohort-specific roots."""
    if not mapping:
        return None
    key_norm = _normalize_cohort_key(cohort_value)
    if key_norm is None:
        return None
    norm_map = { _normalize_cohort_key(k): path for k, path in mapping.items() }
    # 1) exact match after normalization
    if key_norm in norm_map:
        return norm_map[key_norm]
    # 2) fallback: allow substring containment (e.g. 'yalenat' hit 'yale')
    for k_norm, path in norm_map.items():
        if k_norm and (k_norm in key_norm or key_norm in k_norm):
            return path
    return None

def load_cell_polygons(geojson_path, slide=None,
                       geojson_level_key='level',
                       geojson_downsample_key='downsample'):
    """读取 cells.geojson → list[np.ndarray(N,2)]（坐标均在 level-0 空间）。"""
    try:
        import json
        with open(geojson_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[WARN] load_cell_polygons failed: {e}")
        return []

    if isinstance(raw, list):
        features = raw
    elif isinstance(raw, dict):
        features = raw.get('features', [])
        if (not isinstance(features, list)) and raw.get('geometry'):
            features = [raw]
    else:
        features = []

    ds_to_lvl0, lvl = 1.0, None
    for ft in features:
        if not isinstance(ft, dict):
            continue
        props = ft.get('properties') or {}
        if geojson_downsample_key in props:
            try:
                ds_to_lvl0 = float(props[geojson_downsample_key]); lvl = None; break
            except Exception:
                pass
        if geojson_level_key in props:
            try:
                lvl = int(props[geojson_level_key]); break
            except Exception:
                pass
    if (lvl is not None) and (slide is not None):
        lvl = min(max(int(lvl), 0), slide.level_count - 1)
        ds_to_lvl0 = float(slide.level_downsamples[lvl])

    polys = []
    for ft in features:
        geom = ft.get('geometry') if isinstance(ft, dict) else None
        if geom is None and isinstance(ft, dict) and ft.get('type') in ('Polygon', 'MultiPolygon'):
            geom = ft
        if not isinstance(geom, dict):
            continue

        gtype = geom.get('type')
        coords = geom.get('coordinates')

        if _HAS_SHAPELY and gtype:
            try:
                shp = shape(geom)
                if not shp.is_valid:
                    shp = shp.buffer(0)
                if shp.is_empty:
                    continue
                if isinstance(shp, Polygon):
                    polys.append(np.asarray(shp.exterior.coords, np.float32) * ds_to_lvl0)
                    continue
                if isinstance(shp, MultiPolygon):
                    for sub in shp.geoms:
                        if sub.is_empty:
                            continue
                        polys.append(np.asarray(sub.exterior.coords, np.float32) * ds_to_lvl0)
                    continue
            except Exception:
                pass

        if not isinstance(coords, list) or len(coords) == 0:
            continue
        if gtype == 'Polygon':
            ring = coords[0]
            P = np.asarray(ring, np.float32)
            if P.ndim == 2 and P.shape[1] >= 2:
                polys.append(P[:, :2] * ds_to_lvl0)
        elif gtype == 'MultiPolygon':
            for poly in coords:
                if not poly:
                    continue
                ring = poly[0]
                P = np.asarray(ring, np.float32)
                if P.ndim == 2 and P.shape[1] >= 2:
                    polys.append(P[:, :2] * ds_to_lvl0)
        else:
            if isinstance(coords[0], list) and len(coords[0]) and isinstance(coords[0][0], (list, tuple)):
                ring = coords[0]
                P = np.asarray(ring, np.float32)
                if P.ndim == 2 and P.shape[1] >= 2:
                    polys.append(P[:, :2] * ds_to_lvl0)
    return polys

def polygon_centroids(polys_lvl0, return_mask: bool = False):
    cents = []
    mask = []
    for P in polys_lvl0:
        valid = isinstance(P, np.ndarray) and P.ndim == 2 and P.shape[0] >= 3 and P.shape[1] >= 2
        mask.append(valid)
        if not valid:
            continue
        x, y = P[:, 0], P[:, 1]
        a = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2.0
        if abs(a) < 1e-6:
            cx, cy = x.mean(), y.mean()
        else:
            cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
            cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
        cents.append([cx, cy])
    cents_arr = np.asarray(cents, dtype=np.float32)
    if return_mask:
        return cents_arr, np.asarray(mask, dtype=bool)
    return cents_arr

def _majority_vote_knn_classes(query_xy_lvl0, ref_xy_lvl0, ref_class_int, k=3, chunk_size=2048):
    Q = np.asarray(query_xy_lvl0, np.float32)
    R = np.asarray(ref_xy_lvl0, np.float32)
    C = np.asarray(ref_class_int, np.int32).reshape(-1)
    if len(Q) == 0 or len(R) == 0:
        return np.zeros((len(Q),), np.int32), np.zeros((len(Q),), np.float32)

    kk = int(min(max(1, k), R.shape[0]))
    if kk <= 0:
        return np.zeros((len(Q),), np.int32), np.zeros((len(Q),), np.float32)

    idx = None
    if KDTree is not None:
        try:
            tree = KDTree(R)
            idx = tree.query(Q, k=kk, return_distance=False)
        except Exception:
            idx = None

    pred = np.zeros((len(Q),), np.int32)
    conf = np.zeros((len(Q),), np.float32)

    if idx is not None:
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx[:, None]
        votes = C[idx]
        for i in range(len(Q)):
            vs = votes[i]
            vals, counts = np.unique(vs, return_counts=True)
            j = counts.argmax()
            pred[i] = int(vals[j])
            conf[i] = float(counts[j] / float(kk))
        return pred, conf

    B = max(1, int(chunk_size))
    for s in range(0, len(Q), B):
        e = min(s + B, len(Q))
        d2 = ((Q[s:e, None, :] - R[None, :, :]) ** 2).sum(axis=2)
        idx_chunk = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        votes = C[idx_chunk]
        for i in range(e - s):
            vs = votes[i]
            vals, counts = np.unique(vs, return_counts=True)
            j = counts.argmax()
            pred[s + i] = int(vals[j])
            conf[s + i] = float(counts[j] / float(kk))

    return pred, conf



# ==== 通用：把多边形映射为“最近细胞”的标量/类别 ====
def _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, cell_scalar, k=3, clip=(1, 99), chunk_size=2048):
    n = len(polys_lvl0)
    if n == 0 or len(cell_xy_lvl0) == 0:
        return np.zeros((n,), np.float32), np.zeros((n,), np.float32)

    cents, valid_mask = polygon_centroids(polys_lvl0, return_mask=True)
    if cents.size == 0:
        return np.zeros((n,), np.float32), np.zeros((n,), np.float32)

    ref_ids = np.arange(len(cell_xy_lvl0), dtype=np.int32)
    pred_idx, conf = _majority_vote_knn_classes(cents, cell_xy_lvl0, ref_ids, k=k, chunk_size=chunk_size)

    v = np.asarray(cell_scalar, np.float32).reshape(-1)
    if not np.isfinite(v).any():
        v = np.zeros_like(v, dtype=np.float32)
    else:
        v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0)
    try:
        lo, hi = np.nanpercentile(v, list(clip))
    except Exception:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    rng = max(1e-6, hi - lo)
    vv = np.clip((v - lo) / rng, 0, 1)

    scores = np.zeros((n,), np.float32)
    conf_full = np.zeros((n,), np.float32)
    if len(pred_idx):
        scores_valid = vv[pred_idx]
        scores[valid_mask] = scores_valid
        conf_full[valid_mask] = conf
    return scores, conf_full


def _poly_to_cell_scalar_colors(polys_lvl0, cell_xy_lvl0, cell_scalar, k=3, cmap_name='viridis', alpha_fill=140):
    if len(polys_lvl0) == 0 or len(cell_xy_lvl0) == 0:
        return []
    scores, _ = _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, cell_scalar, k=k)
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    colors = [tuple(int(255*c) for c in cmap(float(val))[:3]) + (int(alpha_fill),) for val in np.clip(scores, 0, 1)]
    return colors


def _stretch01_inclusive(arr, q_low=2.0, q_high=98.0):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=np.float32), (0.0, 1.0)
    lo = float(np.percentile(arr[np.isfinite(arr)], q_low))
    hi = float(np.percentile(arr[np.isfinite(arr)], q_high))
    if hi - lo < 1e-8:
        if hi == 0:
            return np.zeros_like(arr, dtype=np.float32), (0.0, 1.0)
        return np.clip(arr / (hi + 1e-8), 0, 1).astype(np.float32), (0.0, hi)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32), (lo, hi)


def _add_colorbar_right(img_rgb, *, cmap_name='magma', value_range=(0.0, 1.0),
                        height_ratio=0.45, width_px=48, pad_px=10, ticks=5,
                        title="alignment score", use_red_half=False):
    if Image is None or ImageDraw is None:
        return img_rgb
    try:
        import matplotlib.cm as cm
    except Exception:
        return img_rgb

    lo, hi = value_range
    W, H = img_rgb.size
    cb_h = max(40, int(H * float(height_ratio)))
    cb_w = int(width_px)
    pad = int(pad_px)

    ramp = np.linspace(1.0, 0.0, cb_h, dtype=np.float32)
    lut_in = 0.5 + 0.5 * ramp if use_red_half else ramp
    cmap = cm.get_cmap(cmap_name)
    lut = cmap(lut_in)[..., :3]
    rgb = (lut * 255).astype(np.uint8)
    bar = Image.fromarray(np.repeat(rgb[:, None, :], cb_w, axis=1), mode='RGB')

    canvas = Image.new('RGB', (W + cb_w + pad + 70, H), (255, 255, 255))
    canvas.paste(img_rgb, (0, 0))
    x0 = W + pad
    y0 = (H - cb_h) // 2
    canvas.paste(bar, (x0, y0))

    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_tick = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font_title = font_tick = None

    ticks = max(2, int(ticks))
    for i in range(ticks):
        t = i / (ticks - 1) if ticks > 1 else 0.0
        y = y0 + int((1.0 - t) * (cb_h - 1))
        draw.line([(x0 + cb_w, y), (x0 + cb_w + 8, y)], fill=(0, 0, 0), width=1)
        val = lo + t * (hi - lo)
        draw.text((x0 + cb_w + 10, y - 6), f"{val:.2f}", fill=(0, 0, 0), font=font_tick)
    if title:
        draw.text((x0, max(0, y0 - 24)), title, fill=(0, 0, 0), font=font_title)
    return canvas


def nn_assign_scores_to_polys(cell_xy_lvl0, cell_scores01, poly_centroids_lvl0,
                              k=3, gaussian_sigma=24.0):
    C = np.asarray(cell_xy_lvl0, np.float32)
    S = np.asarray(cell_scores01, np.float32).reshape(-1)
    P = np.asarray(poly_centroids_lvl0, np.float32)
    if len(C) == 0 or len(P) == 0:
        return np.zeros((len(P),), np.float32)

    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    k = int(min(max(1, k), C.shape[0]))
    idx = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
    rows = np.arange(P.shape[0])[:, None]
    d2_k = d2[rows, idx]
    s_k = S[idx]

    if gaussian_sigma is None or gaussian_sigma <= 0:
        w = 1.0 / (np.sqrt(d2_k) + 1e-6)
    else:
        sig2 = float(gaussian_sigma) ** 2
        w = np.exp(-d2_k / (2.0 * sig2))

    out = (w * s_k).sum(axis=1) / (w.sum(axis=1) + 1e-8)
    return out.astype(np.float32)


@torch.no_grad()
def collect_cell_alignment_scores(model, data, direction='c_g', soften_temp=3.0, reduce='mean'):
    dev = next(model.parameters()).device
    try:
        data = data.to(dev)
    except Exception:
        pass

    forward_fn = getattr(model, 'analysis_forward_with_gate', None)
    if forward_fn is None:
        raise RuntimeError("Model does not expose analysis_forward_with_gate.")

    try:
        z_cell, z_gene = forward_fn(data, soften_gate=True, soften_temp=float(soften_temp))
    except TypeError:
        z_cell, z_gene = forward_fn(data)

    if z_cell is None or z_cell.numel() == 0:
        return np.zeros((0,), np.float32), None

    bridge_fn = getattr(model, '_bridge', None)
    soft_assign = getattr(model, '_soft_assign', None)
    centers_cell = getattr(model, 'centers_cell_ema', None) or getattr(model, 'centers_cell', None)
    centers_gene = getattr(model, 'centers_gene_ema', None) or getattr(model, 'centers_gene', None)
    if soft_assign is None or bridge_fn is None or centers_cell is None or centers_gene is None:
        raise RuntimeError("Model missing _soft_assign/_bridge/centers for alignment scoring.")

    Nc = z_cell.size(0)
    Qc = soft_assign(z_cell, centers_cell)
    Qg = soft_assign(z_gene, centers_gene) if z_gene is not None and z_gene.numel() else torch.zeros((0, Qc.size(1)), device=dev)
    B_c2g, _ = bridge_fn()

    key = ('cell', 'c_g', 'gene') if direction == 'c_g' else ('gene', 'g_c', 'cell')
    if key not in data.edge_types:
        pos = getattr(data['cell'], 'pos', None)
        pos_xy = pos.detach().cpu().numpy().astype(np.float32) if (pos is not None and pos.numel() > 0) else None
        return np.zeros((Nc,), np.float32), pos_xy

    ei = data[key].edge_index
    rows, cols = ei[0].long(), ei[1].long()
    compat = torch.sum((Qc[rows] @ B_c2g) * Qg[cols], dim=1)
    eattr = getattr(data[key], 'edge_attr', None)
    if eattr is not None and eattr.numel() > 0:
        compat = compat * eattr.view(-1).to(compat)

    softened = torch.sign(compat) * torch.pow(torch.abs(compat) + 1e-8, 1.0 / max(1e-6, float(soften_temp)))
    score = torch.zeros(Nc, device=dev, dtype=softened.dtype)
    if reduce == 'max':
        try:
            score.scatter_reduce_(0, rows, softened, reduce='amax', include_self=False)
        except Exception:
            for idx, val in zip(rows.cpu().numpy(), softened.cpu().numpy()):
                if val > score[idx]:
                    score[idx] = val
    else:
        cnt = torch.zeros(Nc, device=dev, dtype=softened.dtype)
        score.index_add_(0, rows, softened)
        cnt.index_add_(0, rows, torch.ones_like(softened))
        score = score / (cnt + 1e-8)

    score_np = score.detach().float().cpu().numpy()
    s_min, s_max = float(score_np.min()), float(score_np.max())
    if s_max - s_min < 1e-8:
        score01 = np.zeros_like(score_np, dtype=np.float32)
    else:
        score01 = ((score_np - s_min) / (s_max - s_min)).astype(np.float32)

    pos = getattr(data['cell'], 'pos', None)
    pos_xy = pos.detach().cpu().numpy().astype(np.float32) if (pos is not None and pos.numel() > 0) else None
    return score01, pos_xy


def overlay_mask_heatmap_on_wsi(
    svs_path,
    polys_lvl0,
    poly_scores01,
    out_png,
    target_level=1,
    blend='sum',
    sigma_lvl0=32.0,
    cmap_name='magma',
    cmap_use_full=None,
    alpha_gamma=0.5,
    alpha_max=240,
    alpha_floor=150,
    brighten_bg=0.08,
    draw_all_outlines=False,
    outline_rgba=(255, 255, 255, 25),
    contour_polys=None,
    contour_rgba=(255, 255, 255, 220),
    contour_width=2,
    low_clip_q=8.0,
    high_clip_q=99.2,
    colorbar_h_ratio=0.40,
    colorbar_w_px=84,
    colorbar_pad_px=24,
    colorbar_title=None,
    fallback_disk_px=4,
    grow_cover_px=4,
    post_smooth_frac=0.0,
    alpha_smooth_frac=0.0,
):
    if OpenSlide is None or Image is None:
        print("[ALIGN] OpenSlide/PIL unavailable; skip heatmap overlay.")
        return

    from scipy.ndimage import gaussian_filter, binary_dilation

    slide = OpenSlide(str(svs_path))
    L = max(0, min(int(target_level), slide.level_count - 1))
    W, H = slide.level_dimensions[L]
    ds = float(slide.level_downsamples[L])

    bg = slide.read_region((0, 0), L, (W, H)).convert('RGB')
    slide.close()

    if brighten_bg:
        arr_bg = np.asarray(bg, dtype=np.float32)
        arr_bg = np.clip(arr_bg * (1.0 + float(brighten_bg)), 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr_bg)

    heat = np.zeros((H, W), np.float32)
    cover = np.zeros((H, W), np.float32) if blend == 'mean' else None

    for P0, val in zip(polys_lvl0, np.asarray(poly_scores01, dtype=np.float32)):
        if P0 is None:
            continue
        P = np.asarray(P0, np.float32) / ds
        if P.ndim != 2 or P.shape[0] < 3:
            continue
        if SKIMAGE_POLYGON is not None:
            rr, cc = SKIMAGE_POLYGON(P[:, 1], P[:, 0], (H, W))
            if rr.size and cc.size:
                np.add.at(heat, (rr, cc), float(val))
                if cover is not None:
                    np.add.at(cover, (rr, cc), 1.0)
        elif ImageDraw is not None:
            mask = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon([tuple(xy) for xy in P], fill=1, outline=1)
            idx = np.array(mask, dtype=bool)
            heat[idx] += float(val)
            if cover is not None:
                cover[idx] += 1.0

    sigma_px = 0.0
    if heat.max() > 0 and sigma_lvl0 > 0:
        sigma_px = max(0.0, float(sigma_lvl0) / ds)
        heat = gaussian_filter(heat, sigma=max(0.5, sigma_px), mode='nearest')

    if cover is not None:
        heat = np.divide(heat, np.maximum(cover, 1e-6), out=np.zeros_like(heat), where=cover > 0)

    heat_norm, (lo, hi) = _stretch01_inclusive(heat, low_clip_q, high_clip_q)
    if sigma_px > 0 and post_smooth_frac > 0:
        heat_norm = gaussian_filter(heat_norm, max(0.3, sigma_px * float(post_smooth_frac)), mode='nearest')

    cmap_use_full = bool(cmap_use_full) if cmap_use_full is not None else True
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    mapped = cmap(heat_norm if cmap_use_full else 0.5 + 0.5 * heat_norm)[..., :3]
    rgba = (mapped * 255).astype(np.uint8)

    alpha = np.zeros_like(heat_norm, dtype=np.float32)
    alpha_mask = heat_norm > 0
    alpha[alpha_mask] = (np.power(heat_norm[alpha_mask], float(alpha_gamma)) * (alpha_max - alpha_floor) + alpha_floor)
    if sigma_px > 0 and alpha_smooth_frac > 0:
        alpha = gaussian_filter(alpha, max(0.5, sigma_px * float(alpha_smooth_frac)), mode='nearest')
    eps = 0.03
    alpha[heat_norm <= eps] = 0
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    overlay_rgba = np.dstack([rgba, alpha])

    if grow_cover_px and grow_cover_px > 0:
        base_mask = alpha > 0
        if base_mask.any():
            try:
                from skimage.morphology import dilation, disk
                grown = dilation(base_mask, footprint=disk(int(grow_cover_px)))
            except Exception:
                grown = binary_dilation(base_mask, iterations=int(grow_cover_px))
            halo_mask = grown & ~base_mask
            if halo_mask.any():
                from scipy.ndimage import distance_transform_edt
                _, (iy, ix) = distance_transform_edt(~base_mask, return_indices=True)
                overlay_rgba[halo_mask, :3] = overlay_rgba[iy[halo_mask], ix[halo_mask], :3]
                overlay_rgba[halo_mask, 3] = np.maximum(overlay_rgba[halo_mask, 3], alpha_floor // 2)

    overlay_img = Image.fromarray(overlay_rgba, mode='RGBA')
    composite = Image.alpha_composite(bg.convert('RGBA'), overlay_img)

    if draw_all_outlines and ImageDraw is not None:
        outline = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        out_draw = ImageDraw.Draw(outline, 'RGBA')
        for P0 in polys_lvl0:
            if P0 is None:
                continue
            P = np.asarray(P0, np.float32) / ds
            if P.ndim != 2 or P.shape[0] < 3:
                continue
            pts = [tuple(xy) for xy in P]
            out_draw.line(pts + [pts[0]], fill=outline_rgba, width=1)
        composite = Image.alpha_composite(composite.convert('RGBA'), outline).convert('RGB')

    if contour_polys and ImageDraw is not None:
        contour_img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        contour_draw = ImageDraw.Draw(contour_img, 'RGBA')
        cw = max(1, int(contour_width))
        for P0 in contour_polys:
            if P0 is None:
                continue
            P = np.asarray(P0, np.float32) / ds
            if P.ndim != 2 or P.shape[0] < 3:
                continue
            pts = [tuple(xy) for xy in P]
            contour_draw.line(pts + [pts[0]], fill=contour_rgba, width=cw)
        composite = Image.alpha_composite(composite.convert('RGBA'), contour_img).convert('RGB')

    composite = composite.convert('RGB')
    composite = _add_colorbar_right(
        composite,
        cmap_name=cmap_name,
        value_range=(lo, hi),
        height_ratio=colorbar_h_ratio,
        width_px=colorbar_w_px,
        pad_px=colorbar_pad_px,
        title=colorbar_title or "alignment",
        use_red_half=(not cmap_use_full)
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_png, quality=95)


def run_alignment_wsi_overlay_for_batch(
    model,
    data_obj,
    svs_root,
    cellgeojson_root,
    cohort_pos_level_map=None,
    target_level=None,
    soften_temp=3.0,
    out_root='./wsi_alignment_overlays',
    nn_k=3,
    nn_sigma_lvl0=32.0,
    cmap_name='magma',
    blend='sum',
    cmap_use_full=None,
    sigma_lvl0=32.0,
    colorbar_h_ratio=0.40,
    colorbar_w_px=84,
    colorbar_pad_px=24,
    low_clip_q=8.0,
    high_clip_q=99.2,
    alpha_gamma=0.5,
    alpha_max=240,
    alpha_floor=150,
    brighten_bg=0.08,
    grow_cover_px=4,
    draw_all_outlines=True,
    fallback_disk_px=4,
    post_smooth_frac=0.0,
    alpha_smooth_frac=0.0,
):
    if OpenSlide is None or Image is None:
        print("[ALIGN] OpenSlide/PIL unavailable; skip alignment overlay.")
        return

    name = data_obj.patient_id[0] if isinstance(data_obj.patient_id, list) else data_obj.patient_id
    cohort_raw = data_obj.cohort[0] if isinstance(data_obj.cohort, list) else data_obj.cohort
    cohort_label = str(cohort_raw).strip() if cohort_raw is not None else 'NA'

    try:
        score01, pos_xy = collect_cell_alignment_scores(model, data_obj, soften_temp=soften_temp)
    except RuntimeError as e:
        print(f"[ALIGN] skip {name}: {e}")
        return

    if pos_xy is None or len(pos_xy) == 0:
        print(f"[ALIGN] skip {name}: missing cell positions")
        return

    svs_path = _find_svs(svs_root, name)
    gj_path = _find_geojson(cellgeojson_root, name)
    if not (svs_path and gj_path):
        print(f"[ALIGN] skip {name}: missing svs/geojson (svs_root={svs_root}, geojson_root={cellgeojson_root})")
        return

    slide = OpenSlide(str(svs_path))
    hint_level = None
    if cohort_pos_level_map:
        hint_level = cohort_pos_level_map.get(cohort_raw) or cohort_pos_level_map.get(cohort_label)
        if hint_level is None:
            norm = _normalize_cohort_key(cohort_raw)
            for key, lvl in cohort_pos_level_map.items():
                if _normalize_cohort_key(key) == norm:
                    hint_level = lvl
                    break
    if hint_level is None:
        hint_level = 1
    hint_level = int(max(0, min(int(hint_level), slide.level_count - 1)))
    ds_pos = float(slide.level_downsamples[hint_level])

    polys_lvl0 = load_cell_polygons(gj_path, slide=slide)
    if not polys_lvl0:
        slide.close()
        print(f"[ALIGN] skip {name}: polygons empty")
        return

    cell_xy_lvl0 = np.asarray(pos_xy, np.float32) * ds_pos
    cents_lvl0, valid_mask = polygon_centroids(polys_lvl0, return_mask=True)
    scores_on_valid = nn_assign_scores_to_polys(cell_xy_lvl0, score01, cents_lvl0, k=nn_k, gaussian_sigma=nn_sigma_lvl0)
    poly_scores = np.zeros((len(polys_lvl0),), np.float32)
    poly_scores[valid_mask] = scores_on_valid

    if target_level is None:
        target_level = hint_level

    out_dir = Path(out_root)
    cohort_dir = out_dir / f"{cohort_label}_{name}"
    out_png = cohort_dir / f"{Path(svs_path).stem}__align_overlay.png"

    overlay_mask_heatmap_on_wsi(
        svs_path=svs_path,
        polys_lvl0=polys_lvl0,
        poly_scores01=poly_scores,
        out_png=str(out_png),
        target_level=target_level,
        blend=blend,
        sigma_lvl0=sigma_lvl0,
        cmap_name=cmap_name,
        cmap_use_full=cmap_use_full,
        alpha_gamma=alpha_gamma,
        alpha_max=alpha_max,
        alpha_floor=alpha_floor,
        brighten_bg=brighten_bg,
        draw_all_outlines=draw_all_outlines,
        low_clip_q=low_clip_q,
        high_clip_q=high_clip_q,
        colorbar_h_ratio=colorbar_h_ratio,
        colorbar_w_px=colorbar_w_px,
        colorbar_pad_px=colorbar_pad_px,
        colorbar_title="alignment",
        fallback_disk_px=fallback_disk_px,
        grow_cover_px=grow_cover_px,
    )

    print(f"[ALIGN] saved: {out_png}")

def overlay_wsi_gate_for_sample(sample_name, cell_df, svs_root, cellgeojson_root, out_png,
                                scalar_col='gate_strength_norm', target_level=1, alpha_outline=80, k_vote=3):
    if (OpenSlide is None) or (Image is None):
        print("[WSI-GATE] OpenSlide or PIL not available; skip.")
        return
    svs_path = _find_svs(svs_root, sample_name)
    gj_path  = _find_geojson(cellgeojson_root, sample_name)
    if not (svs_path and os.path.exists(svs_path) and gj_path and os.path.exists(gj_path)):
        print(f"[WSI-GATE] skip {sample_name}: missing svs or geojson (svs_root={svs_root}, geojson_root={cellgeojson_root})")
        return

    slide = OpenSlide(str(svs_path))
    L = max(0, min(int(target_level), slide.level_count - 1))
    W, H = slide.level_dimensions[L]
    ds   = float(slide.level_downsamples[L])
    bg   = slide.read_region((0, 0), L, (W, H)).convert('RGB')

    sdf = cell_df[cell_df['sample_name']==sample_name].dropna(subset=['pos_x','pos_y', scalar_col]).copy()
    if sdf.empty:
        print(f"[WSI-GATE] {sample_name}: no cells with {scalar_col}."); slide.close(); return
    cell_xy_lvl0 = sdf[['pos_x','pos_y']].to_numpy(np.float32)
    scal = sdf[scalar_col].to_numpy(np.float32)

    polys_lvl0 = load_cell_polygons(gj_path, slide=slide)
    if not polys_lvl0:
        print(f"[WSI-GATE] {sample_name}: polygons empty."); slide.close(); return

    slide.close()

    scores, conf = _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, scal, k=k_vote)
    poly_scores = np.clip(scores * np.power(np.clip(conf, 0.0, 1.0), 0.5), 0, 1)
    overlay_mask_heatmap_on_wsi(
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
        colorbar_title='gate',
        fallback_disk_px=4,
        grow_cover_px=4,
        post_smooth_frac=0.0,
        alpha_smooth_frac=0.0,
    )
    print(f"[WSI-GATE] saved: {out_png}")
                                                                                                                                                                                                                                                                                              
def overlay_wsi_core_edge_for_sample(sample_name, cell_df, svs_root, cellgeojson_root, out_png,
                                     anchor_pid:int, prop_thr:float=0.7, k_vote=3, target_level=1):
    if (OpenSlide is None) or (Image is None):
        print("[WSI-CORE] OpenSlide or PIL not available; skip.")
        return

    svs_path = _find_svs(svs_root, sample_name)
    gj_path = _find_geojson(cellgeojson_root, sample_name)
    if not (svs_path and os.path.exists(svs_path) and gj_path and os.path.exists(gj_path)):
        print(f"[WSI-CORE] skip {sample_name}: missing svs or geojson (svs_root={svs_root}, geojson_root={cellgeojson_root})")
        return

    tag = f'P{int(anchor_pid)}'
    if tag not in cell_df.columns:
        print(f"[WSI-CORE] {sample_name}: missing {tag} (need neigh_proto_prop).")
        return

    sdf = cell_df[cell_df['sample_name'] == sample_name].dropna(subset=['pos_x', 'pos_y']).copy()
    if sdf.empty:
        print(f"[WSI-CORE] {sample_name}: no cells.")
        return

    cell_xy_lvl0 = sdf[['pos_x', 'pos_y']].to_numpy(np.float32)
    cell_core = np.clip(pd.to_numeric(sdf[tag], errors='coerce').fillna(0).to_numpy(np.float32), 0.0, 1.0)
    slide = OpenSlide(str(svs_path))
    polys_lvl0 = load_cell_polygons(gj_path, slide=slide)
    slide.close()
    if not polys_lvl0:
        print(f"[WSI-CORE] {sample_name}: polygons empty.")
        return

    base_path = Path(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    scores_core, conf_core = _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, cell_core, k=k_vote)
    core_heat = np.clip(scores_core * np.power(np.clip(conf_core, 0.0, 1.0), 0.5), 0, 1)
    if np.any(core_heat > 1e-6):
        core_path = base_path.with_name(f"{base_path.stem}_core.png")
        overlay_mask_heatmap_on_wsi(
            svs_path=svs_path,
            polys_lvl0=polys_lvl0,
            poly_scores01=core_heat,
            out_png=str(core_path),
            target_level=target_level,
            blend='sum',
            sigma_lvl0=32.0,
            cmap_name='Reds',
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
            colorbar_title=f"core P{anchor_pid}",
            fallback_disk_px=4,
            grow_cover_px=4,
            post_smooth_frac=0.0,
            alpha_smooth_frac=0.0,
        )
    else:
        core_path = None

    edge_scalar = np.clip(1.0 - cell_core, 0.0, 1.0)
    scores_edge, conf_edge = _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, edge_scalar, k=k_vote)
    edge_heat = np.clip(scores_edge * np.power(np.clip(conf_edge, 0.0, 1.0), 0.5), 0, 1)
    if np.any(edge_heat > 1e-6):
        edge_path = base_path.with_name(f"{base_path.stem}_edge.png")
        overlay_mask_heatmap_on_wsi(
            svs_path=svs_path,
            polys_lvl0=polys_lvl0,
            poly_scores01=edge_heat,
            out_png=str(edge_path),
            target_level=target_level,
            blend='sum',
            sigma_lvl0=32.0,
            cmap_name='Blues',
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
            colorbar_title=f"edge P{anchor_pid}",
            fallback_disk_px=4,
            grow_cover_px=4,
            post_smooth_frac=0.0,
            alpha_smooth_frac=0.0,
        )
    else:
        edge_path = None

    if core_path:
        try:
            shutil.copyfile(core_path, base_path)
        except Exception as e:
            print(f"[WARN] failed to produce legacy core-edge overlay: {e}")
    print(f"[WSI-CORE] saved: {[str(p) for p in [core_path, edge_path] if p]}")



def overlay_wsi_prototypes_for_sample(sample_name, cell_df, svs_root, cellgeojson_root, out_png,
                                      use_merged=True, target_level=1, k_vote=3,
                                      alpha_fill=200, alpha_outline=80, draw_outline=False,
                                      outline_color=(255, 255, 255), alpha_floor_frac=0.6,
                                      core_threshold=0.7, contour_color=(255, 255, 255, 240),
                                      contour_width=2):
    if (OpenSlide is None) or (Image is None):
        print("[WSI-PROTO] OpenSlide or PIL not available; skip.")
        return

    svs_path = _find_svs(svs_root, sample_name)
    gj_path = _find_geojson(cellgeojson_root, sample_name)
    if not (svs_path and os.path.exists(svs_path) and gj_path and os.path.exists(gj_path)):
        print(f"[WSI-PROTO] skip {sample_name}: missing svs or geojson (svs_root={svs_root}, geojson_root={cellgeojson_root})")
        return

    pid_col = "prototype_id_merged" if (use_merged and "prototype_id_merged" in cell_df.columns) else "prototype_id"
    sdf = cell_df[cell_df['sample_name'] == sample_name].dropna(subset=['pos_x', 'pos_y', pid_col]).copy()
    if sdf.empty:
        print(f"[WSI-PROTO] sample {sample_name}: no cells with positions/prototypes.")
        return

    cell_xy_lvl0 = sdf[['pos_x', 'pos_y']].to_numpy(np.float32)
    cell_pid = pd.to_numeric(sdf[pid_col], errors='coerce').fillna(-1).astype(int).to_numpy()
    uniq = sorted([int(p) for p in np.unique(cell_pid) if p >= 0])
    if not uniq:
        print(f"[WSI-PROTO] {sample_name}: no valid prototype ids.")
        return

    slide = OpenSlide(str(svs_path))
    polys_lvl0 = load_cell_polygons(gj_path, slide=slide)
    slide.close()
    if not polys_lvl0:
        print(f"[WSI-PROTO] {sample_name}: polygons empty.")
        return

    cmaps_seq = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'YlOrRd', 'YlGn', 'PuBu', 'BuPu', 'GnBu', 'OrRd']
    alpha_fill = float(max(0, min(alpha_fill, 255)))
    alpha_floor = float(max(0, min(alpha_fill * float(alpha_floor_frac), alpha_fill)))
    alpha_outline = int(max(0, min(alpha_outline, 255)))
    outline_rgba = tuple(list(outline_color[:3]) + [alpha_outline]) if outline_color else (255, 255, 255, alpha_outline)
    base_path = Path(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    outputs = []

    for idx, pid in enumerate(uniq):
        cell_scalar = (cell_pid == pid).astype(np.float32)
        scores, conf = _poly_scalar_scores(polys_lvl0, cell_xy_lvl0, cell_scalar, k=k_vote)
        poly_scores = np.clip(scores * np.power(np.clip(conf, 0.0, 1.0), 0.5), 0, 1)
        if not np.any(poly_scores > 1e-6):
            continue
        cmap_name = cmaps_seq[idx % len(cmaps_seq)]
        out_pid = base_path.with_name(f"{base_path.stem}_P{pid}.png")
        contour_polys = None
        if core_threshold is not None:
            core_mask = poly_scores >= float(core_threshold)
            if np.any(core_mask):
                contour_polys = [polys_lvl0[ii] for ii, flag in enumerate(core_mask) if flag and polys_lvl0[ii] is not None]
        overlay_mask_heatmap_on_wsi(
            svs_path=svs_path,
            polys_lvl0=polys_lvl0,
            poly_scores01=poly_scores,
            out_png=str(out_pid),
            target_level=target_level,
            blend='sum',
            sigma_lvl0=32.0,
            cmap_name=cmap_name,
            cmap_use_full=True,
            alpha_gamma=0.5,
            alpha_max=alpha_fill,
            alpha_floor=alpha_floor,
            brighten_bg=0.08,
            draw_all_outlines=bool(draw_outline),
            outline_rgba=outline_rgba,
            contour_polys=contour_polys,
            contour_rgba=contour_color,
            contour_width=contour_width,
            low_clip_q=8.0,
            high_clip_q=99.2,
            colorbar_h_ratio=0.45,
            colorbar_w_px=90,
            colorbar_pad_px=28,
            colorbar_title=f"P{pid}",
            fallback_disk_px=4,
            grow_cover_px=4,
            post_smooth_frac=0.0,
            alpha_smooth_frac=0.0,
        )
        outputs.append((pid, poly_scores.sum(), out_pid))

    if not outputs:
        print(f"[WSI-PROTO] {sample_name}: no prototype heatmaps generated.")
        return

    # copy dominant heatmap to legacy path
    dominant = max(outputs, key=lambda x: x[1])
    try:
        shutil.copyfile(dominant[2], base_path)
    except Exception as e:
        print(f"[WARN] failed to write legacy prototype overlay: {e}")
    print(f"[WSI-PROTO] saved: {[str(o[2]) for o in outputs]}")


def report_wsi_asset_status(cell_df, svs_root_map, cellgeojson_root_map, out_csv=None, allow_cohorts=None):
    allow_norm = None
    if allow_cohorts:
        allow_norm = {_normalize_cohort_key(c) for c in allow_cohorts}
    rows = []
    for sample, sdf in cell_df.groupby('sample_name'):
        cohort_raw = sdf['cohort'].iat[0] if 'cohort' in sdf.columns else None
        cohort_label = str(cohort_raw).strip() if cohort_raw is not None else 'NA'
        if allow_norm and _normalize_cohort_key(cohort_raw) not in allow_norm:
            continue
        svs_root = _resolve_cohort_path(svs_root_map, cohort_raw)
        gj_root = _resolve_cohort_path(cellgeojson_root_map, cohort_raw)
        svs_path = _find_svs(svs_root, sample) if svs_root else None
        gj_path = _find_geojson(gj_root, sample) if gj_root else None
        rows.append({
            'sample_name': sample,
            'cohort': cohort_label,
            'svs_root': svs_root,
            'geojson_root': gj_root,
            'svs_found': bool(svs_path) and os.path.exists(svs_path or ''),
            'geojson_found': bool(gj_path) and os.path.exists(gj_path or ''),
            'svs_path': svs_path,
            'geojson_path': gj_path,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print("\n[WSI-CHECK] asset availability summary")
    print(df[['svs_found', 'geojson_found']].value_counts())
    missing = df[(~df['svs_found']) | (~df['geojson_found'])]
    if not missing.empty:
        print("[WSI-CHECK] missing entries (up to 10):")
        print(missing.head(10)[['sample_name', 'cohort', 'svs_root', 'geojson_root']])
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df

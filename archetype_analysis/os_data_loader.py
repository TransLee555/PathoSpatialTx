"""OS-specific data loading helpers."""
from __future__ import annotations

import gc
import logging
import os
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from MHGL_ST_model import HierarchicalMultiModalGNN
from stage2_os_training import ProcessedImageDataset

from .cohort_config import CohortSpec

__all__ = [
    "initialize_model",
    "load_all_cohorts",
    "build_sample_dict",
    "get_sample",
    "extract_all_data_for_analysis",
]


def initialize_model(
    cell_in_channels: int,
    gene_in_channels: int,
    hidden_channels: int,
    embedding_dim: int,
    out_channels: int,
    num_shared_clusters: int,
    gnn_type: str,
    num_attention_heads: int,
    dropout_rate: float,
    num_intra_modal_layers: int,
    num_inter_modal_layers: int,
) -> HierarchicalMultiModalGNN:
    return HierarchicalMultiModalGNN(
        cell_in_channels,
        gene_in_channels,
        hidden_channels,
        embedding_dim,
        out_channels,
        num_shared_clusters,
        gnn_type,
        num_attention_heads,
        dropout_rate,
        num_intra_modal_layers,
        num_inter_modal_layers,
    )


def load_all_cohorts(
    cohort_specs: Optional[List[CohortSpec]] = None,
    allowlist: Optional[Iterable[str]] = None,
) -> Tuple[
    DataLoader,
    Dict[str, Tuple[ProcessedImageDataset, int]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, int],
]:
    print("\n[SETUP] Loading OS datasets...")
    cohort_specs = cohort_specs or []

    datasets: Dict[str, ProcessedImageDataset] = {}
    cohort_svs: Dict[str, str] = {}
    cohort_geojson: Dict[str, str] = {}
    cohort_pos_level: Dict[str, int] = {}

    specs_to_use = (
        [spec for spec in cohort_specs if (not allowlist) or spec.name in allowlist]
        if cohort_specs
        else []
    )
    if not specs_to_use:
        raise RuntimeError("OS pipeline requires explicit cohort specs; none were provided.")

    for spec in specs_to_use:
        try:
            ds = ProcessedImageDataset(
                root=str(Path(spec.processed_root).expanduser()),
                nucle_root=str(Path(spec.cell_root).expanduser()),
                gene_root=str(Path(spec.gene_root).expanduser()),
                svs_root=str(Path(spec.svs_root).expanduser()) if spec.svs_root else None,
                label_root=str(Path(spec.label.file).expanduser()) if spec.label else None,
                original_patch_size_level0=spec.pos_level or 512,
            )
            datasets[spec.name] = ds
        except Exception as exc:  # pragma: no cover
            logging.warning("Dataset init failed for cohort %s: %s", spec.name, exc)
            continue
        if spec.svs_root:
            cohort_svs[spec.name] = str(Path(spec.svs_root).expanduser())
        if spec.geojson_root:
            cohort_geojson[spec.name] = str(Path(spec.geojson_root).expanduser())
        if spec.pos_level is not None:
            cohort_pos_level[spec.name] = spec.pos_level

    def _safe_sample_name(data):
        try:
            sn = data.misc_info.get('sample_name', None)
        except Exception:
            sn = None
        if sn is None:
            sn = getattr(data, 'patient_id', None)
        if sn is None and hasattr(data, '__getitem__'):
            try:
                sn = data['name']
            except Exception:
                sn = None
        return str(sn).strip() if sn is not None else ""

    def patched_get_factory(og_get, cohort_name):
        def patched_get(self, idx):
            data = og_get(idx)
            sample_name = _safe_sample_name(data)
            data.patient_id = sample_name
            data.cohort = cohort_name
            if not hasattr(data, 'y') or data.y is None:
                data.y = torch.tensor([-1], dtype=torch.long)
            return data

        return patched_get

    class _OnDemandDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            ds, idx = self.items[i]
            return ds.get(idx)

    items: List[Tuple[ProcessedImageDataset, int]] = []
    sample_index: Dict[str, Tuple[ProcessedImageDataset, int]] = {}
    for name, ds in datasets.items():
        if allowlist and name not in allowlist:
            continue
        try:
            ds.get = types.MethodType(patched_get_factory(ds.get, name), ds)
            for i in range(len(ds)):
                g = None
                try:
                    g = ds.get(i)
                    sid = str(getattr(g, 'patient_id', '')).strip()
                    if sid:
                        items.append((ds, i))
                        sample_index[sid] = (ds, i)
                except Exception as exc:  # pragma: no cover
                    print(f"[WARN] skip {name} idx={i}: {exc}")
                finally:
                    del g
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] patch/get failed for {name}: {exc}")

    if not items:
        raise RuntimeError("No samples were found for the configured OS cohorts.")

    loader = DataLoader(
        _OnDemandDataset(items),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch[0],
    )
    print(f"  Total labeled samples: {len(loader.dataset)}")

    cohort_counts = {}
    for sid, (ds, idx) in sample_index.items():
        try:
            g = ds.get(idx)
            cohort_key = str(getattr(g, 'cohort', 'NA'))
            cohort_counts[cohort_key] = cohort_counts.get(cohort_key, 0) + 1
        except Exception:
            pass
    print("\n== Count by cohort in loader ==")
    total = 0
    for key in sorted(cohort_counts.keys()):
        val = cohort_counts[key]
        print(f"  {key:<7}: {val}")
        total += val
    print(f"  TOTAL  : {total}")

    return loader, sample_index, cohort_svs, cohort_geojson, cohort_pos_level


def build_sample_dict(loader: DataLoader) -> Dict[str, torch.Tensor]:
    table = {}
    for d in loader:
        sid = d.patient_id[0] if isinstance(d.patient_id, list) else d.patient_id
        table[str(sid).strip()] = d.to('cpu')
    return table


def get_sample(sample_dict: Dict[str, torch.Tensor], name: str):
    key = str(name).strip()
    if key in sample_dict:
        return sample_dict[key]
    key2 = key.lstrip('0') or '0'
    if key2 in sample_dict:
        return sample_dict[key2]
    key3 = key2.zfill(3)
    return sample_dict.get(key3, None)


def extract_all_data_for_analysis(model, loader, device: torch.device):
    print("\n[SETUP 2/2] Extracting all raw features and embeddings...")
    model.eval()
    store = {
        "cell": {"feats": [], "embeds": [], "pos": [], "nid": [], "sample": [], "cohort": [],
                  "surv_t": [], "surv_e": []},
        "gene": {"feats": [], "embeds": [], "nid": [], "sample": [], "cohort": [],
                  "surv_t": [], "surv_e": []},
    }
    for d in tqdm(loader, desc="  Processing samples"):
        d = d.to(device)
        ce, ge = model.analysis_forward_with_gate(d)
        sid = d.patient_id[0] if isinstance(d.patient_id, list) else d.patient_id
        coh = d.cohort[0] if isinstance(d.cohort, list) else d.cohort
        surv_time = float(getattr(d, 't', torch.tensor([float('nan')], device=device)).view(-1)[0].item())
        surv_event = float(getattr(d, 'e', torch.tensor([float('nan')], device=device)).view(-1)[0].item())

        if ce is not None and ce.numel() > 0:
            n = ce.size(0)
            store['cell']['feats'].append(d['cell'].x.detach().cpu())
            store['cell']['embeds'].append(ce.detach().cpu())
            if hasattr(d['cell'], 'pos'):
                store['cell']['pos'].append(d['cell'].pos.detach().cpu())
            else:
                store['cell']['pos'].append(torch.full((n, 2), np.nan))
            store['cell']['nid'].append(torch.arange(n))
            store['cell']['sample'].extend([sid] * n)
            store['cell']['cohort'].extend([coh] * n)
            store['cell']['surv_t'].extend([surv_time] * n)
            store['cell']['surv_e'].extend([surv_event] * n)

        if ge is not None and ge.numel() > 0:
            n = ge.size(0)
            store['gene']['feats'].append(d['gene'].x.detach().cpu())
            store['gene']['embeds'].append(ge.detach().cpu())
            store['gene']['nid'].append(torch.arange(n))
            store['gene']['sample'].extend([sid] * n)
            store['gene']['cohort'].extend([coh] * n)
            store['gene']['surv_t'].extend([surv_time] * n)
            store['gene']['surv_e'].extend([surv_event] * n)

    dfs = {}
    for S in ['cell', 'gene']:
        if not store[S]['embeds']:
            dfs[S] = pd.DataFrame()
            continue
        embeds = torch.cat(store[S]['embeds']).numpy()
        df = pd.DataFrame(embeds, columns=[f'embed_{i}' for i in range(embeds.shape[1])])
        feats = torch.cat(store[S]['feats']).numpy()
        col_prefix = 'raw_feat_' if S == 'cell' else 'gene_expr_'
        feat_cols = [f'{col_prefix}{i}' for i in range(feats.shape[1])]
        df = pd.concat([pd.DataFrame(feats, columns=feat_cols), df], axis=1)
        nid = torch.cat(store[S]['nid']).numpy()
        df['node_idx'] = nid
        df['sample_name'] = store[S]['sample']
        df['cohort'] = store[S]['cohort']
        df['survival_time'] = store[S]['surv_t']
        df['survival_event'] = store[S]['surv_e']
        if S == 'cell':
            pos = torch.cat(store['cell']['pos']).numpy()
            if pos.shape[1] >= 2:
                df[['pos_x', 'pos_y']] = pos[:, :2]
        dfs[S] = df

    return dfs['cell'], dfs['gene']

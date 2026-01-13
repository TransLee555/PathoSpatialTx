"""Data loading helpers for MHGL-ST stage3 analysis."""
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
from stage2_pCR_training import ProcessedImageDataset

from .cohort_config import CohortSpec, canon_id, load_labels_for_cohort

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
    model = HierarchicalMultiModalGNN(
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
    return model


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
    """Load ProcessedImageDatasets for all configured cohorts."""

    print("\n[SETUP] Loading datasets...")
    cohort_specs = cohort_specs or []

    # Legacy defaults (kept for backward compatibility)
    yale_root = "/DATA/linzhiquan/lzq/HEST-main/after_bib_result_v2/nat/yale_512/"
    yale_labels_path = (
        "/DATA/linzhiquan/lzq/Yale_NAT/"
        "Yale_trastuzumab_response_cohort_metadata_clean.csv"
    )
    yale_data_root = os.path.join(yale_root, "processed_data_2.5")
    yale_nucle_root = os.path.join(yale_root, "cell_output")
    yale_gene_root = os.path.join(yale_root, "pred")

    purdue_root_base = (
        "/DATA/linzhiquan/lzq/HEST-main/after_bib_result_v2/nat/purdue_her2_512/"
    )
    purdue_labels_path = "/DATA/linzhiquan/lzq/purdue/Purdue_NC.xlsx"
    purdue_data_root = os.path.join(purdue_root_base, "processed_data_2.5")
    purdue_nucle_root = os.path.join(purdue_root_base, "cell_output")
    purdue_gene_root = os.path.join(purdue_root_base, "pred")

    jch_root_base = "/DATA/linzhiquan/lzq/HEST-main/after_bib_result_v2/nat/jch_1024/"
    jch_data_root = os.path.join(jch_root_base, "temp_processed_data_2.5")
    jch_nucle_root = os.path.join(jch_root_base, "cell_output")
    jch_gene_root = os.path.join(jch_root_base, "pred")
    jch_label_path = "/DATA/linzhiquan/lzq/jch_her2/jch_label.csv"
    jch_pos_level = 1024

    patient_to_label: Dict[str, int] = {}
    patient_to_cohort: Dict[str, str] = {}
    datasets: Dict[str, ProcessedImageDataset] = {}
    cohort_pos_level: Dict[str, int] = {}
    cohort_svs: Dict[str, str] = {}
    cohort_geojson: Dict[str, str] = {}

    specs_to_use = (
        [spec for spec in cohort_specs if (not allowlist) or spec.name in allowlist]
        if cohort_specs
        else []
    )

    if specs_to_use:
        for spec in specs_to_use:
            try:
                label_map = load_labels_for_cohort(spec)
                for pid, val in label_map.items():
                    patient_to_label[pid] = val
                    patient_to_cohort[pid] = spec.name
            except Exception as exc:  # pragma: no cover
                logging.warning(
                    "Label parsing failed for cohort %s: %s", spec.name, exc
                )
            try:
                ds = ProcessedImageDataset(
                    root=str(Path(spec.processed_root).expanduser()),
                    nucle_root=str(Path(spec.cell_root).expanduser()),
                    gene_root=str(Path(spec.gene_root).expanduser()),
                    label_root=str(Path(spec.label.file).expanduser())
                    if spec.label
                    else None,
                    svs_root=str(Path(spec.svs_root).expanduser())
                    if spec.svs_root
                    else None,
                    original_patch_size_level0=spec.pos_level or 512,
                )
                datasets[spec.name] = ds
            except Exception as exc:  # pragma: no cover
                logging.warning(
                    "Dataset init failed for cohort %s: %s", spec.name, exc
                )
            if spec.svs_root:
                cohort_svs[spec.name] = str(Path(spec.svs_root).expanduser())
            if spec.geojson_root:
                cohort_geojson[spec.name] = str(Path(spec.geojson_root).expanduser())
            if spec.pos_level is not None:
                cohort_pos_level[spec.name] = spec.pos_level
    else:
        # fallback to legacy hardcoded paths
        have_yale = False
        have_purdue = False
        try:
            yale_labels = pd.read_csv(yale_labels_path)
            for _, r in yale_labels.iterrows():
                pid = str(r.get("Patient")).strip()
                responder = str(r.get("Responder")).strip().lower()
                patient_to_label[pid] = 1 if responder == "responder" else 0
                patient_to_cohort[pid] = "Yale"
            have_yale = True
        except Exception as exc:  # pragma: no cover
            print("[WARN] Yale labels load failed:", exc)

        try:
            purdue_labels = pd.read_excel(purdue_labels_path, sheet_name=0)
            for _, r in purdue_labels.iterrows():
                pid_norm = canon_id(r.get("patients"))
                pid_zeropad = pid_norm.zfill(3)
                pcr = int(r.get("pCR")) if pd.notna(r.get("pCR")) else 0
                for key in (pid_norm, pid_zeropad):
                    patient_to_label[key] = pcr
                    patient_to_cohort[key] = "Purdue"
            have_purdue = True
        except Exception as exc:  # pragma: no cover
            print("[WARN] Purdue labels load failed:", exc)

        if have_yale:
            try:
                yale_ds = ProcessedImageDataset(
                    root=yale_data_root,
                    nucle_root=yale_nucle_root,
                    gene_root=yale_gene_root,
                    svs_root="/DATA/linzhiquan/lzq/Yale_NAT/SVS",
                    label_root=yale_labels_path,
                )
                datasets["Yale"] = yale_ds
                cohort_svs["Yale"] = "/DATA/linzhiquan/lzq/Yale_NAT/SVS"
                cohort_geojson["Yale"] = (
                    "/DATA/linzhiquan/lzq/HEST-main/after_bib_result_v2/nat/yale_512/cell_output"
                )
            except Exception as exc:  # pragma: no cover
                print("[WARN] Yale dataset init failed:", exc)
        if have_purdue:
            try:
                purdue_ds = ProcessedImageDataset(
                    root=purdue_data_root,
                    nucle_root=purdue_nucle_root,
                    gene_root=purdue_gene_root,
                    svs_root="/DATA/linzhiquan/lzq/purdue/Purdue_Her2_wsi",
                    label_root=purdue_labels_path,
                    original_patch_size_level0=512,
                )
                datasets["Purdue"] = purdue_ds
                cohort_svs["Purdue"] = (
                    "/DATA/linzhiquan/lzq/purdue/Purdue_Her2_wsi"
                )
                cohort_geojson["Purdue"] = (
                    "/DATA/linzhiquan/lzq/HEST-main/after_bib_result_v2/nat/purdue_her2_512/cell_output"
                )
            except Exception as exc:  # pragma: no cover
                print("[WARN] Purdue dataset init failed:", exc)

        try:
            jch_ds = ProcessedImageDataset(
                root=jch_data_root,
                nucle_root=jch_nucle_root,
                gene_root=jch_gene_root,
                label_root=jch_label_path,
                svs_root="/DATA/linzhiquan/lzq/jch_her2/wsi/",
                original_patch_size_level0=jch_pos_level,
            )
            jch_labels_df = pd.read_csv(jch_label_path, dtype={"sample": str})
            jch_labels_df["sample"] = jch_labels_df["sample"].astype(str).str.strip()
            for _, r in jch_labels_df.iterrows():
                pid = str(r["sample"]).strip()
                pcr = int(r["pCR"]) if pd.notna(r["pCR"]) else 0
                patient_to_label[pid] = pcr
                patient_to_cohort[pid] = "JCH"
            datasets["JCH"] = jch_ds
            cohort_svs["JCH"] = "/DATA/linzhiquan/lzq/jch_her2/wsi/"
            cohort_geojson["JCH"] = (
                "/DATA/linzhiquan/lzq/HEST-main/after_bib_result_v2/nat/jch_1024/cell_output"
            )
            cohort_pos_level["JCH"] = jch_pos_level
        except Exception as exc:  # pragma: no cover
            print("[WARN] JCH init failed:", exc)

    def _safe_sample_name(data):
        try:
            sn = data.misc_info.get("sample_name", None)
        except Exception:
            sn = None
        if sn is None:
            sn = getattr(data, "patient_id", None)
        if sn is None and hasattr(data, "__getitem__"):
            try:
                sn = data["name"]
            except Exception:
                sn = None
        return str(sn).strip() if sn is not None else ""

    def patched_get_factory(og_get, cohort_name):
        def patched_get(self, idx):
            data = og_get(idx)
            sample_name = _safe_sample_name(data)
            if cohort_name == "Purdue":
                raw = sample_name.split("_")[0] if "_" in sample_name else sample_name
                p_id = canon_id(raw)
                if p_id not in patient_to_label:
                    alt = p_id.zfill(3)
                    if alt in patient_to_label:
                        p_id = alt
            else:
                p_id = sample_name
            data.y = torch.tensor([patient_to_label.get(p_id, -1)], dtype=torch.long)
            data.patient_id = p_id
            data.cohort = patient_to_cohort.get(p_id, cohort_name)
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
                    if int(getattr(g, "y", torch.tensor([-1]))[0].item()) != -1:
                        sid = str(getattr(g, "patient_id", "")).strip()
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
        raise RuntimeError(
            "No labeled samples were found. Please check dataset paths/labels."
        )

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
            cohort_key = str(getattr(g, "cohort", "NA"))
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
        table[str(sid).strip()] = d.to("cpu")
    return table


def get_sample(sample_dict: Dict[str, torch.Tensor], name: str):
    key = str(name).strip()
    if key in sample_dict:
        return sample_dict[key]
    key2 = canon_id(key)
    if key2 in sample_dict:
        return sample_dict[key2]
    key3 = key2.zfill(3)
    return sample_dict.get(key3, None)


@torch.no_grad()
def extract_all_data_for_analysis(model, loader, device: torch.device):
    """Dump cell/gene features & embeddings into DataFrames."""

    print("\n[SETUP 2/2] Extracting all raw features and embeddings...")
    model.eval()
    store = {
        "cell": {
            "feats": [],
            "embeds": [],
            "pos": [],
            "nid": [],
            "sample": [],
            "label": [],
            "cohort": [],
        },
        "gene": {
            "feats": [],
            "embeds": [],
            "nid": [],
            "sample": [],
            "label": [],
            "cohort": [],
        },
    }
    for d in tqdm(loader, desc="  Processing samples"):
        d = d.to(device)
        ce, ge = model.analysis_forward_with_gate(d)
        sid = d.patient_id[0] if isinstance(d.patient_id, list) else d.patient_id
        lbl = int(d.y.item())
        coh = d.cohort[0] if isinstance(d.cohort, list) else d.cohort

        if ce is not None and ce.numel() > 0:
            n = ce.size(0)
            store["cell"]["feats"].append(d["cell"].x.detach().cpu())
            store["cell"]["embeds"].append(ce.detach().cpu())
            if hasattr(d["cell"], "pos"):
                store["cell"]["pos"].append(d["cell"].pos.detach().cpu())
            else:
                store["cell"]["pos"].append(torch.full((n, 2), np.nan))
            store["cell"]["nid"].append(torch.arange(n))
            store["cell"]["sample"].extend([sid] * n)
            store["cell"]["label"].extend([lbl] * n)
            store["cell"]["cohort"].extend([coh] * n)

        if ge is not None and ge.numel() > 0:
            n = ge.size(0)
            store["gene"]["feats"].append(d["gene"].x.detach().cpu())
            store["gene"]["embeds"].append(ge.detach().cpu())
            store["gene"]["nid"].append(torch.arange(n))
            store["gene"]["sample"].extend([sid] * n)
            store["gene"]["label"].extend([lbl] * n)
            store["gene"]["cohort"].extend([coh] * n)

    dfs = {}
    for section in ["cell", "gene"]:
        if not store[section]["embeds"]:
            dfs[section] = pd.DataFrame()
            continue
        embeds = torch.cat(store[section]["embeds"]).numpy()
        df = pd.DataFrame(embeds, columns=[f"embed_{i}" for i in range(embeds.shape[1])])
        feats = torch.cat(store[section]["feats"]).numpy()
        col_prefix = "raw_feat_" if section == "cell" else "gene_expr_"
        feat_cols = [f"{col_prefix}{i}" for i in range(feats.shape[1])]
        df = pd.concat([pd.DataFrame(feats, columns=feat_cols), df], axis=1)
        nid = torch.cat(store[section]["nid"]).numpy()
        df["node_idx"] = nid
        df["sample_name"] = store[section]["sample"]
        df["label"] = store[section]["label"]
        df["cohort"] = store[section]["cohort"]
        if section == "cell":
            pos = torch.cat(store["cell"]["pos"]).numpy()
            if pos.shape[1] >= 2:
                df[["pos_x", "pos_y"]] = pos[:, :2]
        dfs[section] = df

    return dfs["cell"], dfs["gene"]

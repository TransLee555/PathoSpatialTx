"""Config utilities for MHGL-ST stage3 analysis."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

__all__ = [
    "LabelSpec",
    "CohortSpec",
    "load_cohort_specs",
    "load_labels_for_cohort",
    "canon_id",
]


@dataclass
class LabelSpec:
    file: str
    format: str = "csv"  # csv or excel
    patient_column: str = "patient"
    value_column: str = "pCR"
    sheet_name: Optional[str] = None
    positive_values: List[str] = field(
        default_factory=lambda: ["responder", "1", "true", "positive"]
    )
    value_mapping: Optional[Dict[str, int]] = None


@dataclass
class CohortSpec:
    name: str
    processed_root: str
    cell_root: str
    gene_root: str
    svs_root: Optional[str] = None
    geojson_root: Optional[str] = None
    label: Optional[LabelSpec] = None
    pos_level: Optional[int] = None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cohort_specs(
    config_path: Optional[Path],
) -> Tuple[List[CohortSpec], Optional[str], Optional[str], Optional[List[str]]]:
    """Load cohort specifications from a JSON config file."""
    if config_path is None:
        return [], None, None, None
    cfg = _load_json(config_path)
    model_path = cfg.get("model_path")
    output_dir = cfg.get("output_dir")
    allowlist = cfg.get("cohort_allowlist")
    specs: List[CohortSpec] = []
    for entry in cfg.get("cohorts", []):
        label_entry = entry.get("label")
        label_spec = None
        if label_entry:
            label_spec = LabelSpec(
                file=label_entry["file"],
                format=label_entry.get("format", "csv"),
                patient_column=label_entry.get("patient_column", "patient"),
                value_column=label_entry.get("value_column", "pCR"),
                sheet_name=label_entry.get("sheet_name"),
                positive_values=label_entry.get(
                    "positive_values", ["responder", "1", "true", "positive"]
                ),
                value_mapping=label_entry.get("value_mapping"),
            )
        specs.append(
            CohortSpec(
                name=entry["name"],
                processed_root=entry["processed_root"],
                cell_root=entry["cell_root"],
                gene_root=entry["gene_root"],
                svs_root=entry.get("svs_root"),
                geojson_root=entry.get("geojson_root"),
                label=label_spec,
                pos_level=entry.get("pos_level"),
            )
        )
    return specs, model_path, output_dir, allowlist


def _clean_id(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_label_value(raw, spec: LabelSpec) -> int:
    if spec.value_mapping:
        key = str(raw).strip()
        return int(spec.value_mapping.get(key, spec.value_mapping.get("default", 0)))
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    key = str(raw).strip().lower()
    positives = {str(v).strip().lower() for v in spec.positive_values}
    return 1 if key in positives else 0


def load_labels_for_cohort(spec: CohortSpec) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    if not spec.label:
        return labels
    label_spec = spec.label
    label_path = Path(label_spec.file).expanduser()
    if not label_path.exists():
        logging.warning(
            "Label file for cohort %s not found: %s", spec.name, label_path
        )
        return labels
    if label_spec.format.lower() == "excel":
        df = pd.read_excel(label_path, sheet_name=label_spec.sheet_name or 0)
    else:
        df = pd.read_csv(label_path)
    pcol = label_spec.patient_column
    vcol = label_spec.value_column
    if pcol not in df.columns or vcol not in df.columns:
        logging.warning(
            "Label file %s missing required columns (%s, %s)", label_path, pcol, vcol
        )
        return labels
    for _, row in df.iterrows():
        pid = _clean_id(row[pcol])
        if not pid:
            continue
        labels[pid] = _parse_label_value(row[vcol], label_spec)
    return labels


def canon_id(value: Union[str, Iterable[str]]) -> str:
    """Canonicalize IDs: strip whitespace and leading zeros."""
    s = str(value).strip()
    if not s:
        return s
    z = s.lstrip("0")
    return z if z else "0"

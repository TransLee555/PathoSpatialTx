# PathoSpatialTx
Decoding molecularly anchored spatial archetypes of tumor–immune ecology from routine H&E whole-slide images.

## Overview
PathoSpatialTx is a three-stage framework:
1) **Histology-to-omics calibration**: an ST-supervised graph predictor imputes immune gene programs from H&E patches.
2) **Archetype-aware heterogeneous graph learning**: a hetero-GNN integrates inferred immune programs with cell morphology to learn endpoint-linked spatial archetypes.
3) **Ecological readout & cross-modal validation**: archetypes are read out as slide-localizable ecology maps and contextualized in external single-cell and bulk cohorts.

This repository contains training/inference code, configurations, and pretrained PathoSpatialTx checkpoints (pCR and OS).

## Data preparation (Stage I prerequisite)

PathoSpatialTx Stage I trains an ST-supervised histology-to-omics predictor using paired **ST sections + matched H&E images**.
This repository provides preprocessing, training, and evaluation scripts (Stages I–III), but does not redistribute raw WSIs or patient-level private data.

### Spatial transcriptomics (ST) atlas (n=104 paired sections)
We curated 104 publicly available breast-cancer ST sections with matched H&E images from:
- Mendeley Data dataset (ST v1): https://data.mendeley.com/datasets/29ntw7sh4r/5
- 10x Genomics Visium public datasets: https://www.10xgenomics.com/
- NCBI GEO accessions: GSE243275, GSE242311, GSE203612, GSE190870

**Required inputs per section**
- ST expression matrix + spot coordinates (as provided by the data source)
- the **paired H&E image** for the same section (required for spot-to-histology alignment)

### Recommended: use our curated Stage I index (no raw images redistributed)
Public ST sources are heterogeneous in formats and metadata. To reduce friction, we provide a **curated Stage I manifest/index** that standardizes:
- per-section identifiers,
- expected file naming,
- spot-to-histology mapping metadata,
- and training splits (if applicable).

You still need to download the underlying public ST/H&E files from their original sources, but the manifest makes the preprocessing deterministic.

See: `data/manifests/st104_manifest.csv` (or `docs/data_preparation.md`).

### Optional: run preprocessing from raw downloads (advanced)
We also provide preprocessing code to reproduce the Stage I training table from raw public downloads. Due to heterogeneity across sources, users may need minor dataset-specific adjustments.

The preprocessing script performs:
1) reading per-section ST expression + coordinates,
2) aligning spots to the H&E coordinate system,
3) extracting patch features with a chosen patch encoder,
4) exporting a standardized training table for Stage I.

Example (adjust to your entrypoint):
```bash
python -m scripts.stage1_prep_st_atlas \
  --manifest data/manifests/st104_manifest.csv \
  --st_root /path/to/st_downloads \
  --he_root /path/to/paired_he_images \
  --out_root data/st_atlas_prepared \
  --encoder conch

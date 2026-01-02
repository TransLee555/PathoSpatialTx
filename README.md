# PathoSpatialTx
Decoding molecularly anchored spatial archetypes of tumor–immune ecology from routine H&E whole-slide images.

## Overview
PathoSpatialTx is a three-stage framework:
1) **Histology-to-omics calibration**: an ST-supervised graph predictor imputes immune gene programs from H&E patches.
2) **Archetype-aware heterogeneous graph learning**: a hetero-GNN integrates inferred immune programs with cell morphology to learn endpoint-linked spatial archetypes.
3) **Ecological readout & cross-modal validation**: archetypes are read out as slide-localizable ecology maps and contextualized in external single-cell and bulk cohorts.

This repository contains training/inference code, configurations, and pretrained PathoSpatialTx checkpoints (pCR and OS).

## Quickstart (minimal)
### 1) Environment
```bash
conda env create -f env/environment.yml
conda activate pathospatialtx

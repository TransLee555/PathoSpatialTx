# PathoSpatialTx Pipeline Overview

## 0. Environment Setup
```bash
conda create -n PST python=3.9 -y
conda activate PST
```

Install PyTorch (example for CUDA 11.6):
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

Install additional packages:
```bash
pip install \
  opencv-python==4.12.0.88 \
  openslide-python==1.4.2 \
  histomicstk==1.3.14 \
  torch-geometric==2.6.1 \
  numpy==1.26.4 \
  transformers==4.28.0 \
  geopandas==0.13.2 \
  lifelines==0.30.0 \
  pandas==2.2.2 \
  scikit-learn==1.5.2
```
Install `openslide-bin` from conda-forge if needed:
```bash
conda install -c conda-forge openslide-bin
```

## Quick Start

Minimal example commands using the checkpoints in `./checkpoints`:
```bash
# Stage1 inference with pretrained IR-SVG predictor
python stage1_IRSVGs_predictor.py \
  --patch-root /path/to/patch_output \
  --output-dir /path/to/gene_pred_csv \
  --model-checkpoint ./checkpoints/stage1_IRSVGs_predictor.pth \
  --cache-dir /path/to/cache_dir \
  --gene-columns /path/to/gene_ensembl_ids.txt \
  --graph-type radius --radius 0.1 --use-edge-attr \
  --batch-size 1 --device cuda:0 \
  --hf-token <HF_TOKEN> \
  --progress

# Stage2 OS prediction with pretrained checkpoint
python stage2_os_predict.py \
  --root-dir /path/to/os/processed_data \
  --cell-output-dir /path/to/cell_output \
  --gene-dir /path/to/gene_pred_csv \
  --svs-dir /path/to/wsi_root \
  --checkpoint ./checkpoints/stage2_os_predict.pth \
  --batch-size 1 --num-workers 4 \
  --prediction-output /path/to/predictions/os_predictions.csv

# Stage2 pCR prediction with pretrained checkpoint
python stage2_pCR_predict.py \
  --root-dir /path/to/pcr_processed \
  --cell-output-dir /path/to/cell_output \
  --gene-dir /path/to/gene_pred_csv \
  --svs-dir /path/to/wsi_root \
  --checkpoint ./checkpoints/stage2_pCR_predict.pt \
  --prediction-output /path/to/predictions/pcr_predictions.csv
```

---

## Stage 1 - Patch-Level IR-SVG Pipeline

### 1.1 Patch Extraction
```bash
python stage1_patch_extract.py \
  --root /path/to/wsi_root \
  --output-dir /path/to/patch_output \
  --patch-size 512 \
  --level 1 \
  --overlap-threshold 0.2 \
  --skip-existing
```

### 1.2 IR-SVG Predictor Training
```bash
python stage1_IRSVGs_predictor_training.py \
  --patch-root /path/to/patch_output \
  --train-csv /path/to/stage1_train.csv \
  --val-csv /path/to/stage1_val.csv \
  --test-csv /path/to/stage1_test.csv \
  --cache-dir /path/to/cache_training_embeddings \
  --graph-type radius --radius 0.1 --use-edge-attr \
  --batch-size-train 2 --batch-size-eval 1 \
  --epochs 300 --learning-rate 1e-3 \
  --device cuda:0 \
  --conch-model-id conch_ViT-B-16 \
  --conch-checkpoint hf_hub:MahmoodLab/conch \
  --hf-token <HF_TOKEN> \
  --checkpoint-dir /path/to/checkpoints_stage1 \
  --save-prefix stage1_regressor \
  --log-level INFO
```

### 1.3 Gene Prediction (Inference)
```bash
python stage1_IRSVGs_predictor.py \
  --patch-root /path/to/patch_output \
  --output-dir /path/to/gene_pred_csv \
  --model-checkpoint ./checkpoints/stage1_IRSVGs_predictor.pth \
  --cache-dir /path/to/cache_dir \
  --gene-columns /path/to/gene_ensembl_ids.txt \
  --graph-type radius --radius 0.1 --use-edge-attr \
  --batch-size 1 --device cuda:0 \
  --hf-token <HF_TOKEN> \
  --progress
```

---

## Stage 2 - Cell-Level Graph + Survival/pCR Models

### 2.0 CellViT Features

CellViT-based nuclei embeddings are generated with the public repo https://github.com/TIO-IKIM/CellViT. Clone that project, export embeddings into `--cell-output-dir`, and reuse them for Stage 2/3 scripts. Exported features must be written to `--cell-output-dir` so that Stage 2 training and inference scripts can reuse them.

### 2.1 Cell Feature Extraction
```bash
python stage2_cell_feature_extract.py \
  --cell-output-dir /path/to/cell_output \
  --svs-dir /path/to/wsi_root \
  --physical-patch-size-um 2048 \
  --overlap-level0 0 \
  --upsample-factor 2 \
  --svs-level 3 \
  --num-processes 8 \
  --dedup-threshold 5 \
  --output-subdir nuclei_features \
  --log-level INFO
```

### 2.2 OS Model Training
```bash
python stage2_os_training.py \
  --root-dir /path/to/os/processed_data \
  --cell-output-dir /path/to/cell_output \
  --gene-dir /path/to/gene_pred_csv \
  --svs-dir /path/to/wsi_root \
  --patch-size-level0 512 \
  --val-split 0.2 --num-time-bins 5 \
  --learning-rate 1e-4 --epochs 200 \
  --graph-augment-rotation-prob 0.2 \
  --graph-augment-flip-prob 0.2 \
  --device auto --preferred-gpu 0 \
  --log-level INFO
```

### 2.3 OS Prediction
```bash
python stage2_os_predict.py \
  --root-dir /path/to/os/processed_data \
  --cell-output-dir /path/to/cell_output \
  --gene-dir /path/to/gene_pred_csv \
  --svs-dir /path/to/wsi_root \
  --checkpoint ./checkpoints/stage2_os_predict.pth \
  --batch-size 1 --num-workers 4 \
  --prediction-output /path/to/predictions/os_predictions.csv
```

### 2.4 pCR Model Training
```bash
python stage2_pCR_training.py \
  --root-dir /path/to/pcr_processed \
  --cell-output-dir /path/to/cell_output \
  --gene-dir /path/to/gene_pred_csv \
  --svs-dir /path/to/wsi_root \
  --patch-size-level0 512 --val-split 0.2 \
  --hidden-dim 64 --embedding-dim 32 \
  --num-shared-clusters 5 \
  --gnn-type Transformer --num-heads 4 \
  --dropout 0.5 --num-intra-layers 3 --num-inter-layers 2 \
  --learning-rate 1e-4 --epochs 500 \
  --cluster-lambda 0.05 \
  --device cuda:0 \
  --checkpoint-dir /path/to/checkpoints_stage2_pcr \
  --save-prefix stage2_pcr
```

### 2.5 pCR Prediction
```bash
python stage2_pCR_predict.py \
  --root-dir /path/to/pcr_processed \
  --cell-output-dir /path/to/cell_output \
  --gene-dir /path/to/gene_pred_csv \
  --svs-dir /path/to/wsi_root \
  --checkpoint ./checkpoints/stage2_pCR_predict.pt \
  --prediction-output /path/to/predictions/pcr_predictions.csv
```

---

## Stage 3 - Archetype Analysis & Validation

### 3.0 Archetype Analysis

**OS version:**
```bash
python stage3_OS_archetypes_analysis.py \
  --cohort-config /path/to/os_cohort.json \
  --model-path /path/to/os_checkpoint.pt \
  --output-dir /path/to/stage3_os_run \
  --gene-id-path /path/to/gene_ensembl_ids.txt \
  --enable-wsi-overlays \
  --spatial-cache-fullres
```

**pCR version:**
```bash
python stage3_pCR_archetypes_analysis.py \
  --cohort-config /path/to/pcr_cohort.json \
  --model-path /path/to/mhglst_checkpoint.pt \
  --output-dir /path/to/stage3_pcr_run \
  --gene-id-path /path/to/gene_ensembl_ids.txt \
  --core-thr 0.4 --interface-thr 0.4 \
  --enable-gene-sets
```

### 3.1 Cross-Modal Validation (scRNA/Bulk)

**Signature Mapping**
```bash
Rscript stage3_OS_sc_archetypes_mapping.R \
  --analysis-root /path/to/stage3_os_run \
  --gene-mapping /path/to/gene_ensembl_ids.txt \
  --cell-counts-csv /path/to/archetype_celltype_counts_raw.csv
```
Use `stage3_pCR_sc_archetypes_mapping.R` for the pCR pipeline.

**Full scRNA + Bulk Workflow**
```bash
Rscript stage3_OS_sc_archetypes_workflow.R \
  --base-path /path/to/stage3_os_run \
  --sc-dir /path/to/10x_scRNA_dir \
  --sc-metadata metadata.csv \
  --bulk-expr /path/to/bulk_expression.csv \
  --bulk-pheno /path/to/bulk_pheno.csv \
  --proto-signature /path/to/interface_core_signature_means.csv \
  --prior-comp /path/to/cell_prototype_composition.csv
```
`stage3_pCR_sc_archetypes_workflow.R` is analogous.

Run `Rscript <script> --help` for full option descriptions.

---

## Notes
- Replace `/path/to/...` with actual locations for your data, checkpoints, and outputs.
- Protect HuggingFace tokens and any proprietary paths.
- Adjust `--device`, `--batch-size`, worker counts, etc., according to available hardware.
- Confirm OpenSlide libraries are available before enabling `--enable-wsi-overlays`.

---

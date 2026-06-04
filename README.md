# LungSimNet4D

## Anatomy-Informed Synthetic Thoracic Imaging Framework for Lung Cancer Research

LungSimNet4D is a reproducible synthetic thoracic imaging framework designed for the generation, validation, and AI-driven analysis of pulmonary nodules. The framework combines anatomy-aware phantom generation, procedural lesion injection, CT-like voxelization, physics-based validation using GATE, and downstream segmentation and classification workflows.

The framework was developed to support synthetic data generation for Non-Small Cell Lung Cancer (NSCLC) research while enabling reproducible experimentation for segmentation, detection, and classification tasks.

---

## Key Features

- Anatomical phantom generation and variation
- Procedural lung lesion placement
- CT-like volume generation
- Lesion mask generation
- GATE-compatible voxel phantom export
- Physics-based validation using GATE 9.0
- ROOT-based detector hit analysis
- Dataset manifest management
- Train / Validation / Test split generation
- Dataset quality control and visualization
- 3D segmentation workflow
- 3D classification workflow

---

## Dataset and Artifacts

The repository contains the complete source code and reproducibility materials for LungSimNet4D.

Large generated artifacts including:

- Generated anatomical variants
- Voxelized CT volumes
- Lesion masks
- Trained model weights
- Validation outputs
- GATE validation artifacts
- ROOT outputs
- Paper figures

are hosted separately on Figshare due to GitHub storage limitations.

Download:

https://figshare.com/articles/dataset/LungSimNet4D_Synthetic_Thoracic_Imaging_Dataset_GATE_Validation_Artifacts_and_Reproducibility_Package/32569494?file=65261310

After downloading:

```bash
unzip LungSimNet4D_artifacts.zip
```
Place the extracted folders at the project root,

LungSimNet4D/
├── generated/
├── models/
├── results/
├── paper_figures/
├── src/
└── README.md

---

## Repository Structure

```text
src/
├── phantom_generator/
├── lesion_injector/
├── voxelizer/
├── dataset_builder/
├── quality_control/
├── training/
│   ├── sgementation/
│   └── classification/
├── gate/

generated/
models/
results/
paper_figures/
```

---

## Environment Setup

Create a Python virtual environment:

```bash
python -m venv .lungsimnet4d_venv
source .lungsimnet4d_venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Generation Pipeline

## Step 1: Generate Anatomical Variants

```bash
python src/phantom_generator/create_mesh_variants.py
```

Creates anatomical phantom variations from the base thoracic phantom.

---

## Step 2: Build Dataset Manifest

```bash
python src/dataset_builder/generate_dataset_manifest.py
```

Creates dataset metadata and manifest files.

---

## Step 3: Place Lung Lesions

```bash
python src/lesion_injector/place_lesions.py
```

Places benign and malignant lesions inside lung volumes.

---

## Step 4: Update Manifest

```bash
python src/dataset_builder/update_manifest_after_lesions.py
```

Updates metadata after lesion placement.

---

## Step 5: Generate CT-Like Volumes

```bash
python src/voxelizer/voxelize_variants.py
```

Converts anatomical meshes into voxelized CT-like volumes and lesion masks.

---

## Step 6: Update Manifest

```bash
python src/dataset_builder/update_manifest_after_voxelization.py
```

Adds voxelization metadata to the dataset manifest.

---

## Step 7: Generate Visualization Figures

```bash
python src/dataset_builder/generate_figures.py
```

Creates axial, coronal, sagittal and lesion-overlay visualizations.

---

## Step 8: Generate Dataset Splits

```bash
python src/dataset_builder/split_train_val_test.py
```

Creates:

```text
train.csv
val.csv
test.csv
thorix_dataset_with_splits.csv
```

---

## Step 9: Generate Dataset QC

```bash
python src/quality_control/generate_dataset_qc_figures.py
```

Produces:

- Age distributions
- Gender distributions
- Lesion distributions
- PCA projections
- Correlation analysis
- Spatial lesion maps

---

# Segmentation Workflow

Train segmentation model:

```bash
python src/training/sgementation/train_unet3d.py
```

Run inference:

```bash
python src/training/sgementation/predict_unet3d_sample.py --variant variant_0001
```

Outputs:

```text
Dice Score
IoU
Accuracy
Prediction overlays
Saved model weights
```

---

# Classification Workflow

Train classification model:

```bash
python src/training/classification/train_classifier3d.py
```

Run inference:

```bash
python src/training/classification/predict_classifier3d_sample.py --variant variant_0001
```

Outputs:

```text
Accuracy
Precision
Recall
F1 Score
Predicted Class
Malignant Probability
Saved model weights
```

---

# Validation Summary

Generate validation reports:

```bash
python src/quality_control/build_validation_summary.py
```

Produces:

- Segmentation metrics
- Classification metrics
- Dataset statistics
- Validation summary artifacts

---

# GATE Physics Validation

## Export Voxel Phantom

```bash
python src/gate/export_gate_voxels.py
```

Creates:

```text
phantom.h33
phantom.i33
AttnRange.dat
```

---

## Launch GATE 9.0

```bash
docker run -i --rm \
  --platform linux/amd64 \
  -v "$PWD":/APP \
  -w /APP \
  opengatecollaboration/gate:9.0
```

Run the supplied GATE macro:

```text
main.mac
```

Outputs:

```text
OutputFile.root
```

---

# ROOT-Based Validation

Launch ROOT:

```bash
docker run -it --rm \
  --platform linux/amd64 \
  -v "$PWD":/APP \
  -w /APP \
  rootproject/root \
  root -l OutputFile.root
```

Detector Hit Count:

```cpp
Hits->GetEntries()
```

XY Projection:

```cpp
Hits->Draw("posY:posX","","colz")
c1->SaveAs("variant_0001_xy_projection.png")
```

XZ Projection:

```cpp
Hits->Draw("posZ:posX","","colz")
c1->SaveAs("variant_0001_xz_projection.png")
```

YZ Projection:

```cpp
Hits->Draw("posZ:posY","","colz")
c1->SaveAs("variant_0001_yz_projection.png")
```

Hit Sampling:

```cpp
Hits->Scan("posX:posY:posZ","","",20)
```

---

# Reproducibility

The repository includes:

- Phantom generation scripts
- Lesion generation scripts
- Dataset manifests
- Train / Validation / Test splits
- Segmentation training and inference scripts
- Classification training and inference scripts
- GATE validation workflows
- Random seed configuration
- Validation figures
- Quality control figures

---

# Disclaimer

LungSimNet4D is intended for research purposes only. The generated synthetic datasets and associated AI models are not intended for direct clinical use and require further validation using real-world clinical datasets before deployment in medical settings.

# License

This project is released under the MIT License. See the LICENSE file for details.

# Citation

If you use LungSimNet4D in academic work, please cite:

Poornima G., et al. "LungSimNet4D: An Anatomy-Informed Synthetic Thoracic Imaging Framework for Lung Cancer Research." (Under Review).

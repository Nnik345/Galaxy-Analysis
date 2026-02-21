# Galaxy Morphology Classification using Deep Learning and Gradient Boosting

A multi-model comparison for galaxy morphology classification using the Galaxy Zoo 2 dataset. This project compares 5 deep learning architectures, ensemble methods, and gradient boosting models for predicting galaxy morphological features.

---

## Overview

This project performs hierarchical multi-task learning on galaxy morphology classification. It predicts 30 morphological features across 10 hierarchical tasks (smooth vs. featured, edge-on, bar presence, spiral structure, bulge properties, etc.).

---

## Models Implemented

### Deep Learning Models
1. **ResNet-50**
2. **EfficientNet-V2-S**
3. **ConvNeXt-Base**
4. **Swin Transformer V2 Small**
5. **RegNetY-8GF**

### Ensemble Models
1. Equal Weighting
2. Accuracy-Weighted
3. Inverse MAE-Weighted

### Gradient Boosting Models
1. **XGBoost** (5 models, one per CNN backbone)
2. **LightGBM** (5 models, one per CNN backbone)

**Total: 18 models**

---

## Dataset

### Galaxy Zoo 2 Dataset

**Download:**
- Images: [PLACEHOLDER - Add download link]
- Labels (gz2_hart16.csv): [PLACEHOLDER - Add download link]
- Filename mapping (gz2_filename_mapping.csv): [PLACEHOLDER - Add download link]

**Statistics:**
- Total galaxies: ~156,000
- 10 hierarchical morphology tasks
- 30 total output classifications

---

## Requirements

- **Python:** 3.10+
- **GPU:** NVIDIA GPU recommended (code will run on CPU but much slower)
- **Storage:** ~100GB for dataset and models

---

## Installation

### 1. Create Environment

```bash
python3 -m venv galaxy-env
source galaxy-env/bin/activate  # Windows: galaxy-env\Scripts\activate
```

### 2. Install PyTorch

**With NVIDIA GPU (CUDA support):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Without GPU (CPU only):**
```bash
pip install torch torchvision torchaudio
```

### 3. Install Other Dependencies

```bash
pip install pandas numpy pillow tqdm matplotlib seaborn scikit-learn jupyter notebook ipykernel
```

### 4. Install XGBoost and LightGBM

The notebook includes installation cells (Cell 38), or install manually:
```bash
pip install xgboost lightgbm
```

---

## Data Setup

### Directory Structure

```
galaxy-analysis/
├── Galaxy_Analysis.ipynb
├── data/
│   ├── gz2_hart16.csv
│   ├── gz2_filename_mapping.csv
│   └── images_gz2/
│       └── images/
│           ├── 100008.jpg
│           ├── 100023.jpg
│           └── ...
└── results/                    # Created automatically
```

### Setup Steps

1. **Download dataset** (links above)

2. **Create directory and extract:**
   ```bash
   mkdir -p data/images_gz2/images
   unzip galaxy_images.zip -d data/images_gz2/images/
   mv gz2_hart16.csv data/
   mv gz2_filename_mapping.csv data/
   ```

3. **Verify:**
   ```bash
   ls data/
   # Should show: gz2_hart16.csv, gz2_filename_mapping.csv, images_gz2/
   ```

---

## Usage

### Start Notebook

```bash
jupyter notebook Galaxy_Analysis.ipynb
```

### Run Cells in Order

- **Cells 0-9:** Setup, imports, data loading
- **Cells 10-29:** Train individual models (ResNet, EfficientNet, ConvNeXt, Swin, RegNet)
- **Cells 30-32:** Model comparison
- **Cells 33-37:** Ensemble models
- **Cells 38-43:** XGBoost and LightGBM
- **Cell 44:** Final comparison

### Training Options

When prompted during training:
- **`r`** - Resume from checkpoint
- **`restart`** - Start fresh (delete checkpoint)
- **`s`** - Skip training (load saved model)

### Configuration

**Adjust data usage (Cell 3):**
```python
DATA_FRACTION = 0.3  # Use 30% of data
# DATA_FRACTION = 1.0  # Use full dataset
```

**Adjust model hyperparameters:**
Each model has a config cell (e.g., Cell 10 for ResNet-50):
```python
RESNET50_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.0003,
    'num_epochs': 250,
    'patience': 12
}
```

---

## Project Structure

### Notebook Organization

**Setup (Cells 0-9):**
- Environment, imports, configuration
- Data loading and preprocessing
- Model architecture definitions
- Training/evaluation functions

**Model Training (Cells 10-29):**
Each model follows 4-cell pattern:
1. Hyperparameters
2. Model creation
3. Training
4. Evaluation

**Comparisons (Cells 30-44):**
- Model comparison and visualization
- Ensemble learning
- Gradient boosting benchmarks

### Output Files

```
results/
├── resnet50/
│   ├── checkpoints/
│   │   ├── latest_checkpoint.pth
│   │   └── best_model.pth
│   ├── training_complete.txt
│   ├── training_curves.png
│   └── metrics.json
├── efficientnet_v2_s/
├── ... (other models)
└── model_comparison/
    ├── comparison_table.csv
    ├── dl_vs_gb_comparison.csv
    └── *.png (visualizations)
```

---

## Notes

### Memory Usage

If you encounter memory errors:
- Reduce batch size in model config
- Use smaller dataset fraction
- Train one model at a time

### Reproducibility

- Fixed random seeds (seed=42)
- Deterministic data splits
- All configurations saved in JSON

---

## Metrics

- **Accuracy:** Percentage of correct predictions (rounded to 0 or 1)
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error

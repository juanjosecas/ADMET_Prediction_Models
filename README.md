# ADMET Prediction Models

This repository provides scripts and pre-trained models for predicting common **ADMET** (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties using machine learning. The models are built with scikit-learn and optimized using Hyperopt for automated hyperparameter tuning.

## Overview

**What is ADMET?**
ADMET properties are critical pharmacokinetic and safety parameters evaluated during drug discovery. This repository helps predict these properties from molecular structures (SMILES strings), enabling:
- Early-stage compound screening
- Lead optimization guidance
- Reduced need for expensive in vitro/in vivo experiments

**Key Features:**
- 11 pre-trained models for different ADMET properties
- Automated hyperparameter optimization with Hyperopt
- Support for custom model training
- Scaffold-based and stratified data splitting
- Comprehensive evaluation metrics

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Workflow Notebook](#complete-workflow-notebook)
- [Available Models](#available-models)
- [Usage Guide](#usage-guide)
  - [Making Predictions](#making-predictions)
  - [Training Custom Models](#training-custom-models)
  - [Data Preparation](#data-preparation)
  - [Cross-Validation](#cross-validation)
- [Dataset Information](#dataset-information)
- [Classification Criteria](#classification-criteria)
- [Troubleshooting](#troubleshooting)

## Installation

### Requirements
- Python >= 3.9
- scikit-learn >= 1.26.4
- numpy == 1.25.0 *(Note: The previous version 11.5.0 appears to be a typo)*
- hpsklearn == 1.0.3
- hyperopt == 0.2.7
- xgboost == 2.0.3
- rdkit
- pandas
- dgllife

### Setup Instructions

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/CADD-SC/ADMET_Prediction_Models.git
cd ADMET_Prediction_Models

# Create and activate conda environment
conda env create -f env.yaml
conda activate admet
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/CADD-SC/ADMET_Prediction_Models.git
cd ADMET_Prediction_Models

# Install dependencies
pip install -r requirements.txt
```

### Downloading Pre-trained Models

The repository uses Git LFS (Large File Storage) for model files.

**Windows:**
```bash
git lfs install
git lfs pull
```

**Linux:**
```bash
sudo apt-get install git-lfs
git lfs install
git lfs pull
```

**Alternative: Manual Download**
If Git LFS fails, you can download the models manually using `gdown`:
```bash
pip install gdown

# Download pre-trained models
gdown --folder 1AYW-4HXgnU89_BQU-_-rWV_apps-Gp9U

# Download scaler file
gdown --folder 1mbBZt7pEfGu7iqt7WCq5wYqiwoAxpKpB
```

The models should be in the `models/` directory and the scaler in `scaler/new_scaler.pkl`.

## Complete Workflow Notebook

**NEW**: For a comprehensive, interactive walkthrough of all ADMET prediction features, check out the Jupyter notebook:

📓 **[ADMET_Complete_Workflow.ipynb](ADMET_Complete_Workflow.ipynb)**

This notebook provides a complete end-to-end workflow demonstrating:
- ✓ Data loading and preprocessing with molecular feature calculation
- ✓ Making predictions using all 11 pre-trained ADMET models
- ✓ Comprehensive visualization of results (distribution charts, heatmaps, radar charts)
- ✓ Individual molecule ADMET profile analysis
- ✓ Data splitting strategies (stratified and scaffold-based)
- ✓ Custom model training workflow
- ✓ Exporting results in multiple formats

**To use the notebook:**
```bash
# Launch Jupyter
jupyter notebook ADMET_Complete_Workflow.ipynb

# Or use JupyterLab
jupyter lab ADMET_Complete_Workflow.ipynb
```

The notebook is designed to be self-contained with detailed explanations and can serve as both a tutorial and a template for your own ADMET prediction workflows.

## Quick Start

### Making a Simple Prediction

1. Create a CSV file with your molecules (e.g., `my_compounds.csv`):
```csv
SMILES
CCO
c1ccccc1
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

2. Run prediction:
```bash
python predict.py --file_name my_compounds.csv --model_name BBB
```

3. View results in `BBB_predict_results.csv`:
```csv
SMILES,BBB,pred_prob
CCO,1,0.85
c1ccccc1,1,0.92
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0,0.78
```

## Available Models

The repository includes 11 pre-trained models for various ADMET properties:

| Model Name | Property | Description |
|------------|----------|-------------|
| `BBB` | Blood-Brain Barrier | Predicts BBB permeability (logBB) |
| `Papp` | Caco-2 Permeability | Predicts intestinal absorption |
| `P_gp_subs` | P-glycoprotein Substrate | Predicts efflux transporter substrate |
| `CYP1A2` | CYP1A2 Inhibition | Predicts inhibition of CYP1A2 enzyme |
| `CYP2C9` | CYP2C9 Inhibition | Predicts inhibition of CYP2C9 enzyme |
| `CYP2C19` | CYP2C19 Inhibition | Predicts inhibition of CYP2C19 enzyme |
| `CYP2D6` | CYP2D6 Inhibition | Predicts inhibition of CYP2D6 enzyme |
| `CYP3A4` | CYP3A4 Inhibition | Predicts inhibition of CYP3A4 enzyme |
| `HCLint` | Human Hepatic Clearance | Predicts metabolic stability in human liver |
| `RCLint` | Rat Hepatic Clearance | Predicts metabolic stability in rat liver |
| `hERG_inh` | hERG Inhibition | Predicts cardiotoxicity risk |

## Usage Guide

### Making Predictions

Use pre-trained models to predict ADMET properties for your compounds:

```bash
python predict.py --file_name INPUT.csv --model_name MODEL_NAME
```

**Parameters:**
- `--file_name`: Path to CSV file containing a `SMILES` column
- `--model_name`: Name of the model (see [Available Models](#available-models))

**Example:**
```bash
# Predict BBB permeability
python predict.py --file_name compounds.csv --model_name BBB

# Predict CYP3A4 inhibition
python predict.py --file_name compounds.csv --model_name CYP3A4

# Predict hERG cardiotoxicity
python predict.py --file_name compounds.csv --model_name hERG_inh
```

**Output:**
The script creates a file named `{MODEL_NAME}_predict_results.csv` with:
- Original SMILES strings
- Binary predictions (0 or 1)
- Prediction probability scores

### Training Custom Models

Train your own ADMET models with custom datasets.

#### Step 1: Prepare Your Data

Your CSV file must contain:
- `SMILES` column: Molecular structures as SMILES strings
- `bioclass` column: Binary labels (0 or 1)

Example `my_data.csv`:
```csv
SMILES,bioclass
CCO,1
c1ccccc1,0
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,1
```

#### Step 2: Split the Dataset

```bash
python data_split.py --file_name my_data.csv \
                     --split_mode stratify \
                     --test_frac 0.2
```

**Parameters:**
- `--file_name`: Input CSV file with SMILES and bioclass
- `--split_mode`: Splitting strategy
  - `stratify`: Random split maintaining class distribution (default)
  - `scaffold`: Split by molecular scaffolds (recommended for drug discovery)
- `--test_frac`: Fraction for test set (default: 0.2 = 20%)

**Output:**
- `train_my_data.csv`: Training set
- `test_my_data.csv`: Test set

**Why Scaffold Split?**
Scaffold-based splitting groups molecules by their core structures (Bemis-Murcko scaffolds), preventing data leakage and providing more realistic performance estimates for novel compounds.

#### Step 3: Train the Model

```bash
python model.py --file_name my_data.csv \
                --model_name my_custom_model \
                --max_eval 200 \
                --time_out 120 \
                --training
```

**Parameters:**
- `--file_name`: Name of the CSV file (must have corresponding train/test splits)
- `--model_name`: Name for your trained model (default: derived from file name)
- `--max_eval`: Number of hyperparameter optimization trials (default: 200)
- `--time_out`: Maximum seconds per trial (default: 120)
- `--training`: Flag to enable training mode

**Output:**
- `{MODEL_NAME}.pkl`: Trained model file
- Performance metrics printed to console:
  - Sensitivity (Recall)
  - Specificity
  - Accuracy
  - Matthews Correlation Coefficient (MCC)
  - Area Under ROC Curve (AUC)

**Training Time:**
Training duration depends on:
- Dataset size
- Number of trials (`max_eval`)
- Timeout per trial (`time_out`)

Typical training takes 1-6 hours with default settings.

### Cross-Validation

Evaluate model performance with k-fold cross-validation:

```bash
python model.py --file_name my_data.csv \
                --cross_validation \
                --split stratify \
                --k_fold 5 \
                --max_eval 100 \
                --time_out 60
```

**Parameters:**
- `--cross_validation`: Enable cross-validation mode
- `--split`: Splitting method (`stratify` or `scaffold`)
- `--k_fold`: Number of folds (default: 5)
- `--max_eval`: Optimization trials per fold
- `--time_out`: Timeout per trial in seconds

**Output:**
Performance metrics for each fold plus mean ± standard deviation across all folds.

**When to Use:**
- Evaluating model robustness
- Comparing different algorithms or features
- Working with limited data

## Dataset Information

All datasets used in the associated manuscript are available in the `data/` folder. Each subdirectory contains train, test, and validation sets.

**Dataset Structure:**
```
data/
├── BBB/
│   ├── train_BBB.csv
│   ├── test_BBB.csv
│   └── val_BBB.csv
├── CYP1A2/
│   ├── train_CYP1A2.csv
│   └── ...
└── ...
```

Each dataset includes:
- `SMILES`: Molecular structure
- `bioclass`: Binary label (0 or 1)
- Pre-calculated molecular features (descriptors and fingerprints)

## Classification Criteria

Models classify compounds based on the following thresholds. Class 1 indicates the property is present (e.g., permeable, inhibitor, substrate).

| Property | Abbreviation | Criteria (Class 1) | Biological Meaning |
|----------|--------------|--------------------|--------------------|
| **Absorption** | | | |
| Caco-2 permeability | `Papp` | Papp ≥ 8×10⁻⁶ cm/s | Good intestinal absorption |
| P-gp substrate | `P_gp_subs` | ER ≥ 2 | Subject to efflux (lower absorption) |
| BBB permeability | `BBB` | logBB ≥ -1 | Penetrates blood-brain barrier |
| **Metabolism** | | | |
| CYP1A2 inhibition | `CYP1A2` | IC50/AC50 < 10 µM | Inhibits CYP1A2 enzyme |
| CYP2C9 inhibition | `CYP2C9` | IC50/AC50 < 10 µM | Inhibits CYP2C9 enzyme |
| CYP2C19 inhibition | `CYP2C19` | IC50/AC50 < 10 µM | Inhibits CYP2C19 enzyme |
| CYP2D6 inhibition | `CYP2D6` | IC50/AC50 < 10 µM | Inhibits CYP2D6 enzyme |
| CYP3A4 inhibition | `CYP3A4` | IC50/AC50 < 10 µM | Inhibits CYP3A4 enzyme |
| Human hepatic stability | `HCLint` | t½ > 30 min | Metabolically stable in human liver |
| Rat hepatic stability | `RCLint` | t½ > 30 min | Metabolically stable in rat liver |
| **Toxicity** | | | |
| hERG inhibition | `hERG_inh` | IC50 < 10 µM | Risk of cardiotoxicity |

**Understanding the Predictions:**
- **Class 1**: Property is present (e.g., permeable, inhibitor, stable)
- **Class 0**: Property is absent (e.g., not permeable, not inhibitor, unstable)
- **pred_prob**: Confidence score (0-1), higher values indicate more confidence

## Troubleshooting

### Common Issues

**1. "Model not found" error**
```
FileNotFoundError: Model 'BBB.pkl' not found
```
**Solution:** Ensure you've downloaded the pre-trained models using Git LFS or `gdown` (see [Installation](#installation))

**2. "features.txt not found" error**
```
FileNotFoundError: [Errno 2] No such file or directory: 'features.txt'
```
**Solution:** Make sure you're running scripts from the repository root directory

**3. "scaler/new_scaler.pkl not found" error**
```
FileNotFoundError: [Errno 2] No such file or directory: 'scaler/new_scaler.pkl'
```
**Solution:** Download the scaler file using the `gdown` command in [Installation](#installation)

**4. RDKit import errors**
```
ModuleNotFoundError: No module named 'rdkit'
```
**Solution:** Install RDKit via conda:
```bash
conda install -c conda-forge rdkit
```

**5. Invalid SMILES warnings**
```
Warning: Could not canonicalize SMILES: XYZ
```
**Solution:** This is normal. Invalid SMILES are automatically filtered out during preprocessing.

**6. Memory errors during training**
**Solution:** Reduce the number of parallel processes in `utils.py` (line with `NUM_PROCESSES = 24`)

### Getting Help

If you encounter issues:
1. Check that all dependencies are correctly installed
2. Verify your input CSV has the correct format
3. Ensure model files are in the `models/` directory
4. Open an issue on GitHub with:
   - Error message
   - Python version
   - Operating system
   - Command you ran

## Citation

If you use these models in your research, please cite:

```bibtex
@article{your_citation_here,
  title={ADMET Prediction Models},
  author={Your Authors},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub.

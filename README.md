# ADMET Prediction Models

This repository provides scripts and pretrained models for predicting common ADMET properties using scikit-learn and Hyperopt. Example datasets are located in the `data/` directory and pretrained models can be found in `models/`.

## Requirements
- Python >= 3.9
- scikit-learn >= 1.26.4
- numpy == 11.5.0
- hpsklearn == 1.0.3
- hyperopt == 0.2.7
- xgboost == 2.0.3
- rdkit
- pandas
- dgllife

## Repository setup

### Windows
```bash
git lfs install
git clone https://github.com/CADD-SC/ADMET_Prediction_Models.git
cd ADMET_Prediction_Models
git lfs pull
```

### Linux
```bash
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/CADD-SC/ADMET_Prediction_Models.git
cd ADMET_Prediction_Models
git lfs pull
```

If LFS downloads fail you can fetch the model `.pkl` files with:
```bash
pip install gdown
gdown --folder 1AYW-4HXgnU89_BQU-_-rWV_apps-Gp9U
```
The scaler file (`scaler/new_scaler.pkl`) is available at:
```bash
gdown --folder 1mbBZt7pEfGu7iqt7WCq5wYqiwoAxpKpB
```

## Dataset
All datasets used in the manuscript are stored in the `data/` folder. Each subdirectory contains train, test and validation sets.

### Classification criteria
The models classify a compound based on its predicted value. For example, for BBB permeability a logBB >= -1 is considered permeable (class 1).

| Property            | Criteria (class 1)      |
|---------------------|-------------------------|
| Caco-2 permeability | Papp >= 8×10-6 cm/s     |
| P-gp substrate      | ER >= 2                 |
| BBB permeability    | logBB >= -1             |
| CYP1A2 inhibition   | IC50 or AC50 < 10 µM    |
| CYP2C9 inhibition   | IC50 or AC50 < 10 µM    |
| CYP2C19 inhibition  | IC50 or AC50 < 10 µM    |
| CYP2D6 inhibition   | IC50 or AC50 < 10 µM    |
| CYP3A4 inhibition   | IC50 or AC50 < 10 µM    |
| HLM stability       | t1/2 > 30 min           |
| RLM stability       | t1/2 > 30 min           |
| hERG inhibition     | IC50 < 10 µM            |

## Model training

### Data preparation
Use your SMILES data or the provided `smiles.csv` example containing `SMILES` and `bioclass` columns.
```bash
python data_split.py --file_name smiles.csv \
                     --split_mode stratify \
                     --test_frac 0.2
```
This creates `train_smiles.csv` and `test_smiles.csv` in the current directory.

### Train the model
```bash
python model.py --file_name smiles.csv \
                --model_name test \
                --max_eval 200 \
                --time_out 120 \
                --training
```

### Prediction
To predict with a pretrained model, provide a file with a `SMILES` column, for example `tmp.csv`.
```bash
python predict.py --file_name tmp.csv --model_name BBB
```
The results will be written to `BBB_predict_results.csv`.

Available model names are `BBB`, `CYP1A2`, `CYP2C19`, `CYP2C9`, `CYP2D6`, `CYP3A4`, `HCLint`, `P_gp_subs`, `Papp`, `RCLint` and `hERG_inh`.

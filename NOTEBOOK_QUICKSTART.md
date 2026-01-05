# Quick Start Guide - ADMET Complete Workflow Notebook

This guide will help you get started with the ADMET Complete Workflow notebook in just a few minutes.

## Prerequisites
- Python 3.9 or higher
- Git (for cloning the repository)
- Pre-trained ADMET model files (included in the repository via Git LFS)

## Step 1: Clone and Setup (5 minutes)

```bash
# Clone the repository
git clone https://github.com/juanjosecas/ADMET_Prediction_Models.git
cd ADMET_Prediction_Models

# Option A: Install with conda (recommended)
conda env create -f env.yaml
conda activate admet

# Option B: Install with pip
pip install -r requirements.txt
```

## Step 2: Download Model Files (if needed)

If Git LFS didn't automatically download the model files:

```bash
# Check if models are present
ls -lh models/

# If empty, download manually
pip install gdown
gdown --folder 1AYW-4HXgnU89_BQU-_-rWV_apps-Gp9U
gdown --folder 1mbBZt7pEfGu7iqt7WCq5wYqiwoAxpKpB
```

## Step 3: Launch the Notebook (1 minute)

```bash
# Start Jupyter Notebook
jupyter notebook ADMET_Complete_Workflow.ipynb

# OR start JupyterLab
jupyter lab ADMET_Complete_Workflow.ipynb
```

Your browser will open automatically with the notebook.

## Step 4: Run the Workflow (5-10 minutes)

In the Jupyter interface:

1. **Run all cells**: Click `Cell` → `Run All` in the menu
2. **Or run step-by-step**: Press `Shift + Enter` on each cell

The notebook will:
- ✓ Load the sample data (smiles.csv with 300+ molecules)
- ✓ Preprocess and calculate molecular features
- ✓ Make predictions using all 11 ADMET models
- ✓ Create visualizations (charts, heatmaps, plots)
- ✓ Generate detailed analysis
- ✓ Export results to CSV files

## What You'll Get

After running the notebook, you'll have:

### Output Files:
- `all_admet_predictions.csv` - All predictions with confidence scores
- `all_admet_predictions_detailed.csv` - Detailed predictions
- `all_admet_predictions_binary.csv` - Simple binary predictions
- `admet_summary_statistics.csv` - Summary statistics

### Visualizations:
- Class distribution charts
- Prediction distribution across all models
- Heatmap of prediction patterns
- Confidence score box plots
- Individual molecule ADMET profiles (radar charts)
- Molecular structure images

## Understanding the Results

### ADMET Properties Predicted:

| Code | Property | Positive (1) Means |
|------|----------|-------------------|
| BBB | Blood-Brain Barrier | Crosses BBB (logBB ≥ -1) |
| Papp | Caco-2 Permeability | Good absorption (≥ 8×10⁻⁶ cm/s) |
| P_gp_subs | P-gp Substrate | Is a substrate (ER ≥ 2) |
| CYP1A2 | CYP1A2 Inhibition | Inhibits enzyme (IC50 < 10 µM) |
| CYP2C9 | CYP2C9 Inhibition | Inhibits enzyme (IC50 < 10 µM) |
| CYP2C19 | CYP2C19 Inhibition | Inhibits enzyme (IC50 < 10 µM) |
| CYP2D6 | CYP2D6 Inhibition | Inhibits enzyme (IC50 < 10 µM) |
| CYP3A4 | CYP3A4 Inhibition | Inhibits enzyme (IC50 < 10 µM) |
| HCLint | Human Hepatic Clearance | Metabolically stable (t½ > 30 min) |
| RCLint | Rat Hepatic Clearance | Metabolically stable (t½ > 30 min) |
| hERG_inh | hERG Inhibition | Risk of cardiotoxicity (IC50 < 10 µM) |

### Interpreting Predictions:
- **0 (Negative)**: Property absent (e.g., doesn't cross BBB, not an inhibitor)
- **1 (Positive)**: Property present (e.g., crosses BBB, is an inhibitor)
- **pred_prob**: Confidence score (0.0 to 1.0, higher = more confident)

## Customizing the Workflow

### Use Your Own Data:

1. Create a CSV file with your molecules:
   ```csv
   SMILES,bioclass
   CCO,1
   c1ccccc1,0
   ```

2. Modify the notebook:
   ```python
   # Change this line in Section 2
   data_file = 'your_file.csv'  # Instead of 'smiles.csv'
   ```

3. Run the cells!

### Focus on Specific Models:

```python
# Instead of all models, use only what you need
admet_models = ['BBB', 'hERG_inh', 'CYP3A4']  # Only these three
```

### Adjust Visualizations:

```python
# Show more or fewer molecules in visualizations
# Change this value in Section 5
molecule_idx = 5  # Examine a different molecule
```

## Troubleshooting

### Issue: "Model not found"
**Solution**: Ensure model files are in the `models/` directory
```bash
ls models/  # Should show BBB.pkl, CYP1A2.pkl, etc.
```

### Issue: "features.txt not found"
**Solution**: Make sure you're running from the repository root
```bash
pwd  # Should end in ADMET_Prediction_Models
ls features.txt  # Should exist
```

### Issue: "Module not found"
**Solution**: Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue: Notebook runs slowly
**Solution**: The first run calculates features and can take 5-10 minutes. Subsequent runs are faster. You can also:
- Reduce the dataset size for testing
- Adjust NUM_PROCESSES in utils.py (default is 24)

## Next Steps

Once you've run the basic workflow:

1. **Analyze Results**: Review the generated CSV files and visualizations
2. **Customize**: Modify the notebook for your specific needs
3. **Train Models**: Use Section 6 & 7 to train custom models on your data
4. **Integrate**: Incorporate the workflow into your drug discovery pipeline
5. **Scale Up**: Process larger datasets by batch

## Getting Help

- **Documentation**: See [README.md](README.md) for detailed information
- **Code**: Check the source files (predict.py, model.py, utils.py)
- **Issues**: Open an issue on GitHub with error messages and steps to reproduce

## Tips for Success

✓ **Start Small**: Test with a few molecules first before processing large datasets  
✓ **Check Models**: Verify all 11 .pkl files are in the models/ directory  
✓ **Monitor Progress**: Watch the output as cells execute to catch errors early  
✓ **Save Often**: The notebook auto-exports results, but save your work frequently  
✓ **Experiment**: The notebook is designed to be modified and customized  

---

**Total Time to First Results: ~15 minutes**

Happy predicting! 🧪🔬

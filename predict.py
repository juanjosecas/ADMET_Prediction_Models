"""
ADMET Property Prediction Module.

This module provides functionality to predict ADMET (Absorption, Distribution,
Metabolism, Excretion, and Toxicity) properties for molecules using pre-trained
machine learning models.

Available pre-trained models:
- BBB: Blood-Brain Barrier permeability
- CYP1A2, CYP2C19, CYP2C9, CYP2D6, CYP3A4: Cytochrome P450 inhibition
- HCLint: Human hepatic clearance
- P_gp_subs: P-glycoprotein substrate
- Papp: Caco-2 cell permeability
- RCLint: Rat hepatic clearance
- hERG_inh: hERG channel inhibition
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from utils import data_preprocessing, load_features

def load_model(name: str):
    """
    Load a pre-trained model from disk.
    
    Searches for the model in two locations:
    1. models/{name}.pkl (default location for pre-trained models)
    2. {name}.pkl (current directory for user-trained models)
    
    Args:
        name: Name of the model without .pkl extension
              (e.g., 'BBB', 'CYP1A2', 'hERG_inh')
        
    Returns:
        Loaded scikit-learn model object
        
    Raises:
        FileNotFoundError: If the model file is not found in either location
    """
    # Check default locations for model files
    model_paths = [f"models/{name}.pkl", f"{name}.pkl"]
    
    for path in model_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    
    # Model not found in any location
    raise FileNotFoundError(
        f"Model '{name}.pkl' not found. Searched in: {', '.join(model_paths)}"
    )


def predict(file_name: str, model_name: str) -> None:
    """
    Predict ADMET properties for molecules in a CSV file.
    
    This function:
    1. Loads molecular features and the specified model
    2. Reads and preprocesses the input SMILES data
    3. Generates predictions and probability scores
    4. Saves results to a CSV file named '{model_name}_predict_results.csv'
    
    Args:
        file_name: Path to CSV file containing a 'SMILES' column
        model_name: Name of the pre-trained model to use (without .pkl extension)
        
    Output:
        Creates a CSV file with columns:
        - SMILES: Input molecular structure
        - {model_name}: Binary prediction (0 or 1)
        - pred_prob: Maximum probability score
        
    Example:
        >>> predict('compounds.csv', 'BBB')
        # Creates 'BBB_predict_results.csv' with predictions
    """
    # Load feature names used during model training
    features = load_features()
    print(f"Prediction model for {model_name}")
    print(f"Loading the data set: {file_name}")

    # Read input data and add placeholder bioclass column for preprocessing
    df = pd.read_csv(file_name)
    df["bioclass"] = 1  # Dummy value required for preprocessing pipeline
    
    # Preprocess data: validate SMILES, calculate features, and scale
    scaled_data = data_preprocessing(df)

    # Load the pre-trained model
    model = load_model(model_name)

    # Generate predictions
    predictions = model.predict(scaled_data[features].values)
    prediction_probabilities = model.predict_proba(scaled_data[features].values)
    max_probabilities = np.max(prediction_probabilities, axis=1)

    # Create results DataFrame
    result_df = pd.DataFrame({
        "SMILES": scaled_data["SMILES"],
        model_name: predictions,
        "pred_prob": max_probabilities
    })

    # Display and save results
    print(f"\nPrediction results for {model_name}:")
    print(result_df)
    
    output_file = f"{model_name}_predict_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict ADMET properties using pre-trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available pre-trained models:
  BBB          - Blood-Brain Barrier permeability (logBB >= -1: permeable)
  CYP1A2       - CYP1A2 inhibition (IC50/AC50 < 10 µM: inhibitor)
  CYP2C19      - CYP2C19 inhibition (IC50/AC50 < 10 µM: inhibitor)
  CYP2C9       - CYP2C9 inhibition (IC50/AC50 < 10 µM: inhibitor)
  CYP2D6       - CYP2D6 inhibition (IC50/AC50 < 10 µM: inhibitor)
  CYP3A4       - CYP3A4 inhibition (IC50/AC50 < 10 µM: inhibitor)
  HCLint       - Human hepatic clearance (t1/2 > 30 min: stable)
  P_gp_subs    - P-glycoprotein substrate (ER >= 2: substrate)
  Papp         - Caco-2 permeability (Papp >= 8×10⁻⁶ cm/s: permeable)
  RCLint       - Rat hepatic clearance (t1/2 > 30 min: stable)
  hERG_inh     - hERG inhibition (IC50 < 10 µM: inhibitor)

Example:
  python predict.py --file_name compounds.csv --model_name BBB
        """
    )
    parser.add_argument(
        "--file_name",
        default="example_train.csv",
        help="Path to input CSV file with 'SMILES' column"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the pre-trained model (without .pkl extension)"
    )
    args = parser.parse_args()

    predict(args.file_name, args.model_name)

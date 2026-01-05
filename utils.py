"""
Utility functions for molecular feature calculation and data preprocessing.

This module provides functions for:
- Loading feature names
- Data preprocessing and validation
- SMILES canonicalization
- Molecular descriptor and fingerprint calculation
- Data standardization using pre-trained scaler
"""
import argparse
import joblib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

pd.set_option('display.max_columns', None)

def load_features(path: str = 'features.txt') -> list:
    """
    Load feature names from a text file.
    
    Args:
        path: Path to the features text file (default: 'features.txt')
        
    Returns:
        List of feature names used for model training and prediction
        
    Raises:
        FileNotFoundError: If the feature file doesn't exist
    """
    with open(path, 'r') as f:
        return eval(f.read())

def data_preprocessing(tmp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess molecular data: validate, canonicalize, featurize, and scale.
    
    This function performs the following steps:
    1. Convert SMILES to RDKit molecule objects and filter invalid structures
    2. Canonicalize SMILES strings for consistency
    3. Remove duplicate molecules
    4. Calculate molecular descriptors and fingerprints
    5. Apply standard scaling using pre-trained scaler
    
    Args:
        tmp_df: DataFrame with 'SMILES' and 'bioclass' columns
        
    Returns:
        Preprocessed DataFrame with calculated features, SMILES, and bioclass
        
    Raises:
        KeyError: If required columns ('SMILES', 'bioclass') are missing
    """
    original_count = len(tmp_df)
    
    # Convert SMILES to molecule objects and filter invalid structures
    tmp_df['mol'] = [Chem.MolFromSmiles(s) for s in tmp_df['SMILES']]
    data = tmp_df.dropna(subset=['mol'])
    invalid_count = original_count - len(data)
    print(f'Removed {invalid_count} invalid molecule objects')
    
    # Canonicalize SMILES for consistency
    data.loc[:, 'SMILES'] = canonical_smiles(data['SMILES'])
    
    # Remove duplicate molecules
    data = data.drop_duplicates(subset=['SMILES'], keep='first')
    data = data[['SMILES', 'bioclass']]
    
    # Calculate all molecular features (descriptors + fingerprints)
    all_data = calculate_allfeatures(data)
    
    # Apply standard scaling
    all_data = standard_scaled(all_data)
    
    return all_data

def canonical_smiles(smiles_list) -> list:
    """
    Convert SMILES strings to canonical form.
    
    For SMILES with multiple fragments (separated by '.'), only the longest
    fragment is retained and canonicalized. Invalid SMILES are skipped.
    
    Args:
        smiles_list: List or Series of SMILES strings
        
    Returns:
        List of canonical SMILES strings
        
    Note:
        - Multi-fragment SMILES (e.g., salts) are reduced to the largest fragment
        - Invalid SMILES that cannot be canonicalized are skipped
    """
    clean_smiles = []
    for smiles in smiles_list:
        try:
            # Split multi-fragment SMILES and keep the longest fragment
            fragments = str(smiles).split('.')
            longest_fragment = max(fragments, key=len)
            canonical_smi = Chem.CanonSmiles(longest_fragment)
            clean_smiles.append(canonical_smi)
        except Exception as e:
            print(f'Warning: Could not canonicalize SMILES: {smiles} (Error: {e})')
            continue
    return clean_smiles

def calculate_allfeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all molecular features including descriptors and fingerprints.
    
    Uses parallel processing to compute:
    - RDKit molecular descriptors (200+ descriptors)
    - ECFP6 fingerprints (1024-bit Morgan fingerprints with radius 3)
    
    Args:
        df: DataFrame with 'SMILES' and 'bioclass' columns
        
    Returns:
        DataFrame with calculated descriptors, ECFP6 fingerprints, SMILES, and bioclass
        
    Note:
        - Uses multiprocessing with 24 processes for parallel computation
        - ECFP6 (Extended-Connectivity Fingerprint) is a circular fingerprint
          equivalent to Morgan fingerprint with radius 3
    """
    from multiprocessing import Pool
    
    # Number of parallel processes for feature calculation
    NUM_PROCESSES = 24
    
    # Prepare data for parallel processing
    molecule_data = [(smi, bioclass) for smi, bioclass in zip(df['SMILES'], df['bioclass'])]
    
    # Calculate features in parallel
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(calculate_ecfp, molecule_data)
    
    # Extract results from parallel computation
    ecfp6_list = [result[0] for result in results]
    descriptors_list = [result[1] for result in results]
    bioclass_list = [result[2] for result in results]
    smiles_list = [result[3] for result in results]
    
    # Create DataFrames from calculated features
    descriptors_df = pd.DataFrame(descriptors_list)
    metadata_df = pd.DataFrame({'SMILES': smiles_list, 'bioclass': bioclass_list})
    ecfp6_df = pd.DataFrame(
        ecfp6_list,
        columns=[f'ECFP6_{i+1}' for i in range(ecfp6_list[0].size)]
    )
    
    # Combine all features
    features_df = pd.concat([descriptors_df, ecfp6_df, metadata_df], axis=1)
    return features_df

def calculate_ecfp(tup_input: tuple) -> tuple:
    """
    Calculate molecular descriptors and ECFP6 fingerprint for a single molecule.
    
    Args:
        tup_input: Tuple of (smiles, bioclass)
        
    Returns:
        Tuple of (ecfp_array, descriptors_dict, bioclass, smiles)
        - ecfp_array: numpy array of 1024-bit fingerprint
        - descriptors_dict: dictionary of RDKit descriptor names and values
        - bioclass: original bioactivity class
        - smiles: original SMILES string
        
    Note:
        - ECFP6 is Morgan fingerprint with radius 3 (diameter 6)
        - Calculates all available RDKit descriptors (~200 properties)
    """
    # Morgan fingerprint parameters
    MORGAN_RADIUS = 3  # Radius 3 = ECFP6 (diameter 6)
    FINGERPRINT_BITS = 1024  # Standard fingerprint size
    
    smiles, bioclass = tup_input[0], tup_input[1]
    mol = Chem.MolFromSmiles(smiles)
    
    # Calculate all RDKit molecular descriptors
    descriptors = {}
    for descriptor_name, descriptor_function in Descriptors.descList:
        descriptors[descriptor_name] = descriptor_function(mol)
    
    # Calculate ECFP6 (Morgan fingerprint with radius 3)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=FINGERPRINT_BITS)
    
    return np.array(ecfp), descriptors, bioclass, smiles

def standard_scaled(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard scaling to molecular features using pre-trained scaler.
    
    This function:
    1. Handles missing values (fills categorical with 'empty', numeric with 0)
    2. Removes non-numeric and problematic descriptor columns
    3. Applies pre-trained StandardScaler transformation
    4. Rounds scaled values to 10 decimal places
    5. Recombines with SMILES and bioclass metadata
    
    Args:
        all_data: DataFrame with calculated molecular features
        
    Returns:
        DataFrame with scaled features, SMILES, and bioclass
        
    Note:
        - Uses pre-trained scaler from 'scaler/new_scaler.pkl'
        - Excludes specific descriptors that are problematic or not used in models
    """
    # Columns to exclude from scaling (non-numeric metadata)
    METADATA_COLUMNS = ['SMILES', 'bioclass']
    
    # Descriptors excluded due to data quality or model requirements
    EXCLUDED_DESCRIPTORS = [
        'SPS', 'AvgIpc', 'NumAmideBonds', 'NumAtomStereoCenters',
        'NumBridgeheadAtoms', 'NumHeterocycles', 'NumSpiroAtoms',
        'NumUnspecifiedAtomStereoCenters', 'Phi'
    ]
    
    # Handle missing values
    categorical_columns = all_data.select_dtypes(exclude=['int64', 'float64']).columns
    all_data[categorical_columns] = all_data[categorical_columns].fillna('empty')
    all_data.fillna(0, inplace=True)
    
    # Remove metadata and excluded descriptors
    columns_to_drop = METADATA_COLUMNS + EXCLUDED_DESCRIPTORS
    features_df = all_data.drop(columns=columns_to_drop)
    
    # Load pre-trained scaler and transform features
    scaler = joblib.load('scaler/new_scaler.pkl')
    scaled_features = scaler.transform(features_df)
    
    # Convert back to DataFrame and round values
    scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
    scaled_df = scaled_df.map(lambda x: round(x, 10))
    
    # Recombine with metadata
    return pd.concat([scaled_df, all_data[METADATA_COLUMNS]], axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="test_example.csv")
    args = parser.parse_args()

    features = load_features()
    print(f"Loaded {len(features)} features")

    df = pd.read_csv(args.file_name)
    scaled = data_preprocessing(df)
    print(scaled)

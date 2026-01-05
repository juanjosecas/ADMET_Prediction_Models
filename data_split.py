"""
Dataset Splitting Module for ADMET Model Training.

This module provides functionality to split molecular datasets into training
and test sets using different splitting strategies:
- Stratified split: Random split maintaining class distribution
- Scaffold split: Split based on molecular scaffolds to reduce data leakage
"""
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

from utils import data_preprocessing

def split_dataset(df: pd.DataFrame, mode: str, frac: float) -> tuple:
    """
    Split dataset into training and test sets using the specified method.
    
    Two splitting strategies are available:
    1. Stratified: Random split maintaining class distribution (good for general use)
    2. Scaffold: Split based on molecular scaffolds (prevents data leakage)
    
    Args:
        df: DataFrame with preprocessed molecular data
        mode: Splitting method ('stratify' or 'scaffold')
        frac: Fraction of data to use for testing (e.g., 0.2 for 20%)
        
    Returns:
        Tuple of (train_data, test_data) DataFrames
        
    Note:
        - Stratified split uses random_state=42 for reproducibility
        - Scaffold split groups molecules by Bemis-Murcko scaffolds
        - Scaffold split is recommended to avoid overestimation of model performance
    """
    if mode == "stratify":
        # Random split maintaining class distribution
        train_data, test_data = train_test_split(
            df,
            random_state=42,  # For reproducibility
            test_size=frac,
            stratify=df["bioclass"],
            shuffle=True,
        )
    else:  # scaffold split
        from dgllife.utils import ScaffoldSplitter

        # Rename column for scaffold splitter (requires 'smiles' column name)
        df = df.rename(columns={"SMILES": "smiles"})
        
        # Split dataset by molecular scaffolds
        train_set, _, test_set = ScaffoldSplitter.train_val_test_split(
            df,
            frac_train=1 - frac,
            frac_val=0,  # No validation set in this split
            frac_test=frac
        )
        
        train_data = df.iloc[train_set.indices]
        test_data = df.iloc[test_set.indices]

    print(f"Size of train dataset: {len(train_data)}")
    print(f"Size of test  dataset: {len(test_data)}")
    
    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split molecular dataset into training and test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Splitting strategies:
  stratify     - Random split maintaining class distribution (default)
                 Best for: General machine learning tasks
  scaffold     - Split based on molecular scaffolds (Bemis-Murcko)
                 Best for: Avoiding data leakage in drug discovery

Example:
  python data_split.py --file_name smiles.csv --split_mode stratify --test_frac 0.2
        """
    )
    parser.add_argument(
        "--file_name",
        default="smiles.csv",
        help="Path to input CSV file with 'SMILES' and 'bioclass' columns"
    )
    parser.add_argument(
        "--split_mode",
        choices=["scaffold", "stratify"],
        default="stratify",
        help="Dataset splitting method (default: stratify)"
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Test set fraction, e.g., 0.2 for 20%% (default: 0.2)"
    )
    args = parser.parse_args()

    # Load and preprocess data
    df = pd.read_csv(args.file_name)
    print(f"Starting data preprocessing for {args.file_name}")
    df = data_preprocessing(df)

    # Split dataset
    train_data, test_data = split_dataset(df, args.split_mode, args.test_frac)
    
    # Save split datasets
    train_output = f"train_{args.file_name}"
    test_output = f"test_{args.file_name}"
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)
    
    print(f"\nSplit datasets saved:")
    print(f"  Training set: {train_output}")
    print(f"  Test set:     {test_output}")

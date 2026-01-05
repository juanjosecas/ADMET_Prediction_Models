"""
ADMET Model Training Module.

This module provides functionality for training ADMET prediction models using:
- Hyperparameter optimization with Hyperopt and hpsklearn
- Cross-validation with stratified or scaffold-based splitting
- Automated model selection from scikit-learn classifiers
- Comprehensive model evaluation metrics

The module supports two modes:
1. Training mode: Train a single model on train/test split
2. Cross-validation mode: Evaluate model with k-fold cross-validation
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from hyperopt import tpe
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from dgllife.utils import ScaffoldSplitter

from utils import data_preprocessing, load_features

class MoleculeDataset(Dataset):
    """
    PyTorch Dataset wrapper for molecular data.
    
    Used primarily for scaffold-based cross-validation splitting.
    
    Args:
        smiles: List of SMILES strings
        labels: List of bioactivity labels (0 or 1)
    """
    def __init__(self, smiles: list, labels: list):
        self.smiles = smiles
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of molecules in the dataset."""
        return len(self.smiles)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single molecule and its label.
        
        Args:
            idx: Index of the molecule
            
        Returns:
            Tuple of (smiles, label)
        """
        return self.smiles[idx], self.labels[idx]


def print_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Calculate and print classification performance metrics.
    
    Prints:
    - Sensitivity (Se): True positive rate, recall
    - Specificity (Sp): True negative rate
    - Accuracy (acc): Overall accuracy
    - Matthews Correlation Coefficient (MCC): Balanced metric for imbalanced data
    - Area Under ROC Curve (AUC): Discrimination capability
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    auc = roc_auc_score(y_true, y_proba)
    
    print(f'  Sensitivity (Recall):  {sensitivity:.4f}')
    print(f'  Specificity:           {specificity:.4f}')
    print(f'  Accuracy:              {accuracy:.4f}')
    print(f'  MCC:                   {mcc:.4f}')
    print(f'  AUC-ROC:               {auc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ADMET prediction models with automated hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training modes:
  --training            Train a single model on train/test split
  --cross_validation    Evaluate with k-fold cross-validation

Splitting methods (for cross-validation):
  stratify              Random split maintaining class distribution
  scaffold              Split by molecular scaffolds (Bemis-Murcko)

Examples:
  # Train a model
  python model.py --file_name smiles.csv --model_name BBB --max_eval 200 --training
  
  # Cross-validation with scaffold split
  python model.py --file_name smiles.csv --cross_validation --split scaffold --k_fold 5
        """
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default='smiles.csv',
        help='Path to input CSV file with SMILES and bioclass columns'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Name for the trained model (default: derived from file_name)'
    )
    parser.add_argument(
        '--max_eval',
        type=int,
        default=200,
        help='Maximum number of hyperparameter evaluations (default: 200)'
    )
    parser.add_argument(
        '--time_out',
        type=int,
        default=120,
        help='Maximum time in seconds for each trial (default: 120)'
    )
    parser.add_argument(
        '--cross_validation',
        action='store_true',
        default=False,
        help='Enable k-fold cross-validation mode'
    )
    parser.add_argument(
        '--training',
        action='store_true',
        default=False,
        help='Enable training mode (train on train set, evaluate on test set)'
    )
    parser.add_argument(
        '--split',
        choices=['scaffold', 'stratify'],
        default='stratify',
        help='Dataset split method for cross-validation (default: stratify)'
    )
    parser.add_argument(
        '--k_fold',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )

    args = parser.parse_args()
    print("Arguments:")
    print(args)
    
    # Load feature names
    features = load_features()
    
    # Set model name if not provided
    if args.model_name is None:
        args.model_name = args.file_name[:-4]  # Remove .csv extension
    print(f'Model name: {args.model_name}')

    if args.training:
        print("\n" + "="*80)
        print("TRAINING MODE")
        print("="*80)
        
        # Load preprocessed train and test data
        train_data = pd.read_csv(f'train_{args.file_name}')
        test_data = pd.read_csv(f'test_{args.file_name}')
        print(f'Train set size: {len(train_data)}')
        print(f'Test set size:  {len(test_data)}')
        print(f'\nHyperparameter optimization settings:')
        print(f'  max_eval:  {args.max_eval} (number of trials)')
        print(f'  time_out:  {args.time_out} seconds (per trial)')
    
        print(f'\nStarting model training...')
    
        # Train model with hyperparameter optimization
        start_time = time.time()
        estimator = HyperoptEstimator(
            classifier=any_classifier('cla'),  # Try various classifiers
            max_evals=args.max_eval,
            trial_timeout=args.time_out,
            preprocessing=any_preprocessing('pre'),  # Try various preprocessors
            algo=tpe.suggest  # Tree-structured Parzen Estimator
        )
        
        estimator.fit(train_data[features].values, np.array(train_data['bioclass']))
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
    
        # Evaluate on test set
        test_score = estimator.score(test_data[features].values, np.array(test_data['bioclass']))
        print(f'\nTest set score: {test_score:.4f}')
        
        # Get best model
        best_model = estimator.best_model()['learner']
        print(f'Best model: {best_model}')
        
        # Save model
        model_path = f'./{args.model_name}.pkl'
        pickle.dump(best_model, open(model_path, 'wb'))
        print(f'\nModel saved to: {model_path}')
        
        # Generate predictions
        predicted = best_model.predict(test_data[features].values)
        try:
            predicted_proba = best_model.predict_proba(test_data[features].values)[:, 1]
        except AttributeError:
            # Some models don't have predict_proba (e.g., SVM without probability=True)
            predicted_proba = best_model.decision_function(test_data[features].values)

        # Print performance metrics
        print('\nPerformance metrics on test set:')
        print_performance_metrics(test_data['bioclass'], predicted, predicted_proba)
        print(f'\n{"="*80}')
        print(f'Training completed for {args.model_name} using {args.file_name}')
        print("="*80)

    if args.cross_validation:
        print("\n" + "="*80)
        print(f"{args.k_fold}-FOLD CROSS-VALIDATION MODE")
        print("="*80)
        
        train_data = pd.read_csv(f'train_{args.file_name}')
        print(f'Dataset size: {len(train_data)}')
        print(f'Split method: {args.split}')
        
        X = train_data[features]
        y = train_data['bioclass']
        n_samples, n_features = X.shape
        print(f'Features: {n_features}')
        print(f'Samples:  {n_samples}')
    
        fold_results = {'aucs': [], 'sensitivities': [], 'specificities': [], 'accuracies': [], 'mccs': []}
        
        # Setup cross-validation splits
        if args.split == 'stratify':
            print(f'\nUsing stratified {args.k_fold}-fold split (maintains class distribution)')
            cv = StratifiedKFold(n_splits=args.k_fold)
            folds = cv.split(X, y)
        elif args.split == 'scaffold':
            print(f'\nUsing scaffold-based {args.k_fold}-fold split (groups by molecular scaffold)')
            list_smiles = train_data['SMILES'].tolist()
            list_labels = train_data['bioclass'].tolist()
            dataset = MoleculeDataset(list_smiles, list_labels)
            splitter = ScaffoldSplitter()
            folds = splitter.k_fold_split(dataset=dataset, k=args.k_fold)

        # Perform cross-validation
        for fold_idx, (train_subset, test_subset) in enumerate(folds):
            print(f'\n{"-"*80}')
            print(f'Fold {fold_idx + 1}/{args.k_fold}')
            print(f'{"-"*80}')
            
            # Extract train/test indices
            try:
                train_indices = train_subset.indices
                test_indices = test_subset.indices
            except AttributeError:
                train_indices = train_subset
                test_indices = test_subset

            print(f'Train samples: {len(train_indices)}')
            print(f'Test samples:  {len(test_indices)}')
            
            # Train model on this fold
            classifier = HyperoptEstimator(
                classifier=any_classifier('cla'),
                max_evals=args.max_eval,
                trial_timeout=args.time_out,
                preprocessing=any_preprocessing('pre'),
                algo=tpe.suggest
            )
    
            classifier.fit(X.iloc[train_indices].values, y.iloc[train_indices].values)
            classifier = classifier.best_model()['learner']
            print(f'Best model for fold {fold_idx + 1}: {classifier}')

            # Evaluate on test fold
            predicted = classifier.predict(X.iloc[test_indices].values)
            predicted_proba = classifier.predict_proba(X.iloc[test_indices].values)[:, 1]
            
            # Calculate metrics
            print(f'\nPerformance metrics for fold {fold_idx + 1}:')
            tn, fp, fn, tp = confusion_matrix(y.iloc[test_indices].values, predicted).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
            auc = roc_auc_score(y.iloc[test_indices].values, predicted_proba)
            
            print(f'  Sensitivity: {sensitivity:.4f}')
            print(f'  Specificity: {specificity:.4f}')
            print(f'  Accuracy:    {accuracy:.4f}')
            print(f'  MCC:         {mcc:.4f}')
            print(f'  AUC:         {auc:.4f}')
            
            fold_results['sensitivities'].append(sensitivity)
            fold_results['specificities'].append(specificity)
            fold_results['accuracies'].append(accuracy)
            fold_results['mccs'].append(mcc)
            fold_results['aucs'].append(auc)
        
        # Print summary statistics
        print(f'\n{"="*80}')
        print(f'CROSS-VALIDATION SUMMARY ({args.k_fold} folds)')
        print("="*80)
        print(f'Mean ± Std across {args.k_fold} folds:')
        print(f'  Sensitivity: {np.mean(fold_results["sensitivities"]):.4f} ± {np.std(fold_results["sensitivities"]):.4f}')
        print(f'  Specificity: {np.mean(fold_results["specificities"]):.4f} ± {np.std(fold_results["specificities"]):.4f}')
        print(f'  Accuracy:    {np.mean(fold_results["accuracies"]):.4f} ± {np.std(fold_results["accuracies"]):.4f}')
        print(f'  MCC:         {np.mean(fold_results["mccs"]):.4f} ± {np.std(fold_results["mccs"]):.4f}')
        print(f'  AUC:         {np.mean(fold_results["aucs"]):.4f} ± {np.std(fold_results["aucs"]):.4f}')
        print("="*80)

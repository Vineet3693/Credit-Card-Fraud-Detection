"""
Preprocessing Module for Credit Card Fraud Detection

This module handles data preprocessing including scaling, handling 
imbalanced data, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from typing import Tuple, Optional, Dict


def split_data(df: pd.DataFrame, 
               target_col: str = 'Class',
               test_size: float = 0.2,
               val_size: float = 0.0,
               random_state: int = 42,
               stratify: bool = True) -> Tuple:
    """
    Split the dataset into training, validation (optional), and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    target_col : str
        Name of the target column
    test_size : float
        Proportion of the dataset to include in the test split
    val_size : float
        Proportion of the dataset to include in the validation split
    random_state : int
        Random state for reproducibility
    stratify : bool
        Whether to stratify the split
        
    Returns
    -------
    tuple
        X_train, X_val (or None), X_test, y_train, y_val (or None), y_test
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train+val vs test
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
    
    # Second split: train vs val (if validation set is requested)
    if val_size > 0:
        val_size_adj = val_size / (1 - test_size)
        
        if stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size_adj,
                random_state=random_state,
                stratify=y_train
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size_adj,
                random_state=random_state
            )
        
        print(f"Train size: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"Val size: {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"Test size: {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    print(f"Train size: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Test size: {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
    
    return X_train, None, X_test, y_train, None, y_test


def scale_features(X_train: pd.DataFrame, 
                   X_test: pd.DataFrame,
                   X_val: Optional[pd.DataFrame] = None,
                   method: str = 'standard',
                   fit_on_train_only: bool = True) -> Tuple:
    """
    Scale features using specified method.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    X_val : pd.DataFrame, optional
        Validation features
    method : str
        Scaling method: 'standard' or 'robust'
    fit_on_train_only : bool
        Whether to fit scaler only on training data
        
    Returns
    -------
    tuple
        Scaled X_train, X_val (or None), X_test, and the scaler object
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        print(f"Features scaled using {method} scaling.")
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    print(f"Features scaled using {method} scaling.")
    return X_train_scaled, None, X_test_scaled, scaler


def handle_imbalance(X: pd.DataFrame, 
                     y: pd.DataFrame,
                     method: str = 'smote',
                     random_state: int = 42,
                     sampling_strategy: Optional[float] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using various resampling techniques.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    method : str
        Resampling method:
        - 'smote': SMOTE oversampling
        - 'adasyn': ADASYN oversampling
        - 'random_under': Random undersampling
        - 'nearmiss': NearMiss undersampling
        - 'smote_tomek': SMOTE + Tomek links
        - 'smote_enn': SMOTE + Edited Nearest Neighbors
        - 'none': No resampling
    random_state : int
        Random state for reproducibility
    sampling_strategy : float, optional
        Desired ratio of minority to majority class (e.g., 0.5 for 1:2 ratio)
        If None, uses default (1.0 for oversampling, 1.0 for undersampling)
        
    Returns
    -------
    tuple
        Resampled X and y
    """
    # Count original classes
    original_counts = y.value_counts()
    print(f"\nOriginal class distribution:")
    print(original_counts)
    
    if method.lower() == 'none' or method is None:
        print("\nNo resampling applied.")
        return X, y
    
    # Define resampling methods
    methods = {
        'smote': SMOTE(random_state=random_state, sampling_strategy=sampling_strategy),
        'adasyn': ADASYN(random_state=random_state, sampling_strategy=sampling_strategy),
        'random_under': RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy),
        'nearmiss': NearMiss(version=3, sampling_strategy=sampling_strategy),
        'smote_tomek': SMOTETomek(random_state=random_state, sampling_strategy=sampling_strategy),
        'smote_enn': SMOTEENN(random_state=random_state, sampling_strategy=sampling_strategy)
    }
    
    if method.lower() not in methods:
        raise ValueError(f"Unknown resampling method: {method}. "
                        f"Available methods: {list(methods.keys())}")
    
    # Apply resampling
    sampler = methods[method.lower()]
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Print new class distribution
    new_counts = y_resampled.value_counts()
    print(f"\nResampled class distribution (using {method.upper()}):")
    print(new_counts)
    
    # Calculate change
    fraud_increase = new_counts.get(1, 0) - original_counts.get(1, 0)
    if fraud_increase > 0:
        print(f"\nAdded {fraud_increase:,} synthetic fraud samples.")
    else:
        normal_decrease = original_counts.get(0, 0) - new_counts.get(0, 0)
        print(f"\nRemoved {normal_decrease:,} normal transaction samples.")
    
    return X_resampled, y_resampled


def preprocess_pipeline(df: pd.DataFrame,
                        target_col: str = 'Class',
                        test_size: float = 0.2,
                        val_size: float = 0.0,
                        scale_method: str = 'standard',
                        resampling_method: str = 'smote',
                        random_state: int = 42) -> Dict:
    """
    Complete preprocessing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    target_col : str
        Name of the target column
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set
    scale_method : str
        Scaling method
    resampling_method : str
        Resampling method for handling imbalance
    random_state : int
        Random state
        
    Returns
    -------
    dict
        Dictionary containing preprocessed data and objects
    """
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Split data
    print("\n[Step 1/3] Splitting data...")
    if val_size > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, target_col, test_size, val_size, random_state
        )
    else:
        X_train, _, X_test, y_train, _, y_test = split_data(
            df, target_col, test_size, val_size, random_state
        )
        X_val, y_val = None, None
    
    # Step 2: Scale features
    print("\n[Step 2/3] Scaling features...")
    if X_val is not None:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_test, X_val, scale_method
        )
    else:
        X_train_scaled, _, X_test_scaled, scaler = scale_features(
            X_train, X_test, method=scale_method
        )
    
    # Step 3: Handle imbalance (only on training data)
    print("\n[Step 3/3] Handling class imbalance...")
    X_train_resampled, y_train_resampled = handle_imbalance(
        X_train_scaled, y_train, method=resampling_method, random_state=random_state
    )
    
    # Prepare results
    results = {
        'X_train': X_train_resampled,
        'y_train': y_train_resampled,
        'X_val': X_val_scaled if X_val_scaled is not None else X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'original_train_size': len(X_train),
        'resampled_train_size': len(X_train_resampled)
    }
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nFinal training set size: {len(X_train_resampled):,}")
    print(f"Test set size: {len(X_test_scaled):,}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_loading import load_data
    
    df = load_data()
    results = preprocess_pipeline(df, resampling_method='smote')

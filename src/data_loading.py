"""
Data Loading Module for Credit Card Fraud Detection

This module handles loading the credit card fraud dataset from Kaggle.
Place the creditcard.csv file in the data/ directory.
"""

import pandas as pd
import os
from pathlib import Path


def load_data(data_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the dataset
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame
    """
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please download the dataset from Kaggle and place it in the data/ directory."
        )
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def get_data_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
    """
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Statistical Summary ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Class Distribution ---")
    if 'Class' in df.columns:
        print(df['Class'].value_counts())
        print(f"\nFraud percentage: {(df['Class'].sum() / len(df) * 100):.4f}%")


if __name__ == "__main__":
    # Example usage
    df = load_data()
    get_data_info(df)

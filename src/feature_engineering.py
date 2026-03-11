"""
Feature Engineering Module for Credit Card Fraud Detection

This module provides functions for creating new features, feature selection,
and feature transformation.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List, Dict


def create_time_features(df: pd.DataFrame, time_col: str = 'Time') -> pd.DataFrame:
    """
    Create time-based features from the Time column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    time_col : str
        Name of the time column
        
    Returns
    -------
    pd.DataFrame
        Dataset with added time features
    """
    df = df.copy()
    
    # Convert time to hours (assuming seconds since midnight)
    df['Time_Hours'] = df[time_col] / 3600 % 24
    
    # Create time bins (morning, afternoon, evening, night)
    df['Time_Period'] = pd.cut(
        df['Time_Hours'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )
    
    # Binary features for time periods
    df['Is_Night'] = ((df['Time_Hours'] >= 0) & (df['Time_Hours'] < 6)).astype(int)
    df['Is_Morning'] = ((df['Time_Hours'] >= 6) & (df['Time_Hours'] < 12)).astype(int)
    df['Is_Afternoon'] = ((df['Time_Hours'] >= 12) & (df['Time_Hours'] < 18)).astype(int)
    df['Is_Evening'] = ((df['Time_Hours'] >= 18) & (df['Time_Hours'] < 24)).astype(int)
    
    # Cyclical encoding of time (preserves continuity)
    df['Time_Hours_Sin'] = np.sin(2 * np.pi * df['Time_Hours'] / 24)
    df['Time_Hours_Cos'] = np.cos(2 * np.pi * df['Time_Hours'] / 24)
    
    print("Time features created successfully!")
    return df


def create_amount_features(df: pd.DataFrame, amount_col: str = 'Amount') -> pd.DataFrame:
    """
    Create amount-based features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    amount_col : str
        Name of the amount column
        
    Returns
    -------
    pd.DataFrame
        Dataset with added amount features
    """
    df = df.copy()
    
    # Log transform (add 1 to handle zeros)
    df['Amount_Log'] = np.log1p(df[amount_col])
    
    # Binning
    df['Amount_Bin'] = pd.qcut(df[amount_col], q=10, labels=False, duplicates='drop')
    
    # Binary flags
    df['Is_High_Amount'] = (df[amount_col] > df[amount_col].quantile(0.95)).astype(int)
    df['Is_Very_High_Amount'] = (df[amount_col] > df[amount_col].quantile(0.99)).astype(int)
    df['Is_Low_Amount'] = (df[amount_col] < df[amount_col].quantile(0.25)).astype(int)
    
    # Z-score (how many standard deviations from mean)
    df['Amount_ZScore'] = (df[amount_col] - df[amount_col].mean()) / df[amount_col].std()
    
    print("Amount features created successfully!")
    return df


def create_interaction_features(df: pd.DataFrame, 
                                pca_cols: Optional[List[str]] = None,
                                n_interactions: int = 10) -> pd.DataFrame:
    """
    Create interaction features between PCA components or top correlated features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    pca_cols : list, optional
        List of PCA column names to create interactions for
    n_interactions : int
        Number of interaction features to create
        
    Returns
    -------
    pd.DataFrame
        Dataset with added interaction features
    """
    df = df.copy()
    
    if pca_cols is None:
        # Use V1-V5 as default (typically most correlated with target)
        pca_cols = [col for col in df.columns if col.startswith('V')][:5]
    
    # Create pairwise products
    count = 0
    for i in range(len(pca_cols)):
        for j in range(i+1, len(pca_cols)):
            if count >= n_interactions:
                break
            
            col1, col2 = pca_cols[i], pca_cols[j]
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                count += 1
    
    print(f"Created {count} interaction features!")
    return df


def apply_pca(df: pd.DataFrame, 
              features: Optional[List[str]] = None,
              n_components: float = 0.95,
              random_state: int = 42) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply PCA dimensionality reduction.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    features : list, optional
        Features to apply PCA on. If None, uses all V columns.
    n_components : float or int
        Number of components or variance to retain
    random_state : int
        Random state
        
    Returns
    -------
    tuple
        DataFrame with PCA components and the fitted PCA object
    """
    df = df.copy()
    
    if features is None:
        features = [col for col in df.columns if col.startswith('V')]
    
    # Extract features
    X = df[features].values
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    # Create PCA component columns
    for i in range(X_pca.shape[1]):
        df[f'PCA_{i+1}'] = X_pca[:, i]
    
    # Print explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA Applied:")
    print(f"Original features: {len(features)}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Variance retained: {cumulative_variance[-1]*100:.2f}%")
    print(f"\nExplained variance per component:")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"  PC{i+1}: {var*100:.2f}%")
    
    return df, pca


def select_features(X: pd.DataFrame, 
                    y: pd.Series,
                    method: str = 'mutual_info',
                    k: int = 20) -> Tuple[List[str], object]:
    """
    Select top k features using specified method.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    method : str
        Selection method: 'f_test', 'mutual_info', or 'correlation'
    k : int
        Number of features to select
        
    Returns
    -------
    tuple
        List of selected feature names and the selector object
    """
    if method == 'f_test':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        # mutual_info_classif doesn't have a k parameter in constructor
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    elif method == 'correlation':
        # Custom correlation-based selection
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected_features = correlations.head(k).index.tolist()
        print(f"\nSelected {k} features based on correlation:")
        print(selected_features)
        return selected_features, None
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    # Fit selector
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()
    
    print(f"\nSelected {len(selected_features)} features using {method}:")
    print(selected_features)
    
    return selected_features, selector


def get_feature_importance_from_model(model, 
                                      feature_names: List[str],
                                      top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Parameters
    ----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to return
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(importance_df.head(top_n))
    
    return importance_df


def engineer_features_pipeline(df: pd.DataFrame,
                               add_time_features: bool = True,
                               add_amount_features: bool = True,
                               add_interactions: bool = False,
                               apply_pca_transform: bool = False,
                               n_pca_components: float = 0.95,
                               select_k_features: Optional[int] = None,
                               exclude_original_v: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    add_time_features : bool
        Whether to add time-based features
    add_amount_features : bool
        Whether to add amount-based features
    add_interactions : bool
        Whether to add interaction features
    apply_pca_transform : bool
        Whether to apply PCA
    n_pca_components : float
        Number of PCA components or variance to retain
    select_k_features : int, optional
        Number of features to select at the end
    exclude_original_v : bool
        Whether to exclude original V columns after PCA
        
    Returns
    -------
    tuple
        Engineered DataFrame and metadata dictionary
    """
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    df_engineered = df.copy()
    metadata = {
        'original_columns': df.columns.tolist(),
        'added_features': [],
        'pca_object': None,
        'selector_object': None,
        'selected_features': None
    }
    
    # Step 1: Time features
    if add_time_features:
        print("\n[Step 1/4] Creating time features...")
        df_engineered = create_time_features(df_engineered)
        time_features = ['Time_Hours', 'Time_Hours_Sin', 'Time_Hours_Cos', 
                        'Is_Night', 'Is_Morning', 'Is_Afternoon', 'Is_Evening']
        metadata['added_features'].extend(time_features)
    
    # Step 2: Amount features
    if add_amount_features:
        print("\n[Step 2/4] Creating amount features...")
        df_engineered = create_amount_features(df_engineered)
        amount_features = ['Amount_Log', 'Amount_Bin', 'Is_High_Amount', 
                          'Is_Very_High_Amount', 'Is_Low_Amount', 'Amount_ZScore']
        metadata['added_features'].extend(amount_features)
    
    # Step 3: Interaction features
    if add_interactions:
        print("\n[Step 3/4] Creating interaction features...")
        df_engineered = create_interaction_features(df_engineered)
        interaction_cols = [col for col in df_engineered.columns if '_x_' in col]
        metadata['added_features'].extend(interaction_cols)
    
    # Step 4: PCA
    if apply_pca_transform:
        print("\n[Step 4/4] Applying PCA...")
        v_columns = [col for col in df_engineered.columns if col.startswith('V')]
        df_engineered, pca = apply_pca(df_engineered, features=v_columns, 
                                       n_components=n_pca_components)
        metadata['pca_object'] = pca
        
        if exclude_original_v:
            df_engineered = df_engineered.drop(columns=v_columns)
    
    # Step 5: Feature selection
    if select_k_features is not None:
        print(f"\n[Final Step] Selecting top {select_k_features} features...")
        # Exclude target column
        feature_cols = [col for col in df_engineered.columns if col != 'Class']
        
        # For feature selection, we need to work with numeric data only
        numeric_df = df_engineered.select_dtypes(include=[np.number])
        if 'Class' in numeric_df.columns:
            X = numeric_df.drop(columns=['Class'])
            y = numeric_df['Class']
        else:
            X = numeric_df
            y = df_engineered['Class']
        
        selected_features, selector = select_features(X, y, k=select_k_features)
        metadata['selected_features'] = selected_features
        metadata['selector_object'] = selector
        
        # Keep only selected features + Class + Time + Amount (original)
        keep_cols = selected_features + ['Class', 'Time', 'Amount']
        keep_cols = [col for col in keep_cols if col in df_engineered.columns]
        df_engineered = df_engineered[keep_cols]
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"\nOriginal features: {len(metadata['original_columns'])}")
    print(f"Final features: {len(df_engineered.columns)}")
    print(f"New features added: {len(metadata['added_features'])}")
    
    return df_engineered, metadata


if __name__ == "__main__":
    # Example usage
    from data_loading import load_data
    
    df = load_data()
    df_engineered, metadata = engineer_features_pipeline(
        df,
        add_time_features=True,
        add_amount_features=True,
        add_interactions=False,
        apply_pca_transform=False
    )
    print(f"\nFinal shape: {df_engineered.shape}")

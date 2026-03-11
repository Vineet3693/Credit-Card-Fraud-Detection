"""
Model Training Module for Credit Card Fraud Detection

This module provides functions for training various machine learning models
including Logistic Regression, Random Forest, XGBoost, LightGBM, and more.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef
)
from typing import Dict, Optional, Tuple, List
import joblib
import os


def get_models() -> Dict[str, object]:
    """
    Return a dictionary of initialized models with default parameters.
    
    Returns
    -------
    dict
        Dictionary of model names to initialized model objects
    """
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=42,
            scale_pos_weight=1,  # Will be adjusted based on data
            n_jobs=-1,
            eval_metric='auc'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100,
            random_state=42,
            verbose=0,
            thread_count=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    }
    
    return models


def calculate_metrics(y_true: pd.Series, 
                      y_pred: np.ndarray, 
                      y_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
        
    Returns
    -------
    dict
        Dictionary of metric names to values
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'Average Precision': average_precision_score(y_true, y_proba),
        'Matthews Correlation': matthews_corrcoef(y_true, y_pred)
    }
    
    return metrics


def train_model(model_name: str,
                model: object,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                X_val: Optional[pd.DataFrame] = None,
                y_val: Optional[pd.Series] = None,
                X_test: pd.DataFrame = None,
                y_test: pd.Series = None,
                fit_params: Optional[Dict] = None,
                save_path: Optional[str] = None,
                verbose: bool = True) -> Tuple[object, Dict, Dict]:
    """
    Train a single model and evaluate it.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    model : object
        Initialized model object
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    fit_params : dict, optional
        Additional parameters for model.fit()
    save_path : str, optional
        Path to save the trained model
    verbose : bool
        Whether to print training information
        
    Returns
    -------
    tuple
        Trained model, training metrics, test metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
    
    # Prepare fit parameters
    if fit_params is None:
        fit_params = {}
    
    # Add validation set for models that support it
    if X_val is not None and y_val is not None:
        if model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            if model_name == 'XGBoost':
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = False
            elif model_name == 'LightGBM':
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = -1
            elif model_name == 'CatBoost':
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = 0
    
    # Adjust scale_pos_weight for XGBoost based on class imbalance
    if model_name == 'XGBoost':
        if 'scale_pos_weight' not in fit_params:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                model.scale_pos_weight = neg_count / pos_count
    
    # Train the model
    model.fit(X_train, y_train, **fit_params)
    
    # Evaluate on training set
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
    else:
        y_train_proba = model.decision_function(X_train)
    
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    
    if verbose:
        print(f"\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Evaluate on test set
    test_metrics = {}
    if X_test is not None and y_test is not None:
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = model.decision_function(X_test)
        
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        if verbose:
            print(f"\nTest Metrics:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_test_pred))
            
            print(f"Confusion Matrix:")
            print(confusion_matrix(y_test, y_test_pred))
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        if verbose:
            print(f"\nModel saved to: {save_path}")
    
    return model, train_metrics, test_metrics


def train_multiple_models(X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          model_names: Optional[List[str]] = None,
                          save_dir: Optional[str] = None,
                          verbose: bool = True) -> Dict[str, Tuple[object, Dict, Dict]]:
    """
    Train multiple models and compare their performance.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation labels
    model_names : list, optional
        List of model names to train. If None, trains all available models.
    save_dir : str, optional
        Directory to save trained models
    verbose : bool
        Whether to print training information
        
    Returns
    -------
    dict
        Dictionary of model names to (model, train_metrics, test_metrics) tuples
    """
    # Get all available models
    all_models = get_models()
    
    # Filter models if specific names provided
    if model_names is not None:
        models_to_train = {name: all_models[name] for name in model_names if name in all_models}
    else:
        models_to_train = all_models
    
    if verbose:
        print("="*60)
        print("TRAINING MULTIPLE MODELS")
        print("="*60)
        print(f"\nTraining {len(models_to_train)} models...")
    
    results = {}
    
    for model_name, model in models_to_train.items():
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"{model_name}_model.joblib")
        
        try:
            trained_model, train_metrics, test_metrics = train_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                save_path=save_path,
                verbose=verbose
            )
            
            results[model_name] = (trained_model, train_metrics, test_metrics)
            
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            results[model_name] = None
    
    # Print comparison table
    if verbose and len(results) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON (Test Set)")
        print("="*60)
        
        comparison_data = []
        for model_name, result in results.items():
            if result is not None:
                _, _, test_metrics = result
                comparison_data.append({
                    'Model': model_name,
                    'ROC-AUC': test_metrics.get('ROC-AUC', 0),
                    'F1 Score': test_metrics.get('F1 Score', 0),
                    'Precision': test_metrics.get('Precision', 0),
                    'Recall': test_metrics.get('Recall', 0),
                    'Accuracy': test_metrics.get('Accuracy', 0)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Save comparison to CSV
        if save_dir:
            comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
            print(f"\nComparison saved to: {os.path.join(save_dir, 'model_comparison.csv')}")
    
    return results


def load_trained_model(model_path: str) -> object:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file
        
    Returns
    -------
    object
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded successfully from: {model_path}")
    return model


if __name__ == "__main__":
    # Example usage
    from data_loading import load_data
    from preprocessing import preprocess_pipeline
    
    # Load and preprocess data
    df = load_data()
    preprocessed = preprocess_pipeline(df, resampling_method='smote')
    
    X_train = preprocessed['X_train']
    y_train = preprocessed['y_train']
    X_test = preprocessed['X_test']
    y_test = preprocessed['y_test']
    
    # Train multiple models
    results = train_multiple_models(
        X_train, y_train, X_test, y_test,
        model_names=['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM'],
        save_dir='models',
        verbose=True
    )

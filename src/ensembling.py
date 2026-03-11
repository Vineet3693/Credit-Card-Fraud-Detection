"""
Ensembling Module for Credit Card Fraud Detection

This module provides functions for creating ensemble models including
Voting, Stacking, Bagging, and custom weighted ensembles.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier, BaggingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score
)
from typing import Dict, List, Tuple, Optional
import joblib
import os


def create_voting_classifier(models: Dict[str, object],
                             voting: str = 'soft') -> VotingClassifier:
    """
    Create a voting classifier from multiple models.
    
    Parameters
    ----------
    models : dict
        Dictionary of model names to initialized model objects
    voting : str
        Voting strategy: 'soft' for probability-based, 'hard' for majority vote
        
    Returns
    -------
    VotingClassifier
        Initialized voting classifier
    """
    estimators = [(name, model) for name, model in models.items()]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )
    
    print(f"Voting Classifier created with {len(estimators)} models ({voting} voting)")
    return voting_clf


def create_stacking_classifier(base_models: Dict[str, object],
                               final_estimator=None,
                               stack_method: str = 'predict_proba',
                               cv: int = 5) -> StackingClassifier:
    """
    Create a stacking classifier with base models and a meta-learner.
    
    Parameters
    ----------
    base_models : dict
        Dictionary of base model names to initialized model objects
    final_estimator : object, optional
        Meta-learner model. If None, uses LogisticRegression.
    stack_method : str
        Method to generate predictions from base models
    cv : int
        Number of cross-validation folds
        
    Returns
    -------
    StackingClassifier
        Initialized stacking classifier
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method=stack_method,
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )
    
    print(f"Stacking Classifier created with {len(estimators)} base models")
    print(f"Meta-learner: {final_estimator.__class__.__name__}")
    return stacking_clf


def create_bagging_ensemble(base_model: object,
                            n_estimators: int = 10,
                            max_samples: float = 0.8,
                            max_features: float = 0.8,
                            bootstrap: bool = True,
                            random_state: int = 42) -> BaggingClassifier:
    """
    Create a bagging ensemble from a base model.
    
    Parameters
    ----------
    base_model : object
        Base model to ensemble
    n_estimators : int
        Number of estimators in the ensemble
    max_samples : float
        Fraction of samples to draw for each estimator
    max_features : float
        Fraction of features to draw for each estimator
    bootstrap : bool
        Whether to sample with replacement
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    BaggingClassifier
        Initialized bagging classifier
    """
    bagging_clf = BaggingClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=-1
    )
    
    print(f"Bagging Ensemble created with {n_estimators} estimators")
    print(f"Base model: {base_model.__class__.__name__}")
    return bagging_clf


def create_weighted_ensemble(models: Dict[str, object],
                             weights: Dict[str, float]) -> callable:
    """
    Create a custom weighted ensemble function.
    
    Parameters
    ----------
    models : dict
        Dictionary of trained model objects
    weights : dict
        Dictionary of model names to weights (should sum to 1)
        
    Returns
    -------
    callable
        Function that takes X and returns weighted predictions
    """
    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    def predict(X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions."""
        weighted_proba = None
        
        for model_name, model in models.items():
            weight = normalized_weights.get(model_name, 0)
            if weight > 0:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]
                else:
                    proba = model.decision_function(X)
                
                if weighted_proba is None:
                    weighted_proba = weight * proba
                else:
                    weighted_proba += weight * proba
        
        return (weighted_proba >= 0.5).astype(int)
    
    def predict_proba(X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble probability predictions."""
        weighted_proba = None
        
        for model_name, model in models.items():
            weight = normalized_weights.get(model_name, 0)
            if weight > 0:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]
                else:
                    proba = model.decision_function(X)
                
                if weighted_proba is None:
                    weighted_proba = weight * proba
                else:
                    weighted_proba += weight * proba
        
        return weighted_proba
    
    return predict, predict_proba


def evaluate_ensemble(ensemble_model,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      name: str = "Ensemble") -> Dict[str, float]:
    """
    Evaluate an ensemble model.
    
    Parameters
    ----------
    ensemble_model : object
        Trained ensemble model or tuple of (predict, predict_proba) functions
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    name : str
        Name of the ensemble for reporting
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Check if it's a custom weighted ensemble (tuple of functions)
    if isinstance(ensemble_model, tuple):
        predict_func, predict_proba_func = ensemble_model
        y_pred = predict_func(X_test)
        y_proba = predict_proba_func(X_test)
    else:
        # Standard sklearn model
        if hasattr(ensemble_model, 'predict_proba'):
            y_proba = ensemble_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = ensemble_model.decision_function(X_test)
        y_pred = ensemble_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Average Precision': average_precision_score(y_test, y_proba)
    }
    
    print(f"\n{'='*60}")
    print(f"{name} - Evaluation Metrics")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return metrics


def train_and_evaluate_ensemble(ensemble_type: str,
                                 base_models: Dict[str, object],
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series,
                                 X_val: Optional[pd.DataFrame] = None,
                                 y_val: Optional[pd.Series] = None,
                                 ensemble_params: Optional[Dict] = None,
                                 weights: Optional[Dict[str, float]] = None,
                                 save_path: Optional[str] = None) -> Tuple[object, Dict]:
    """
    Train and evaluate an ensemble model.
    
    Parameters
    ----------
    ensemble_type : str
        Type of ensemble: 'voting', 'stacking', 'bagging', or 'weighted'
    base_models : dict
        Dictionary of base model names to initialized model objects
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
    ensemble_params : dict, optional
        Additional parameters for ensemble creation
    weights : dict, optional
        Weights for weighted ensemble
    save_path : str, optional
        Path to save the trained ensemble
        
    Returns
    -------
    tuple
        Trained ensemble model and evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Creating {ensemble_type.upper()} Ensemble")
    print(f"{'='*60}")
    
    # Create ensemble based on type
    if ensemble_params is None:
        ensemble_params = {}
    
    if ensemble_type.lower() == 'voting':
        ensemble = create_voting_classifier(
            base_models,
            voting=ensemble_params.get('voting', 'soft')
        )
    elif ensemble_type.lower() == 'stacking':
        ensemble = create_stacking_classifier(
            base_models,
            final_estimator=ensemble_params.get('final_estimator', None),
            stack_method=ensemble_params.get('stack_method', 'predict_proba'),
            cv=ensemble_params.get('cv', 5)
        )
    elif ensemble_type.lower() == 'bagging':
        base_model = list(base_models.values())[0]
        ensemble = create_bagging_ensemble(
            base_model,
            n_estimators=ensemble_params.get('n_estimators', 10),
            max_samples=ensemble_params.get('max_samples', 0.8),
            max_features=ensemble_params.get('max_features', 0.8)
        )
    elif ensemble_type.lower() == 'weighted':
        if weights is None:
            # Equal weights by default
            weights = {name: 1.0/len(base_models) for name in base_models}
        # For weighted ensemble, we need trained models first
        trained_models = {}
        for name, model in base_models.items():
            print(f"Training {name} for weighted ensemble...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        ensemble = create_weighted_ensemble(trained_models, weights)
        
        # Evaluate immediately since models are already trained
        metrics = evaluate_ensemble(ensemble, X_test, y_test, name="Weighted Ensemble")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump({'models': trained_models, 'weights': weights}, save_path)
            print(f"\nWeighted ensemble saved to: {save_path}")
        
        return ensemble, metrics
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    # Train the ensemble
    print(f"\nTraining {ensemble_type} ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_ensemble(ensemble, X_test, y_test, name=f"{ensemble_type.title()} Ensemble")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(ensemble, save_path)
        print(f"\nEnsemble saved to: {save_path}")
    
    return ensemble, metrics


def compare_ensembles(base_models: Dict[str, object],
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      ensemble_types: List[str] = ['voting', 'stacking', 'weighted'],
                      save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Compare different ensemble strategies.
    
    Parameters
    ----------
    base_models : dict
        Dictionary of base models
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    ensemble_types : list
        List of ensemble types to compare
    save_dir : str, optional
        Directory to save results
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    print("="*60)
    print("COMPARING ENSEMBLE STRATEGIES")
    print("="*60)
    
    results = []
    
    for ens_type in ensemble_types:
        try:
            weights = None
            if ens_type == 'weighted':
                # Use ROC-AUC from individual models as weights
                # This is a simple heuristic
                weights = {name: 1.0/len(base_models) for name in base_models}
            
            _, metrics = train_and_evaluate_ensemble(
                ensemble_type=ens_type,
                base_models=base_models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                weights=weights
            )
            
            results.append({
                'Ensemble Type': ens_type,
                'ROC-AUC': metrics['ROC-AUC'],
                'F1 Score': metrics['F1 Score'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'Accuracy': metrics['Accuracy']
            })
            
        except Exception as e:
            print(f"Error with {ens_type}: {str(e)}")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    print("\n" + "="*60)
    print("ENSEMBLE COMPARISON RESULTS")
    print("="*60)
    print("\n" + comparison_df.to_string(index=False))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(save_dir, 'ensemble_comparison.csv'), index=False)
        print(f"\nResults saved to: {os.path.join(save_dir, 'ensemble_comparison.csv')}")
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    from data_loading import load_data
    from preprocessing import preprocess_pipeline
    from model_training import get_models
    
    # Load and preprocess data
    df = load_data()
    preprocessed = preprocess_pipeline(df, resampling_method='smote')
    
    X_train = preprocessed['X_train']
    y_train = preprocessed['y_train']
    X_test = preprocessed['X_test']
    y_test = preprocessed['y_test']
    
    # Get base models
    all_models = get_models()
    base_models = {
        'RF': all_models['RandomForest'],
        'XGB': all_models['XGBoost'],
        'LGBM': all_models['LightGBM']
    }
    
    # Compare ensembles
    comparison = compare_ensembles(
        base_models=base_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        ensemble_types=['voting', 'stacking'],
        save_dir='models'
    )

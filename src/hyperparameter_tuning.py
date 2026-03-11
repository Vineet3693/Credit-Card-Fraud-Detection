"""
Hyperparameter Tuning Module for Credit Card Fraud Detection

This module provides functions for hyperparameter optimization using
Grid Search, Random Search, and Optuna (Bayesian Optimization).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from typing import Dict, Optional, Tuple, List
import joblib
import os

try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna")


def get_param_grids() -> Dict[str, Dict]:
    """
    Return parameter grids for different models.
    
    Returns
    -------
    dict
        Dictionary of model names to parameter grids
    """
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'scale_pos_weight': [1, 10, 50],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'class_weight': ['balanced', None]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
    
    return param_grids


def tune_with_grid_search(model,
                          param_grid: Dict,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          cv: int = 5,
                          scoring: str = 'roc_auc',
                          n_jobs: int = -1,
                          verbose: int = 1,
                          save_path: Optional[str] = None) -> Tuple[object, Dict]:
    """
    Tune hyperparameters using Grid Search.
    
    Parameters
    ----------
    model : object
        Initialized model to tune
    param_grid : dict
        Parameter grid for grid search
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    save_path : str, optional
        Path to save the best model
        
    Returns
    -------
    tuple
        Best model and results dictionary
    """
    print(f"\n{'='*60}")
    print("GRID SEARCH HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameter combinations: {len(list(GridSearchCV(model, param_grid, cv=cv).cv.split(X_train, y_train))) * len(list(GridSearchCV(model, param_grid, cv=cv)).param_combinations_ if hasattr(GridSearchCV(model, param_grid, cv=cv), 'param_combinations_') else 'N/A')}")
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Fit
    print("\nStarting Grid Search...")
    grid_search.fit(X_train, y_train)
    
    # Results
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best {scoring} Score: {grid_search.best_score_:.4f}")
    
    # Save results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'best_estimator': grid_search.best_estimator_
    }
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(grid_search.best_estimator_, save_path)
        print(f"\nBest model saved to: {save_path}")
        
        # Save full results
        results_path = save_path.replace('.joblib', '_results.joblib')
        joblib.dump(results, results_path)
        print(f"Results saved to: {results_path}")
    
    return grid_search.best_estimator_, results


def tune_with_random_search(model,
                            param_distributions: Dict,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            n_iter: int = 50,
                            cv: int = 5,
                            scoring: str = 'roc_auc',
                            n_jobs: int = -1,
                            verbose: int = 1,
                            random_state: int = 42,
                            save_path: Optional[str] = None) -> Tuple[object, Dict]:
    """
    Tune hyperparameters using Random Search.
    
    Parameters
    ----------
    model : object
        Initialized model to tune
    param_distributions : dict
        Parameter distributions for random search
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_iter : int
        Number of parameter settings sampled
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    random_state : int
        Random state
    save_path : str, optional
        Path to save the best model
        
    Returns
    -------
    tuple
        Best model and results dictionary
    """
    print(f"\n{'='*60}")
    print("RANDOM SEARCH HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Iterations: {n_iter}")
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Initialize Random Search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True
    )
    
    # Fit
    print("\nStarting Random Search...")
    random_search.fit(X_train, y_train)
    
    # Results
    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best {scoring} Score: {random_search.best_score_:.4f}")
    
    # Save results
    results = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'best_estimator': random_search.best_estimator_
    }
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(random_search.best_estimator_, save_path)
        print(f"\nBest model saved to: {save_path}")
        
        # Save full results
        results_path = save_path.replace('.joblib', '_results.joblib')
        joblib.dump(results, results_path)
        print(f"Results saved to: {results_path}")
    
    return random_search.best_estimator_, results


def optimize_with_optuna(model_name: str,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         n_trials: int = 100,
                         timeout: Optional[int] = None,
                         cv: int = 5,
                         scoring: str = 'roc_auc',
                         study_name: Optional[str] = None,
                         save_path: Optional[str] = None,
                         verbose: bool = True) -> Tuple[Dict, object]:
    """
    Tune hyperparameters using Optuna (Bayesian Optimization).
    
    Parameters
    ----------
    model_name : str
        Name of the model to optimize
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_trials : int
        Number of trials
    timeout : int, optional
        Timeout in seconds
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    study_name : str, optional
        Name of the Optuna study
    save_path : str, optional
        Path to save the best model
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    tuple
        Best parameters and best model
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    print(f"\n{'='*60}")
    print("OPTUNA BAYESIAN OPTIMIZATION")
    print(f"{'='*60}")
    print(f"\nModel: {model_name}")
    print(f"Trials: {n_trials}")
    
    def objective(trial):
        """Objective function for Optuna."""
        
        # Model-specific parameter suggestions
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'auc'
            }
            from xgboost import XGBClassifier
            model = XGBClassifier(**params)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**params)
            
        elif model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
            
        elif model_name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 1e-5, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': 1000,
                'random_state': 42
            }
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**params)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            
            if scoring == 'roc_auc':
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
            elif scoring == 'f1':
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val)
                score = model.score(X_val, y_val)
            
            scores.append(score)
        
        return np.mean(scores)
    
    # Create study
    if study_name is None:
        study_name = f"{model_name}_optimization"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    print("\nStarting Optuna optimization...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    
    # Results
    print(f"\nBest Parameters: {study.best_params}")
    print(f"Best {scoring} Score: {study.best_value:.4f}")
    
    # Train final model with best parameters
    if model_name == 'XGBoost':
        from xgboost import XGBClassifier
        best_model = XGBClassifier(**study.best_params)
    elif model_name == 'LightGBM':
        from lightgbm import LGBMClassifier
        best_model = LGBMClassifier(**study.best_params)
    elif model_name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        best_model = RandomForestClassifier(**study.best_params)
    elif model_name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        best_model = LogisticRegression(**study.best_params)
    
    best_model.fit(X_train, y_train)
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'trials': study.trials,
        'best_model': best_model
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(best_model, save_path)
        print(f"\nBest model saved to: {save_path}")
        
        # Save study
        study_path = save_path.replace('_model.joblib', '_study.joblib')
        joblib.dump(study, study_path)
        print(f"Study saved to: {study_path}")
    
    return study.best_params, best_model


def compare_tuning_methods(model,
                           model_name: str,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           param_grid: Dict,
                           n_trials_optuna: int = 50,
                           n_iter_random: int = 30,
                           cv: int = 5,
                           save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Compare different tuning methods.
    
    Parameters
    ----------
    model : object
        Initialized model
    model_name : str
        Name of the model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    param_grid : dict
        Parameter grid
    n_trials_optuna : int
        Number of trials for Optuna
    n_iter_random : int
        Number of iterations for Random Search
    cv : int
        Cross-validation folds
    save_dir : str, optional
        Directory to save results
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    print("="*60)
    print("COMPARING TUNING METHODS")
    print("="*60)
    
    results = []
    
    # Grid Search (use subset of param_grid for speed)
    print("\n[1/3] Running Grid Search...")
    try:
        _, grid_results = tune_with_grid_search(
            model, param_grid, X_train, y_train, cv=cv, verbose=0,
            save_path=f"{save_dir}/{model_name}_grid.joblib" if save_dir else None
        )
        results.append({
            'Method': 'Grid Search',
            'Best Score': grid_results['best_score'],
            'Best Params': str(grid_results['best_params'])
        })
    except Exception as e:
        print(f"Grid Search failed: {e}")
    
    # Random Search
    print("\n[2/3] Running Random Search...")
    try:
        _, random_results = tune_with_random_search(
            model, param_grid, X_train, y_train, 
            n_iter=n_iter_random, cv=cv, verbose=0,
            save_path=f"{save_dir}/{model_name}_random.joblib" if save_dir else None
        )
        results.append({
            'Method': 'Random Search',
            'Best Score': random_results['best_score'],
            'Best Params': str(random_results['best_params'])
        })
    except Exception as e:
        print(f"Random Search failed: {e}")
    
    # Optuna
    if OPTUNA_AVAILABLE:
        print("\n[3/3] Running Optuna...")
        try:
            optuna_params, optuna_model = optimize_with_optuna(
                model_name, X_train, y_train,
                n_trials=n_trials_optuna, cv=cv, verbose=False,
                save_path=f"{save_dir}/{model_name}_optuna.joblib" if save_dir else None
            )
            results.append({
                'Method': 'Optuna',
                'Best Score': optuna_model.score(X_train, y_train),
                'Best Params': str(optuna_params)
            })
        except Exception as e:
            print(f"Optuna failed: {e}")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Best Score', ascending=False)
    
    print("\n" + "="*60)
    print("TUNING METHOD COMPARISON")
    print("="*60)
    print("\n" + comparison_df.to_string(index=False))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(save_dir, 'tuning_comparison.csv'), index=False)
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    from data_loading import load_data
    from preprocessing import preprocess_pipeline
    from xgboost import XGBClassifier
    
    # Load and preprocess data
    df = load_data()
    preprocessed = preprocess_pipeline(df, resampling_method='smote')
    
    X_train = preprocessed['X_train']
    y_train = preprocessed['y_train']
    
    # Optuna optimization for XGBoost
    best_params, best_model = optimize_with_optuna(
        model_name='XGBoost',
        X_train=X_train,
        y_train=y_train,
        n_trials=50,
        cv=5,
        save_path='models/xgboost_optimized.joblib'
    )
    
    print(f"\nBest XGBoost parameters: {best_params}")

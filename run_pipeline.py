"""
Complete Pipeline Runner for Credit Card Fraud Detection

This script runs the entire machine learning pipeline from data loading
to model training and evaluation.

Usage:
    python run_pipeline.py [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loading import load_data, get_data_info
from src.eda import run_full_eda
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features_pipeline
from src.model_training import train_multiple_models, get_models
from src.ensembling import compare_ensembles
from src.hyperparameter_tuning import tune_with_random_search, get_param_grids


def run_complete_pipeline(data_path: str = "data/creditcard.csv",
                          output_dir: str = "models",
                          run_eda: bool = True,
                          run_feature_engineering: bool = True,
                          run_ensembling: bool = True,
                          run_hyperparameter_tuning: bool = False,
                          resampling_method: str = 'smote',
                          models_to_train: list = None):
    """
    Run the complete fraud detection pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
    output_dir : str
        Directory to save models and outputs
    run_eda : bool
        Whether to run EDA
    run_feature_engineering : bool
        Whether to run feature engineering
    run_ensembling : bool
        Whether to run ensembling
    run_hyperparameter_tuning : bool
        Whether to run hyperparameter tuning
    resampling_method : str
        Resampling method for handling class imbalance
    models_to_train : list
        List of model names to train
    """
    print("="*80)
    print("CREDIT CARD FRAUD DETECTION - COMPLETE PIPELINE")
    print("="*80)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("notebooks/eda_outputs", exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    df = load_data(data_path)
    get_data_info(df)
    
    # =========================================================================
    # STEP 2: Exploratory Data Analysis
    # =========================================================================
    if run_eda:
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*80)
        run_full_eda(df, output_dir="notebooks/eda_outputs")
    else:
        print("\nSkipping EDA (set run_eda=True to enable)")
    
    # =========================================================================
    # STEP 3: Preprocessing
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: PREPROCESSING")
    print("="*80)
    
    preprocessed = preprocess_pipeline(
        df,
        test_size=0.2,
        val_size=0.0,
        scale_method='standard',
        resampling_method=resampling_method,
        random_state=42
    )
    
    X_train = preprocessed['X_train']
    y_train = preprocessed['y_train']
    X_test = preprocessed['X_test']
    y_test = preprocessed['y_test']
    
    # Save scaler for deployment
    import joblib
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(preprocessed['scaler'], scaler_path)
    print(f"\nScaler saved to: {scaler_path}")
    
    # =========================================================================
    # STEP 4: Feature Engineering (Optional)
    # =========================================================================
    if run_feature_engineering:
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING")
        print("="*80)
        
        # Note: For this dataset, the V features are already PCA-transformed
        # So we'll just add time and amount features
        df_engineered, metadata = engineer_features_pipeline(
            df,
            add_time_features=True,
            add_amount_features=True,
            add_interactions=False,
            apply_pca_transform=False,
            select_k_features=None
        )
        
        # Re-preprocess with engineered features
        print("\nRe-preprocessing with engineered features...")
        preprocessed_eng = preprocess_pipeline(
            df_engineered,
            test_size=0.2,
            val_size=0.0,
            scale_method='standard',
            resampling_method=resampling_method,
            random_state=42
        )
        
        X_train = preprocessed_eng['X_train']
        y_train = preprocessed_eng['y_train']
        X_test = preprocessed_eng['X_test']
        y_test = preprocessed_eng['y_test']
    else:
        print("\nSkipping feature engineering (set run_feature_engineering=True to enable)")
    
    # =========================================================================
    # STEP 5: Model Training
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: MODEL TRAINING")
    print("="*80)
    
    if models_to_train is None:
        models_to_train = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']
    
    results = train_multiple_models(
        X_train, y_train, X_test, y_test,
        model_names=models_to_train,
        save_dir=output_dir,
        verbose=True
    )
    
    # =========================================================================
    # STEP 6: Ensembling (Optional)
    # =========================================================================
    if run_ensembling:
        print("\n" + "="*80)
        print("STEP 6: ENSEMBLING")
        print("="*80)
        
        # Select best models for ensembling
        base_models = {}
        for name in ['RandomForest', 'XGBoost', 'LightGBM']:
            if name in results and results[name] is not None:
                base_models[name] = results[name][0]
        
        if len(base_models) >= 2:
            comparison_df = compare_ensembles(
                base_models=base_models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                ensemble_types=['voting', 'stacking', 'weighted'],
                save_dir=output_dir
            )
        else:
            print("\nNot enough models for ensembling (need at least 2)")
    else:
        print("\nSkipping ensembling (set run_ensembling=True to enable)")
    
    # =========================================================================
    # STEP 7: Hyperparameter Tuning (Optional)
    # =========================================================================
    if run_hyperparameter_tuning:
        print("\n" + "="*80)
        print("STEP 7: HYPERPARAMETER TUNING")
        print("="*80)
        
        # Get parameter grids
        param_grids = get_param_grids()
        
        # Tune best performing model (typically XGBoost or LightGBM)
        if 'XGBoost' in results and results['XGBoost'] is not None:
            print("\nTuning XGBoost hyperparameters...")
            from xgboost import XGBClassifier
            
            # Use smaller grid for faster tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2],
                'scale_pos_weight': [results['XGBoost'][0].scale_pos_weight]
            }
            
            best_model, tuning_results = tune_with_random_search(
                model=XGBClassifier(random_state=42, n_jobs=-1, eval_metric='auc'),
                param_distributions=param_grid,
                X_train=X_train,
                y_train=y_train,
                n_iter=10,
                cv=5,
                scoring='roc_auc',
                save_path=os.path.join(output_dir, "xgboost_tuned_model.joblib"),
                verbose=1
            )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    
    print("\nNext steps:")
    print("1. Review model comparison in models/model_comparison.csv")
    print("2. Launch Streamlit app: streamlit run streamlit_app/app.py")
    print("3. Explore EDA visualizations in notebooks/eda_outputs/")
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run the complete credit card fraud detection pipeline"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard.csv",
        help="Path to the dataset CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models and outputs"
    )
    
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis"
    )
    
    parser.add_argument(
        "--skip-feature-engineering",
        action="store_true",
        help="Skip feature engineering"
    )
    
    parser.add_argument(
        "--skip-ensembling",
        action="store_true",
        help="Skip ensembling"
    )
    
    parser.add_argument(
        "--run-hyperparameter-tuning",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    
    parser.add_argument(
        "--resampling-method",
        type=str,
        default="smote",
        choices=['none', 'smote', 'adasyn', 'random_under', 'nearmiss', 'smote_tomek', 'smote_enn'],
        help="Resampling method for handling class imbalance"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of models to train (default: LogisticRegression RandomForest XGBoost LightGBM)"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_complete_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        run_eda=not args.skip_eda,
        run_feature_engineering=not args.skip_feature_engineering,
        run_ensembling=not args.skip_ensembling,
        run_hyperparameter_tuning=args.run_hyperparameter_tuning,
        resampling_method=args.resampling_method,
        models_to_train=args.models
    )
    
    print("\n✅ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()

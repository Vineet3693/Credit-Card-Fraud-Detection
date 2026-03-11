# Credit Card Fraud Detection Pipeline

A comprehensive, modular machine learning pipeline for credit card fraud detection using the Kaggle Credit Card Fraud dataset.

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
├── data/                       # Dataset directory (place creditcard.csv here)
├── src/                        # Source code modules
│   ├── data_loading.py         # Data loading utilities
│   ├── eda.py                  # Exploratory Data Analysis & Visualization
│   ├── preprocessing.py        # Data preprocessing & handling imbalance
│   ├── feature_engineering.py  # Feature creation & selection
│   ├── model_training.py       # Model training & evaluation
│   ├── ensembling.py           # Ensemble methods
│   └── hyperparameter_tuning.py # Hyperparameter optimization
├── models/                     # Trained models directory
├── notebooks/                  # Jupyter notebooks
│   └── eda_outputs/            # EDA visualization outputs
├── streamlit_app/              # Streamlit deployment app
│   └── app.py                  # Main Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) from Kaggle and place `creditcard.csv` in the `data/` directory.

### 3. Run the Complete Pipeline

```bash
# Run individual modules
python src/data_loading.py
python src/eda.py
python src/preprocessing.py
python src/feature_engineering.py
python src/model_training.py
python src/ensembling.py
python src/hyperparameter_tuning.py

# Launch Streamlit app
streamlit run streamlit_app/app.py
```

## 📊 Pipeline Stages

### 1. Data Loading (`src/data_loading.py`)
- Load the credit card fraud dataset
- Display basic statistics and class distribution
- Handle missing values

### 2. EDA & Visualization (`src/eda.py`)
- Class distribution analysis
- Correlation heatmaps
- Feature distributions by class
- Boxplots for outlier detection

### 3. Preprocessing (`src/preprocessing.py`)
- Train/test/validation splitting
- Feature scaling (Standard/Robust)
- Handling class imbalance:
  - SMOTE
  - ADASYN
  - Random Undersampling
  - NearMiss
  - SMOTETomek
  - SMOTEENN

### 4. Feature Engineering (`src/feature_engineering.py`)
- Time-based features (cyclical encoding)
- Amount-based features (log transform, binning)
- Interaction features
- PCA dimensionality reduction
- Feature selection (F-test, Mutual Information, Correlation)

### 5. Model Training (`src/model_training.py`)
Supported models:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- K-Nearest Neighbors
- SVM

Evaluation metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Average Precision
- Matthews Correlation Coefficient

### 6. Ensembling (`src/ensembling.py`)
- Voting Classifier (soft/hard voting)
- Stacking Classifier
- Bagging Ensemble
- Custom Weighted Ensemble

### 7. Hyperparameter Tuning (`src/hyperparameter_tuning.py`)
- Grid Search CV
- Random Search CV
- Optuna (Bayesian Optimization)

### 8. Deployment (`streamlit_app/app.py`)
Interactive web application for real-time fraud prediction.

## 📈 Usage Examples

### Basic Pipeline

```python
from src.data_loading import load_data
from src.preprocessing import preprocess_pipeline
from src.model_training import train_multiple_models

# Load data
df = load_data("data/creditcard.csv")

# Preprocess
preprocessed = preprocess_pipeline(
    df,
    test_size=0.2,
    scale_method='standard',
    resampling_method='smote'
)

# Train models
results = train_multiple_models(
    preprocessed['X_train'],
    preprocessed['y_train'],
    preprocessed['X_test'],
    preprocessed['y_test'],
    model_names=['XGBoost', 'LightGBM', 'RandomForest'],
    save_dir='models'
)
```

### With Feature Engineering

```python
from src.feature_engineering import engineer_features_pipeline

# Create new features
df_engineered, metadata = engineer_features_pipeline(
    df,
    add_time_features=True,
    add_amount_features=True,
    add_interactions=False,
    apply_pca_transform=False
)
```

### Hyperparameter Optimization

```python
from src.hyperparameter_tuning import optimize_with_optuna

best_params, best_model = optimize_with_optuna(
    model_name='XGBoost',
    X_train=X_train,
    y_train=y_train,
    n_trials=100,
    cv=5,
    save_path='models/xgboost_optimized.joblib'
)
```

## ⚙️ Configuration Options

### Preprocessing
```python
preprocess_pipeline(
    df,
    test_size=0.2,           # Test set proportion
    val_size=0.0,            # Validation set proportion
    scale_method='standard', # 'standard' or 'robust'
    resampling_method='smote', # Resampling technique
    random_state=42
)
```

### Resampling Methods
- `'none'` - No resampling
- `'smote'` - SMOTE oversampling
- `'adasyn'` - ADASYN oversampling
- `'random_under'` - Random undersampling
- `'nearmiss'` - NearMiss undersampling
- `'smote_tomek'` - SMOTE + Tomek links
- `'smote_enn'` - SMOTE + Edited Nearest Neighbors

## 🎯 Model Performance Tips

1. **Class Imbalance**: Use SMOTE or class_weight='balanced'
2. **Feature Scaling**: StandardScaler works well for most models
3. **Best Models**: XGBoost and LightGBM typically perform best
4. **Ensembling**: Stacking often improves ROC-AUC scores
5. **Threshold Tuning**: Adjust classification threshold based on business needs

## 🌐 Running the Streamlit App

```bash
# Make sure you have trained models in the models/ directory
streamlit run streamlit_app/app.py
```

The app will:
- Allow model selection from trained models
- Provide interactive input for transaction features
- Display real-time fraud predictions with confidence scores
- Show detailed model information

## 📝 Requirements

See `requirements.txt` for all dependencies:

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0
imbalanced-learn>=0.10.0
optuna>=3.0.0
joblib>=1.2.0
streamlit>=1.18.0
```

## 📄 License

This project is for educational purposes. The dataset is subject to Kaggle's terms of service.

## 🔗 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Optuna Documentation](https://optuna.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

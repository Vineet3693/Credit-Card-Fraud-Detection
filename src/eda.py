"""
Exploratory Data Analysis (EDA) & Visualization Module

This module provides functions for exploratory data analysis and visualization
of the credit card fraud dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


def plot_class_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of the target variable (Class).
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Count plot
    ax = sns.countplot(x='Class', data=df, palette='Set2')
    
    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                   (p.get_x() + p.get_width() / 2., height),
                   ha='center', va='bottom', fontsize=12)
    
    plt.title('Class Distribution (0: Normal, 1: Fraud)', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    fraud_count = df['Class'].sum()
    total_count = len(df)
    fraud_percentage = (fraud_count / total_count) * 100
    
    print(f"\nClass Distribution Statistics:")
    print(f"Total transactions: {total_count:,}")
    print(f"Normal transactions: {(total_count - fraud_count):,} ({100-fraud_percentage:.4f}%)")
    print(f"Fraudulent transactions: {fraud_count:,} ({fraud_percentage:.4f}%)")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot correlation heatmap of numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    save_path : str, optional
        Path to save the figure
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(20, 16))
    correlation_matrix = numeric_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                cmap='coolwarm', 
                center=0,
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot=False)  # Set to True for smaller datasets
    
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print correlations with target
    print("\nTop 10 correlations with Class:")
    corr_with_target = correlation_matrix['Class'].drop('Class').abs().sort_values(ascending=False)
    print(corr_with_target.head(10))


def plot_feature_distributions(df: pd.DataFrame, 
                               features: Optional[list] = None,
                               n_features: int = 10,
                               save_path: Optional[str] = None) -> None:
    """
    Plot distributions of selected features by class.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    features : list, optional
        List of features to plot. If None, selects top correlated features.
    n_features : int
        Number of features to plot if features is None
    save_path : str, optional
        Path to save the figure
    """
    if features is None:
        # Select top correlated features with Class
        corr_with_target = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
        features = corr_with_target.head(n_features).index.tolist()
    
    n_plots = len(features)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, feature in enumerate(features):
        if idx < n_plots:
            ax = axes[idx]
            
            # Plot distributions for each class
            for class_val in [0, 1]:
                subset = df[df['Class'] == class_val][feature]
                label = 'Fraud' if class_val == 1 else 'Normal'
                color = 'red' if class_val == 1 else 'blue'
                alpha = 0.7 if class_val == 1 else 0.5
                
                subset.plot(kind='density', ax=ax, label=label, color=color, alpha=alpha)
            
            ax.set_title(f'Distribution of {feature}', fontsize=10)
            ax.set_xlabel('')
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_boxplots_by_class(df: pd.DataFrame, 
                           features: Optional[list] = None,
                           n_features: int = 10,
                           save_path: Optional[str] = None) -> None:
    """
    Plot boxplots of selected features by class.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    features : list, optional
        List of features to plot
    n_features : int
        Number of features to plot if features is None
    save_path : str, optional
        Path to save the figure
    """
    if features is None:
        corr_with_target = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
        features = corr_with_target.head(n_features).index.tolist()
    
    n_plots = len(features)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, feature in enumerate(features):
        if idx < n_plots:
            ax = axes[idx]
            
            df.boxplot(column=feature, by='Class', ax=ax, patch_artist=True)
            ax.set_title(f'{feature}')
            ax.set_xlabel('Class')
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Boxplots by Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_full_eda(df: pd.DataFrame, output_dir: Optional[str] = "notebooks/eda_outputs") -> None:
    """
    Run complete EDA pipeline and save visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    output_dir : str, optional
        Directory to save output figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("RUNNING COMPLETE EDA PIPELINE")
    print("="*60)
    
    # 1. Class Distribution
    print("\n[1/4] Plotting class distribution...")
    plot_class_distribution(df, save_path=f"{output_dir}/class_distribution.png")
    
    # 2. Correlation Heatmap
    print("\n[2/4] Plotting correlation heatmap...")
    plot_correlation_heatmap(df, save_path=f"{output_dir}/correlation_heatmap.png")
    
    # 3. Feature Distributions
    print("\n[3/4] Plotting feature distributions...")
    plot_feature_distributions(df, save_path=f"{output_dir}/feature_distributions.png")
    
    # 4. Boxplots
    print("\n[4/4] Plotting boxplots...")
    plot_boxplots_by_class(df, save_path=f"{output_dir}/boxplots.png")
    
    print("\n" + "="*60)
    print("EDA COMPLETE! Figures saved to:", output_dir)
    print("="*60)


if __name__ == "__main__":
    # Example usage
    from data_loading import load_data
    
    df = load_data()
    run_full_eda(df)

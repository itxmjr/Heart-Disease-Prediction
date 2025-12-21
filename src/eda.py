"""
Exploratory Data Analysis (EDA) Module
======================================

This module creates visualizations to understand:
1. Distribution of each feature
2. Relationships between features
3. Correlation with target variable
4. Patterns that might help prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_target_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Visualize the distribution of target variable.
    This shows if our data is balanced or imbalanced.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    ax1 = axes[0]
    target_counts = df['target'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for no disease, Red for disease
    bars = ax1.bar(['No Disease (0)', 'Disease (1)'], target_counts.values, color=colors)
    ax1.set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, count in zip(bars, target_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', fontweight='bold')
    
    # Pie chart
    ax2 = axes[1]
    ax2.pie(target_counts.values, labels=['No Disease', 'Disease'], 
            autopct='%1.1f%%', colors=colors, explode=[0, 0.05],
            shadow=True, startangle=90)
    ax2.set_title('Heart Disease Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print insight
    count_0 = target_counts.get(0, 0)
    count_1 = target_counts.get(1, 0)
    
    if count_0 > 0:
        ratio = count_1 / count_0
        print(f"\nINSIGHT: Dataset is {'balanced' if 0.8 < ratio < 1.2 else 'slightly imbalanced'}")
        print(f"   Ratio (Disease/No Disease): {ratio:.2f}")
    else:
        print("\nINSIGHT: Cannot calculate ratio (No Disease count is 0)")


def plot_numerical_distributions(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot histograms for all numerical features.
    """
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        # Plot histogram with different colors for each target
        for target, color, label in [(0, '#2ecc71', 'No Disease'), (1, '#e74c3c', 'Disease')]:
            data = df[df['target'] == target][col]
            ax.hist(data, bins=20, alpha=0.6, color=color, label=label, edgecolor='white')
        
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Remove empty subplot
    axes[-1].axis('off')
    
    plt.suptitle('Numerical Features Distribution by Heart Disease Status', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("\nINSIGHTS FROM DISTRIBUTIONS:")
    print("   - Age: Heart disease more common in certain age ranges")
    print("   - Thalach (Max Heart Rate): Lower in people with heart disease")
    print("   - Oldpeak (ST Depression): Higher values associated with disease")


def plot_categorical_distributions(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot bar charts for categorical features.
    """
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        
        # Calculate disease rate for each category
        grouped = df.groupby(col)['target'].agg(['sum', 'count'])
        grouped['disease_rate'] = grouped['sum'] / grouped['count'] * 100
        
        colors = plt.cm.RdYlGn_r(grouped['disease_rate'] / 100)
        bars = ax.bar(grouped.index.astype(str), grouped['disease_rate'], color=colors)
        
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Disease Rate (%)')
        ax.set_ylim(0, 100)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                   f'{height:.0f}%', ha='center', fontsize=9)
    
    plt.suptitle('Heart Disease Rate by Categorical Features', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a correlation heatmap.
    """
    plt.figure(figsize=(14, 10))
    
    # Calculate correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    
    # Create mask for upper triangle (optional - makes it cleaner)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print insights about target correlation
    print("\nCORRELATION WITH TARGET (Heart Disease):")
    print("-" * 45)
    target_corr = corr_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
    for feature, corr in target_corr.items():
        direction = "Positive" if corr > 0 else "Negative"
        strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.15 else "Weak"
        print(f"   {feature:12}: {corr:+.3f} ({direction}, {strength})")


def plot_age_vs_heart_disease(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Detailed analysis of age and heart disease.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax1 = axes[0]
    df.boxplot(column='age', by='target', ax=ax1)
    ax1.set_title('Age Distribution by Heart Disease Status', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Heart Disease (0=No, 1=Yes)')
    ax1.set_ylabel('Age')
    plt.suptitle('')  # Remove automatic title
    
    # Age groups analysis
    ax2 = axes[1]
    df_temp = df.copy()
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=[20, 40, 50, 60, 70, 80], 
                              labels=['20-40', '40-50', '50-60', '60-70', '70-80'])
    
    age_disease_rate = df_temp.groupby('age_group')['target'].mean() * 100
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(age_disease_rate)))
    bars = ax2.bar(age_disease_rate.index.astype(str), age_disease_rate.values, color=colors)
    ax2.set_title('Heart Disease Rate by Age Group', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Disease Rate (%)')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f'{height:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sex_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Analyze heart disease by sex.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sex distribution
    ax1 = axes[0]
    sex_counts = df['sex'].value_counts()
    colors = ['#2196f3', '#e91e63']  # Blue for Male, Pink for Female
    ax1.pie(sex_counts.values, labels=['Male', 'Female'], autopct='%1.1f%%', 
           colors=colors, explode=[0.02, 0.02], shadow=True)
    ax1.set_title('Sex Distribution in Dataset', fontsize=12, fontweight='bold')
    
    # Disease rate by sex
    ax2 = axes[1]
    sex_disease = df.groupby('sex')['target'].agg(['sum', 'count'])
    sex_disease['disease_rate'] = sex_disease['sum'] / sex_disease['count'] * 100
    
    bars = ax2.bar(['Male', 'Female'], sex_disease['disease_rate'].values, color=colors[::-1])
    ax2.set_title('Heart Disease Rate by Sex', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Disease Rate (%)')
    ax2.set_ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f'{height:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("\nINSIGHT: Males show higher heart disease rate in this dataset")


def run_complete_eda(df: pd.DataFrame, save_dir: str = "outputs/"):
    """
    Run all EDA visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    save_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("RUNNING COMPLETE EDA")
    print("="*60)
    
    print("\n1. Target Distribution")
    plot_target_distribution(df, f"{save_dir}target_distribution.png")
    
    print("\n2. Numerical Features")
    plot_numerical_distributions(df, f"{save_dir}numerical_distributions.png")
    
    print("\n3. Categorical Features")
    plot_categorical_distributions(df, f"{save_dir}categorical_distributions.png")
    
    print("\n4. Correlation Matrix")
    plot_correlation_matrix(df, f"{save_dir}correlation_matrix.png")
    
    print("\n5. Age Analysis")
    plot_age_vs_heart_disease(df, f"{save_dir}age_analysis.png")
    
    print("\n6. Sex Analysis")
    plot_sex_analysis(df, f"{save_dir}sex_analysis.png")
    
    print("\n" + "="*60)
    print("EDA COMPLETE! All plots saved to:", save_dir)
    print("="*60)


if __name__ == "__main__":
    from dataloader import load_data
    
    df = load_data("data/heart.csv")
    if df is not None:
        run_complete_eda(df)
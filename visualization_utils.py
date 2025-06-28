import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import io

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def create_confusion_matrix(y_true, y_pred):
    """
    Create a confusion matrix visualization for classification problems.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_feature_importance_plot(importance_dict, top_n=15):
    """
    Create a feature importance plot.
    """
    # Get top N features
    top_features = list(importance_dict.items())[:top_n]
    features, importance = zip(*top_features)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {len(features)} Feature Importance', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        ax.text(bar.get_width() + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_performance_plots(y_true, y_pred, problem_type):
    """
    Create performance visualization plots based on problem type.
    """
    if problem_type == "regression":
        return create_regression_plots(y_true, y_pred)
    else:
        # For classification, we'll create a prediction distribution plot
        return create_classification_distribution(y_true, y_pred)

def create_regression_plots(y_true, y_pred):
    """
    Create regression performance plots: actual vs predicted and residuals.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Actual vs Predicted plot
    axes[0].scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_classification_distribution(y_true, y_pred):
    """
    Create classification performance visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # True vs Predicted distribution
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    x = np.arange(len(unique_labels))
    width = 0.35
    
    true_counts = [np.sum(y_true == label) for label in unique_labels]
    pred_counts = [np.sum(y_pred == label) for label in unique_labels]
    
    axes[0].bar(x - width/2, true_counts, width, label='True', alpha=0.8, color='blue')
    axes[0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='orange')
    
    axes[0].set_xlabel('Classes', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(unique_labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy by class
    class_accuracy = []
    for label in unique_labels:
        mask = y_true == label
        if np.sum(mask) > 0:
            accuracy = np.sum((y_true == label) & (y_pred == label)) / np.sum(mask)
            class_accuracy.append(accuracy)
        else:
            class_accuracy.append(0)
    
    bars = axes[1].bar(unique_labels, class_accuracy, alpha=0.8, color='green')
    axes[1].set_xlabel('Classes', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy by Class', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracy):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_data_overview_plots(df):
    """
    Create data overview plots for initial data exploration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Missing values heatmap
    if df.isnull().sum().sum() > 0:
        sns.heatmap(df.isnull(), cbar=True, ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                       transform=axes[0, 0].transAxes, fontsize=16)
        axes[0, 0].set_title('Missing Values', fontsize=14, fontweight='bold')
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Data Types Distribution', fontsize=14, fontweight='bold')
    
    # Numeric columns correlation (if any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Correlation Matrix (Numeric Features)', fontsize=14, fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'Not enough numeric columns\nfor correlation', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Dataset shape and info
    info_text = f"""
    Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns
    
    Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    
    Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}
    Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
    
    Missing Values: {df.isnull().sum().sum()}
    """
    
    axes[1, 1].text(0.1, 0.5, info_text, ha='left', va='center', 
                   transform=axes[1, 1].transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    axes[1, 1].set_title('Dataset Overview', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

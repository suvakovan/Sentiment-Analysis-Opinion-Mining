"""
Model Evaluation Module for Sentiment Analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and visually compares classification models."""

    def __init__(self, output_dir='results/plots', dpi=150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }

    def print_classification_report(self, y_true, y_pred, model_name="Model"):
        print(f"\n{'=' * 50}")
        print(f"Classification Report: {model_name}")
        print('=' * 50)
        print(classification_report(y_true, y_pred,
                                    target_names=['Negative', 'Positive'],
                                    zero_division=0))

    def plot_confusion_matrix(self, y_true, y_pred, model_name, show=False):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        filepath = os.path.join(self.output_dir, f'cm_{safe_name}.png')
        plt.savefig(filepath, dpi=self.dpi)
        if show:
            plt.show()
        plt.close()

    def plot_roc_curves(self, y_true, model_probs_dict, show=False):
        plt.figure(figsize=(10, 8))
        for model_name, y_prob in model_probs_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Comparison')
        plt.legend(loc="lower right")
        filepath = os.path.join(self.output_dir, 'roc_comparison.png')
        plt.savefig(filepath, dpi=self.dpi)
        if show:
            plt.show()
        plt.close()

    def plot_metrics_comparison(self, metrics_dict, show=False):
        df = self.generate_comparison_table(metrics_dict)
        ax = df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right')
        plt.xticks(rotation=0)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(filepath, dpi=self.dpi)
        if show:
            plt.show()
        plt.close()
        return df

    def generate_comparison_table(self, metrics_dict):
        return pd.DataFrame(metrics_dict).T

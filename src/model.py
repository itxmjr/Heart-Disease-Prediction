"""
Machine Learning Model Module
==============================

This module handles:
1. Data preprocessing for ML
2. Model training (Logistic Regression, Decision Tree)
3. Model evaluation (Accuracy, ROC-AUC, Confusion Matrix)
4. Feature importance analysis
5. Model saving and loading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_curve, 
    roc_auc_score,
    f1_score
)
import joblib
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseModel:
    """
    A class to handle all ML operations for heart disease prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model class.
        
        Parameters:
        -----------
        random_state : int
            For reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def prepare_data(self, df: pd.DataFrame, 
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning.
        Splits features (X) and target (y), scales features, and creates train/test sets.
        """
        print("\nPREPARING DATA FOR ML")
        print("-" * 40)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Save feature names for later
        self.feature_names = X.columns.tolist()
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        print(f"   Features: {len(self.feature_names)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("   Features scaled using StandardScaler")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_logistic_regression(self) -> Dict[str, Any]:
        """
        Train a Logistic Regression model.
        """
        print("\nTRAINING LOGISTIC REGRESSION")
        print("-" * 40)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        # Train the model
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.models['Logistic Regression'] = results
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def train_decision_tree(self) -> Dict[str, Any]:
        """
        Train a Decision Tree model.
        """
        print("\nTRAINING DECISION TREE")
        print("-" * 40)
        
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state
        )
        
        # Decision trees don't need scaling, but we use scaled for consistency
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.models['Decision Tree'] = results
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def train_random_forest(self) -> Dict[str, Any]:
        """
        Train a Random Forest model.
        """
        print("\nTRAINING RANDOM FOREST")
        print("-" * 40)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.models['Random Forest'] = results
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        """
        print("\nMODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for name, results in self.models.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'ROC-AUC': results['roc_auc'],
                'F1 Score': results['f1_score'],
                'CV Mean': results['cv_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Select best model
        best_idx = comparison_df['ROC-AUC'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        
        return comparison_df
    
    def plot_confusion_matrix(self, model_name: str = None, save_path: str = None):
        """
        Plot confusion matrix for a model.
        """
        if model_name is None:
            model_name = self.best_model_name
            
        results = self.models[model_name]
        
        cm = confusion_matrix(self.y_test, results['y_pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Add annotations
        tn, fp, fn, tp = cm.ravel()
        plt.figtext(0.02, 0.02, 
                   f"TN={tn} | FP={fp} | FN={fn} | TP={tp}\n"
                   f"Precision={tp/(tp+fp):.2f} | Recall={tp/(tp+fn):.2f}", 
                   fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        print("\nCONFUSION MATRIX INTERPRETATION:")
        print(f"   True Negatives (correctly predicted no disease): {tn}")
        print(f"   False Positives (incorrectly predicted disease): {fp}")
        print(f"   False Negatives (missed disease cases): {fn}")
        print(f"   True Positives (correctly predicted disease): {tp}")
    
    def plot_roc_curves(self, save_path: str = None):
        """
        Plot ROC curves for all models.
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for idx, (name, results) in enumerate(self.models.items()):
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            auc = results['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f'{name} (AUC = {auc:.3f})')
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        print("\nROC CURVE INTERPRETATION:")
        print("   - Closer to top-left corner = Better model")
        print("   - AUC > 0.9 = Excellent")
        print("   - AUC > 0.8 = Good")
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance from the model.
        """
        if model_name is None:
            model_name = self.best_model_name
            
        model = self.models[model_name]['model']
        
        if model_name == 'Logistic Regression':
            # For logistic regression, use absolute coefficients
            importance = np.abs(model.coef_[0])
        else:
            # For tree-based models
            importance = model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance(self, model_name: str = None, save_path: str = None):
        """
        Visualize feature importance.
        """
        if model_name is None:
            model_name = self.best_model_name
            
        importance_df = self.get_feature_importance(model_name)
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
        
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add value labels
        for bar, val in zip(bars, importance_df['Importance']):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        print("\nTOP 5 MOST IMPORTANT FEATURES:")
        for idx, row in importance_df.head().iterrows():
            print(f"   {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    def print_classification_report(self, model_name: str = None):
        """
        Print detailed classification report.
        """
        if model_name is None:
            model_name = self.best_model_name
            
        results = self.models[model_name]
        
        print(f"\nCLASSIFICATION REPORT - {model_name}")
        print("="*50)
        print(classification_report(self.y_test, results['y_pred'], 
                                   target_names=['No Disease', 'Disease']))
    
    def save_model(self, filepath: str = 'models/heart_model.pkl'):
        """
        Save the best model and scaler.
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str = 'models/heart_model.pkl'):
        """
        Load a saved model.
        """
        model_data = joblib.load(filepath)
        print(f"Model loaded: {model_data['model_name']}")
        return model_data
    
    def predict_single(self, input_data: Dict[str, float]) -> Tuple[int, float]:
        """
        Make prediction for a single patient.
        
        Parameters:
        -----------
        input_data : dict
            Dictionary with feature names as keys
            
        Returns:
        --------
        prediction : int (0 or 1)
        probability : float (0.0 to 1.0)
        """
        # Create dataframe from input
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in right order
        input_df = input_df[self.feature_names]
        
        # Scale the input
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        prediction = self.best_model.predict(input_scaled)[0]
        probability = self.best_model.predict_proba(input_scaled)[0][1]
        
        return int(prediction), float(probability)


def run_complete_ml_pipeline(df: pd.DataFrame, save_dir: str = 'outputs/'):
    """
    Run the complete machine learning pipeline.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("RUNNING COMPLETE ML PIPELINE")
    print("="*60)
    
    # Initialize model
    model = HeartDiseaseModel(random_state=42)
    
    # Prepare data
    model.prepare_data(df)
    
    # Train models
    model.train_logistic_regression()
    model.train_decision_tree()
    model.train_random_forest()
    
    # Compare models
    comparison = model.compare_models()
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    model.plot_confusion_matrix(save_path=f'{save_dir}confusion_matrix.png')
    model.plot_roc_curves(save_path=f'{save_dir}roc_curves.png')
    model.plot_feature_importance(save_path=f'{save_dir}feature_importance.png')
    
    # Classification report
    model.print_classification_report()
    
    # Save model
    model.save_model('models/heart_model.pkl')
    
    print("\n" + "="*60)
    print("ML PIPELINE COMPLETE!")
    print("="*60)
    
    return model


if __name__ == "__main__":
    from .dataloader import load_data
    
    df = load_data("data/heart.csv")
    if df is not None:
        model = run_complete_ml_pipeline(df)
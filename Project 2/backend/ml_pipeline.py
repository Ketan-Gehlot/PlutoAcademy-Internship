import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

class MLPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }
        self.trained_models = {}
        self.evaluation_results = {}
        self.best_model_name = None
        self.dropped_features = []
        self.feature_names = []

    def load_and_preprocess(self):
        # Step 1: Load, Explore & Preprocess
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Check missing values
        missing_values = self.df.isnull().sum().to_dict()
        
        # We assume the target column is 'target' for heart disease dataset
        if 'target' not in self.df.columns:
            raise ValueError("Target column 'target' not found in dataset.")
            
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        self.feature_names = X.columns.tolist()
        
        # Split 80/20
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features (important for KNN and Logistic Regression)
        self.X_train_scaled = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test_scaled = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)
        
        return {
            "dataset_shape": self.df.shape,
            "missing_values": missing_values,
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "features": self.feature_names,
            "preprocessing_explanation": "Checked for missing values (none found typically in this clean dataset). Split data into 80% training and 20% testing sets. Applied StandardScaler to normalize features, which is crucial for distance-based algorithms like KNN and helps Logistic Regression converge faster."
        }

    def feature_engineering(self):
        # Step 2: Feature Engineering
        if self.df is None:
            self.load_and_preprocess()
            
        # Calculate correlation with target
        correlation = self.df.corr()['target'].drop('target')
        
        # Train a quick Random Forest to get feature importance
        rf = RandomForestClassifier(random_state=42)
        rf.fit(self.X_train_scaled, self.y_train)
        importance = rf.feature_importances_
        
        feature_analysis = []
        for i, col in enumerate(self.X_train_scaled.columns):
            feature_analysis.append({
                "feature": col,
                "correlation": float(correlation[col]),
                "importance": float(importance[i])
            })
            
        # Identify least important features to drop (e.g., bottom 2 by importance)
        sorted_by_importance = sorted(feature_analysis, key=lambda x: x['importance'])
        features_to_drop = [f['feature'] for f in sorted_by_importance[:2]]
        self.dropped_features = features_to_drop
        
        # Drop features from datasets
        self.X_train_scaled = self.X_train_scaled.drop(columns=features_to_drop)
        self.X_test_scaled = self.X_test_scaled.drop(columns=features_to_drop)
        self.feature_names = self.X_train_scaled.columns.tolist()
        
        return {
            "feature_analysis": feature_analysis,
            "dropped_features": features_to_drop,
            "explanation": f"Calculated Pearson correlation with the target and trained a baseline Random Forest to extract feature importance. Dropped the least important features ({', '.join(features_to_drop)}) to reduce noise and simplify the model without sacrificing much predictive power."
        }

    def train_models(self):
        # Step 3: Train 3 Different Models
        if self.X_train_scaled is None:
            self.feature_engineering()
            
        for name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)
            self.trained_models[name] = model
            
        return {
            "trained_models": list(self.models.keys()),
            "explanation": "Trained three diverse classification models: Logistic Regression (linear baseline), Random Forest (ensemble of decision trees, robust to non-linearities), and K-Nearest Neighbors (distance-based instance learner). All models were trained on the engineered and scaled training dataset."
        }

    def evaluate_models(self):
        # Step 4: Evaluate & Compare All Models
        if not self.trained_models:
            self.train_models()
            
        for name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test_scaled)
            
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            self.evaluation_results[name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            }
            
        return {
            "metrics": self.evaluation_results,
            "explanation": "Evaluated each model on the unseen test set using Accuracy (overall correctness), Precision (exactness of positive predictions), Recall (completeness of positive predictions), and F1-Score (harmonic mean of Precision and Recall)."
        }

    def get_best_model_analysis(self):
        # Step 5: Best Model Analysis & Conclusion
        if not self.evaluation_results:
            self.evaluate_models()
            
        # Determine best model based on F1-Score
        best_model_name = max(self.evaluation_results, key=lambda k: self.evaluation_results[k]['f1_score'])
        self.best_model_name = best_model_name
        
        best_model = self.trained_models[best_model_name]
        y_pred = best_model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred)
        
        return {
            "best_model": best_model_name,
            "metrics": self.evaluation_results[best_model_name],
            "confusion_matrix": cm.tolist(),
            "classes": ["No Disease (0)", "Disease (1)"],
            "conclusion": f"The {best_model_name} model performed best overall, achieving the highest F1-Score. This indicates it maintained the best balance between correctly identifying patients with heart disease (Recall) while minimizing false alarms (Precision). The confusion matrix reveals exactly how many true positives and true negatives it successfully captured versus its errors. This rigorous pipeline demonstrates that selecting the right algorithm and tuning features significantly impacts diagnostic accuracy."
        }

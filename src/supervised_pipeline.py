"""
Supervised Learning Pipeline for Pump Fault Classification

This module implements a complete supervised learning workflow:
1. Data loading and preprocessing
2. Feature extraction
3. Model training (Random Forest, SVM, XGBoost)
4. Evaluation and validation
5. Model deployment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_recall_fscore_support, cohen_kappa_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from feature_extraction import (
    VibrationFeatureExtractor, preprocess_signal,
    segment_signal, extract_features_from_multiaxis
)


class PumpFaultClassifier:
    """
    Complete supervised learning pipeline for pump fault classification.
    """

    def __init__(self, model_type='random_forest', fs=10240):
        """
        Initialize classifier.

        Parameters:
        -----------
        model_type : str
            Type of model: 'random_forest', 'svm', 'xgboost'
        fs : float
            Sampling frequency in Hz
        """
        self.model_type = model_type
        self.fs = fs
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = None

    def create_model(self, **kwargs):
        """
        Create machine learning model based on type.

        Parameters:
        -----------
        **kwargs : dict
            Model-specific hyperparameters
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'svm':
            self.model = SVC(
                C=kwargs.get('C', 10),
                gamma=kwargs.get('gamma', 'scale'),
                kernel=kwargs.get('kernel', 'rbf'),
                class_weight='balanced',
                probability=True,
                random_state=42
            )

        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=kwargs.get('n_estimators', 200),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    subsample=kwargs.get('subsample', 0.8),
                    colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
            except ImportError:
                print("XGBoost not installed. Using GradientBoosting instead.")
                self.model = GradientBoostingClassifier(
                    n_estimators=kwargs.get('n_estimators', 200),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    subsample=kwargs.get('subsample', 0.8),
                    random_state=42
                )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return self.model

    def extract_features_from_data(self, signals_dict, labels, operating_conditions=None):
        """
        Extract features from multiple recordings.

        Parameters:
        -----------
        signals_dict : dict
            Dictionary: {recording_id: {'X': signal_x, 'Y': signal_y, 'Z': signal_z}}
        labels : dict
            Dictionary: {recording_id: 'fault_label'}
        operating_conditions : dict, optional
            Dictionary: {recording_id: {'speed': rpm, 'pressure': bar}}

        Returns:
        --------
        pd.DataFrame
            Features with labels
        """
        feature_list = []
        label_list = []

        extractor = VibrationFeatureExtractor(fs=self.fs)

        for rec_id, signals in signals_dict.items():
            # Preprocess each axis
            signal_x = preprocess_signal(signals['X'], fs=self.fs)
            signal_y = preprocess_signal(signals['Y'], fs=self.fs)
            signal_z = preprocess_signal(signals['Z'], fs=self.fs)

            # Extract features from all axes
            features = extract_features_from_multiaxis(
                [signal_x, signal_y, signal_z], fs=self.fs
            )

            # Add operating conditions if available
            if operating_conditions and rec_id in operating_conditions:
                features['speed'] = operating_conditions[rec_id].get('speed', 0)
                features['pressure'] = operating_conditions[rec_id].get('pressure', 0)
                features['displacement'] = operating_conditions[rec_id].get('displacement', 0)

            feature_list.append(features)
            label_list.append(labels[rec_id])

        # Create DataFrame
        df = pd.DataFrame(feature_list)
        df['label'] = label_list

        return df

    def prepare_data(self, df, feature_selection='none', n_features=50):
        """
        Prepare data for training: split, scale, and select features.

        Parameters:
        -----------
        df : pd.DataFrame
            Features with 'label' column
        feature_selection : str
            Method: 'none', 'kbest', 'rfe'
        n_features : int
            Number of features to select

        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Train-test split (stratified to preserve class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Feature selection
        if feature_selection == 'kbest':
            self.feature_selector = SelectKBest(f_classif, k=min(n_features, X_train.shape[1]))
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)

            # Update feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]

        elif feature_selection == 'rfe':
            # Recursive Feature Elimination (requires model)
            if self.model is None:
                self.create_model()
            self.feature_selector = RFE(
                estimator=self.model, n_features_to_select=min(n_features, X_train.shape[1])
            )
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)

            # Update feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train, cv=5):
        """
        Train the model with cross-validation.

        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        cv : int
            Number of cross-validation folds

        Returns:
        --------
        float
            Mean cross-validation score
        """
        if self.model is None:
            self.create_model()

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv, scoring='f1_weighted'
        )
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train on full training set
        self.model.fit(X_train, y_train)

        return cv_scores.mean()

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate model on test set.

        Parameters:
        -----------
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        verbose : bool
            Print detailed results

        Returns:
        --------
        dict
            Evaluation metrics
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)

        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )

        results = {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'cohen_kappa': kappa,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1_per_class,
            'support': support
        }

        if verbose:
            print("\n" + "="*60)
            print("MODEL EVALUATION RESULTS")
            print("="*60)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score (weighted): {f1:.4f}")
            print(f"Cohen's Kappa: {kappa:.4f}")

            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_
            ))

            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)

        return results

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance (for tree-based models).

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame
            Feature importance sorted
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"\nTop {top_n} Most Important Features:")
            print(feature_imp.head(top_n))

            return feature_imp
        else:
            print("Model does not support feature importance.")
            return None

    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        param_grid : dict
            Parameter grid for search
        cv : int
            Cross-validation folds

        Returns:
        --------
        dict
            Best parameters
        """
        if param_grid is None:
            # Default parameter grids
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                }
            elif self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3]
                }

        if self.model is None:
            self.create_model()

        print(f"Performing hyperparameter tuning with {cv}-fold CV...")
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='f1_weighted',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        return grid_search.best_params_

    def save_model(self, filepath):
        """Save trained model and preprocessing objects."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'fs': self.fs
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model and preprocessing objects."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.fs = model_data['fs']
        print(f"Model loaded from {filepath}")

    def predict_online(self, signal_window_xyz):
        """
        Predict fault class for online monitoring.

        Parameters:
        -----------
        signal_window_xyz : list of np.array
            [signal_x, signal_y, signal_z] for one time window

        Returns:
        --------
        dict
            {'class': predicted_class, 'probability': confidence, 'probabilities': all_class_probs}
        """
        # Preprocess
        signal_x = preprocess_signal(signal_window_xyz[0], fs=self.fs)
        signal_y = preprocess_signal(signal_window_xyz[1], fs=self.fs)
        signal_z = preprocess_signal(signal_window_xyz[2], fs=self.fs)

        # Extract features
        features = extract_features_from_multiaxis(
            [signal_x, signal_y, signal_z], fs=self.fs
        )

        # Convert to DataFrame (to maintain feature order)
        X = pd.DataFrame([features])[self.feature_names]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Feature selection
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)

        # Predict
        y_pred = self.model.predict(X_scaled)[0]
        y_pred_proba = self.model.predict_proba(X_scaled)[0]

        result = {
            'class': self.label_encoder.inverse_transform([y_pred])[0],
            'probability': np.max(y_pred_proba),
            'probabilities': dict(zip(
                self.label_encoder.classes_,
                y_pred_proba
            ))
        }

        return result


# Example usage and demonstration
if __name__ == "__main__":
    print("="*70)
    print("SUPERVISED LEARNING PIPELINE FOR PUMP FAULT CLASSIFICATION")
    print("="*70)

    # Generate synthetic data for demonstration
    np.random.seed(42)
    fs = 10240
    duration = 2.56

    # Simulate different fault classes
    fault_classes = ['Normal', 'Bearing Fault', 'Misalignment', 'Cavitation']
    n_samples_per_class = 25

    signals_dict = {}
    labels = {}
    sample_id = 0

    for fault_class in fault_classes:
        for i in range(n_samples_per_class):
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            # Generate synthetic signals with different characteristics
            if fault_class == 'Normal':
                # Low vibration, mostly 1x rotation
                signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.random.randn(len(t))
                signal_y = 0.1 * np.sin(2 * np.pi * 50 * t + np.pi/4) + 0.02 * np.random.randn(len(t))
                signal_z = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))

            elif fault_class == 'Bearing Fault':
                # High-frequency impulses + harmonics
                signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                          0.3 * np.sin(2 * np.pi * 1234 * t) + \
                          0.1 * np.random.randn(len(t))
                signal_y = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                          0.25 * np.sin(2 * np.pi * 1234 * t) + \
                          0.1 * np.random.randn(len(t))
                signal_z = 0.15 * np.sin(2 * np.pi * 1234 * t) + \
                          0.05 * np.random.randn(len(t))

            elif fault_class == 'Misalignment':
                # Strong 2x and 3x harmonics
                signal_x = 0.2 * np.sin(2 * np.pi * 50 * t) + \
                          0.15 * np.sin(2 * np.pi * 100 * t) + \
                          0.1 * np.sin(2 * np.pi * 150 * t) + \
                          0.03 * np.random.randn(len(t))
                signal_y = 0.25 * np.sin(2 * np.pi * 50 * t) + \
                          0.2 * np.sin(2 * np.pi * 100 * t) + \
                          0.03 * np.random.randn(len(t))
                signal_z = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                          0.02 * np.random.randn(len(t))

            elif fault_class == 'Cavitation':
                # Broadband noise + low-frequency components
                signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                          0.2 * np.random.randn(len(t)) + \
                          0.1 * np.sin(2 * np.pi * 5 * t)
                signal_y = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                          0.2 * np.random.randn(len(t)) + \
                          0.1 * np.sin(2 * np.pi * 7 * t)
                signal_z = 0.15 * np.random.randn(len(t))

            signals_dict[sample_id] = {'X': signal_x, 'Y': signal_y, 'Z': signal_z}
            labels[sample_id] = fault_class
            sample_id += 1

    print(f"\nGenerated {len(signals_dict)} synthetic samples")
    print(f"Classes: {fault_classes}")
    print(f"Samples per class: {n_samples_per_class}")

    # Initialize classifier
    print("\n" + "-"*70)
    print("Step 1: Initialize Classifier")
    print("-"*70)
    classifier = PumpFaultClassifier(model_type='random_forest', fs=fs)

    # Extract features
    print("\n" + "-"*70)
    print("Step 2: Extract Features")
    print("-"*70)
    df = classifier.extract_features_from_data(signals_dict, labels)
    print(f"Extracted features shape: {df.shape}")
    print(f"Features: {df.columns.tolist()[:10]}... (showing first 10)")

    # Prepare data
    print("\n" + "-"*70)
    print("Step 3: Prepare Data (Split, Scale, Select Features)")
    print("-"*70)
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        df, feature_selection='kbest', n_features=30
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train model
    print("\n" + "-"*70)
    print("Step 4: Train Model with Cross-Validation")
    print("-"*70)
    classifier.train(X_train, y_train, cv=5)

    # Evaluate
    print("\n" + "-"*70)
    print("Step 5: Evaluate on Test Set")
    print("-"*70)
    results = classifier.evaluate(X_test, y_test)

    # Feature importance
    print("\n" + "-"*70)
    print("Step 6: Feature Importance")
    print("-"*70)
    classifier.get_feature_importance(top_n=15)

    # Save model
    print("\n" + "-"*70)
    print("Step 7: Save Model")
    print("-"*70)
    model_path = Path(__file__).parent.parent / 'models' / 'pump_classifier_rf.pkl'
    model_path.parent.mkdir(exist_ok=True, parents=True)
    classifier.save_model(model_path)

    # Test online prediction
    print("\n" + "-"*70)
    print("Step 8: Test Online Prediction")
    print("-"*70)
    test_signal = signals_dict[0]  # Use first sample
    prediction = classifier.predict_online([
        test_signal['X'], test_signal['Y'], test_signal['Z']
    ])
    print(f"Predicted class: {prediction['class']}")
    print(f"Confidence: {prediction['probability']:.4f}")
    print(f"All probabilities: {prediction['probabilities']}")

    print("\n" + "="*70)
    print("PIPELINE DEMONSTRATION COMPLETE")
    print("="*70)

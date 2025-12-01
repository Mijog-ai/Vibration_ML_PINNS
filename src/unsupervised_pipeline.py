"""
Unsupervised Learning Pipeline for Anomaly Detection

This module implements anomaly detection methods for early fault detection:
1. Reconstruction-based: Autoencoder, LSTM Autoencoder
2. Distance-based: One-Class SVM, Isolation Forest, Local Outlier Factor
3. Statistical: Mahalanobis Distance, Statistical Process Control
4. Ensemble methods combining multiple approaches
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.covariance import EllipticEnvelope
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from feature_extraction import (
    VibrationFeatureExtractor, preprocess_signal,
    segment_signal, extract_features_from_multiaxis
)


class AnomalyDetector:
    """
    Unsupervised anomaly detection for pump fault detection.
    """

    def __init__(self, method='isolation_forest', fs=10240):
        """
        Initialize anomaly detector.

        Parameters:
        -----------
        method : str
            Detection method: 'isolation_forest', 'ocsvm', 'lof', 'autoencoder',
                            'mahalanobis', 'gmm', 'ensemble'
        fs : float
            Sampling frequency in Hz
        """
        self.method = method
        self.fs = fs
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.feature_names = None

        # For Mahalanobis distance
        self.mean = None
        self.cov_inv = None

        # For ensemble
        self.ensemble_models = {}
        self.ensemble_weights = {}

    def create_model(self, **kwargs):
        """
        Create anomaly detection model.

        Parameters:
        -----------
        **kwargs : dict
            Method-specific hyperparameters
        """
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=kwargs.get('contamination', 0.05),
                n_estimators=kwargs.get('n_estimators', 200),
                max_samples=kwargs.get('max_samples', 'auto'),
                random_state=42,
                n_jobs=-1
            )

        elif self.method == 'ocsvm':
            self.model = OneClassSVM(
                nu=kwargs.get('nu', 0.05),  # Expected outlier fraction
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale')
            )

        elif self.method == 'lof':
            self.model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=kwargs.get('contamination', 0.05),
                novelty=True  # For prediction on new data
            )

        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=kwargs.get('n_components', 3),
                covariance_type=kwargs.get('covariance_type', 'full'),
                random_state=42
            )

        elif self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=kwargs.get('n_clusters', 5),
                random_state=42,
                n_init=10
            )

        elif self.method == 'autoencoder':
            # Will implement with TensorFlow/Keras if available
            try:
                import tensorflow as tf
                from tensorflow import keras
                self._create_autoencoder(**kwargs)
            except ImportError:
                print("TensorFlow not available. Use other methods.")
                self.method = 'isolation_forest'
                self.create_model(**kwargs)

        elif self.method == 'ensemble':
            # Create multiple models
            self.ensemble_models = {
                'isolation_forest': IsolationForest(contamination=0.05, n_estimators=100, random_state=42),
                'ocsvm': OneClassSVM(nu=0.05, kernel='rbf'),
                'lof': LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
            }
            self.ensemble_weights = {
                'isolation_forest': 0.4,
                'ocsvm': 0.3,
                'lof': 0.3
            }

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.model

    def _create_autoencoder(self, input_dim=None, encoding_dim=16, **kwargs):
        """Create autoencoder architecture."""
        import tensorflow as tf
        from tensorflow import keras

        if input_dim is None:
            input_dim = 100  # Will be updated during training

        # Encoder
        encoder_input = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(128, activation='relu')(encoder_input)
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        encoded = keras.layers.Dense(32, activation='relu')(encoded)
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = keras.layers.Dense(32, activation='relu')(encoded)
        decoded = keras.layers.Dense(64, activation='relu')(decoded)
        decoded = keras.layers.Dense(128, activation='relu')(decoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)

        # Autoencoder model
        self.model = keras.Model(encoder_input, decoded)
        self.model.compile(optimizer='adam', loss='mse')

        # Encoder model (for latent representation)
        self.encoder = keras.Model(encoder_input, encoded)

    def fit(self, X_normal, percentile_threshold=95):
        """
        Train anomaly detector on normal data only.

        Parameters:
        -----------
        X_normal : np.array or pd.DataFrame
            Normal (healthy) data features
        percentile_threshold : float
            Percentile for setting anomaly threshold (95-99)

        Returns:
        --------
        self
        """
        # Convert to DataFrame if needed
        if isinstance(X_normal, pd.DataFrame):
            self.feature_names = X_normal.columns.tolist()
            X_normal = X_normal.values
        else:
            X_normal = np.array(X_normal)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_normal)

        if self.method == 'mahalanobis':
            # Compute mean and covariance
            self.mean = np.mean(X_scaled, axis=0)
            cov = np.cov(X_scaled, rowvar=False)
            # Add regularization to avoid singular matrix
            cov += np.eye(cov.shape[0]) * 1e-6
            self.cov_inv = np.linalg.inv(cov)

            # Compute threshold based on chi-square distribution
            self.threshold = chi2.ppf(percentile_threshold / 100, df=X_scaled.shape[1])

            print(f"Mahalanobis threshold: {self.threshold:.4f}")

        elif self.method == 'ensemble':
            # Train each model in ensemble
            for name, model in self.ensemble_models.items():
                print(f"Training {name}...")
                model.fit(X_scaled)

            # Compute ensemble threshold
            scores = self._compute_ensemble_scores(X_scaled)
            self.threshold = np.percentile(scores, percentile_threshold)

        elif self.method == 'autoencoder':
            # Train autoencoder
            print("Training autoencoder...")
            self.model.fit(
                X_scaled, X_scaled,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_split=0.1,
                verbose=0
            )

            # Compute reconstruction errors on training data
            X_reconstructed = self.model.predict(X_scaled, verbose=0)
            reconstruction_errors = np.mean((X_scaled - X_reconstructed)**2, axis=1)
            self.threshold = np.percentile(reconstruction_errors, percentile_threshold)

            print(f"Autoencoder reconstruction threshold: {self.threshold:.6f}")

        else:
            # Train model
            if self.model is None:
                self.create_model()

            self.model.fit(X_scaled)

            # Compute anomaly scores on training data to set threshold
            scores = self._compute_anomaly_scores(X_scaled)
            self.threshold = np.percentile(scores, percentile_threshold)

            print(f"Anomaly score threshold ({percentile_threshold}th percentile): {self.threshold:.4f}")

        return self

    def _compute_anomaly_scores(self, X_scaled):
        """Compute anomaly scores for given data."""
        if self.method == 'isolation_forest':
            # Isolation Forest: lower score = more anomalous
            scores = -self.model.score_samples(X_scaled)

        elif self.method in ['ocsvm', 'lof']:
            # Decision function: negative = anomaly
            scores = -self.model.decision_function(X_scaled)

        elif self.method == 'gmm':
            # Negative log-likelihood
            scores = -self.model.score_samples(X_scaled)

        elif self.method == 'kmeans':
            # Distance to nearest cluster center
            distances = self.model.transform(X_scaled)
            scores = np.min(distances, axis=1)

        elif self.method == 'mahalanobis':
            # Mahalanobis distance
            scores = np.array([
                mahalanobis(x, self.mean, self.cov_inv) for x in X_scaled
            ])

        elif self.method == 'autoencoder':
            # Reconstruction error
            X_reconstructed = self.model.predict(X_scaled, verbose=0)
            scores = np.mean((X_scaled - X_reconstructed)**2, axis=1)

        else:
            scores = np.zeros(X_scaled.shape[0])

        return scores

    def _compute_ensemble_scores(self, X_scaled):
        """Compute weighted ensemble anomaly scores."""
        ensemble_scores = np.zeros(X_scaled.shape[0])

        for name, model in self.ensemble_models.items():
            # Get scores from each model
            if name == 'isolation_forest':
                scores = -model.score_samples(X_scaled)
            elif name in ['ocsvm', 'lof']:
                scores = -model.decision_function(X_scaled)
            else:
                scores = np.zeros(X_scaled.shape[0])

            # Normalize scores to [0, 1]
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

            # Add to ensemble
            ensemble_scores += self.ensemble_weights[name] * scores_norm

        return ensemble_scores

    def predict(self, X, return_scores=False):
        """
        Predict anomalies on new data.

        Parameters:
        -----------
        X : np.array or pd.DataFrame
            Feature matrix
        return_scores : bool
            If True, return anomaly scores as well

        Returns:
        --------
        predictions : np.array
            1 = normal, -1 = anomaly
        scores : np.array (optional)
            Anomaly scores
        """
        # Convert to array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Compute anomaly scores
        if self.method == 'ensemble':
            scores = self._compute_ensemble_scores(X_scaled)
        else:
            scores = self._compute_anomaly_scores(X_scaled)

        # Threshold to get predictions
        predictions = np.where(scores > self.threshold, -1, 1)  # -1 = anomaly

        if return_scores:
            return predictions, scores
        else:
            return predictions

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate anomaly detector on labeled test set.

        Parameters:
        -----------
        X_test : np.array
            Test features
        y_test : np.array
            Test labels (0 = normal, 1 = anomaly/fault)
        verbose : bool
            Print results

        Returns:
        --------
        dict
            Evaluation metrics
        """
        # Get predictions and scores
        predictions, scores = self.predict(X_test, return_scores=True)

        # Convert predictions: -1 (anomaly) → 1, 1 (normal) → 0
        y_pred = np.where(predictions == -1, 1, 0)

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, scores)
            fpr, tpr, thresholds = roc_curve(y_test, scores)
        except:
            auc = 0.0
            fpr, tpr, thresholds = None, None, None

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }

        if verbose:
            print("\n" + "="*60)
            print("ANOMALY DETECTION EVALUATION")
            print("="*60)
            print(f"Method: {self.method}")
            print(f"Threshold: {self.threshold:.6f}")
            print(f"\nAccuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {auc:.4f}")

            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"              Predicted")
            print(f"              Normal  Anomaly")
            print(f"Actual Normal   {cm[0,0]:5d}   {cm[0,1]:5d}")
            print(f"      Anomaly   {cm[1,0]:5d}   {cm[1,1]:5d}")

        return results

    def detect_online(self, signal_window_xyz, return_score=True):
        """
        Detect anomaly for online monitoring.

        Parameters:
        -----------
        signal_window_xyz : list of np.array
            [signal_x, signal_y, signal_z]
        return_score : bool
            Return anomaly score

        Returns:
        --------
        dict
            {'is_anomaly': bool, 'anomaly_score': float, 'threshold': float}
        """
        # Preprocess
        signal_x = preprocess_signal(signal_window_xyz[0], fs=self.fs)
        signal_y = preprocess_signal(signal_window_xyz[1], fs=self.fs)
        signal_z = preprocess_signal(signal_window_xyz[2], fs=self.fs)

        # Extract features
        features = extract_features_from_multiaxis(
            [signal_x, signal_y, signal_z], fs=self.fs
        )

        # Convert to DataFrame
        if self.feature_names:
            X = pd.DataFrame([features])[self.feature_names]
        else:
            X = pd.DataFrame([features])

        # Predict
        prediction, score = self.predict(X, return_scores=True)

        result = {
            'is_anomaly': prediction[0] == -1,
            'anomaly_score': score[0],
            'threshold': self.threshold,
            'severity': self._compute_severity(score[0])
        }

        return result

    def _compute_severity(self, score):
        """Compute severity level based on anomaly score."""
        if score < self.threshold:
            return 'NORMAL'
        elif score < self.threshold * 1.5:
            return 'CAUTION'
        elif score < self.threshold * 2.0:
            return 'WARNING'
        else:
            return 'CRITICAL'

    def save_model(self, filepath):
        """Save trained model."""
        model_data = {
            'method': self.method,
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'fs': self.fs
        }

        # For Mahalanobis
        if self.method == 'mahalanobis':
            model_data['mean'] = self.mean
            model_data['cov_inv'] = self.cov_inv

        # For ensemble
        if self.method == 'ensemble':
            model_data['ensemble_models'] = self.ensemble_models
            model_data['ensemble_weights'] = self.ensemble_weights

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model."""
        model_data = joblib.load(filepath)
        self.method = model_data['method']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.feature_names = model_data['feature_names']
        self.fs = model_data['fs']

        if 'mean' in model_data:
            self.mean = model_data['mean']
            self.cov_inv = model_data['cov_inv']

        if 'ensemble_models' in model_data:
            self.ensemble_models = model_data['ensemble_models']
            self.ensemble_weights = model_data['ensemble_weights']

        print(f"Model loaded from {filepath}")


class StatisticalProcessControl:
    """
    Simple Statistical Process Control for real-time monitoring.
    """

    def __init__(self, feature_names=None):
        """
        Initialize SPC monitor.

        Parameters:
        -----------
        feature_names : list
            List of features to monitor
        """
        self.feature_names = feature_names
        self.means = {}
        self.stds = {}
        self.ucl = {}  # Upper Control Limit
        self.lcl = {}  # Lower Control Limit

    def fit(self, X_normal, n_sigma=3):
        """
        Compute control limits from normal data.

        Parameters:
        -----------
        X_normal : pd.DataFrame
            Normal operating data
        n_sigma : float
            Number of standard deviations for control limits (default: 3)
        """
        if isinstance(X_normal, pd.DataFrame):
            self.feature_names = X_normal.columns.tolist()
        else:
            X_normal = pd.DataFrame(X_normal)

        for feature in self.feature_names:
            self.means[feature] = X_normal[feature].mean()
            self.stds[feature] = X_normal[feature].std()
            self.ucl[feature] = self.means[feature] + n_sigma * self.stds[feature]
            self.lcl[feature] = self.means[feature] - n_sigma * self.stds[feature]

        print(f"SPC control limits set for {len(self.feature_names)} features")

    def check(self, X):
        """
        Check if any feature violates control limits.

        Parameters:
        -----------
        X : pd.DataFrame or dict
            Feature values (single sample)

        Returns:
        --------
        dict
            {'violations': list, 'is_alarm': bool}
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[0].to_dict()

        violations = []
        for feature in self.feature_names:
            value = X[feature]
            if value > self.ucl[feature] or value < self.lcl[feature]:
                violations.append({
                    'feature': feature,
                    'value': value,
                    'ucl': self.ucl[feature],
                    'lcl': self.lcl[feature]
                })

        return {
            'violations': violations,
            'is_alarm': len(violations) > 0,
            'n_violations': len(violations)
        }


# Example usage and demonstration
if __name__ == "__main__":
    print("="*70)
    print("UNSUPERVISED ANOMALY DETECTION FOR PUMP FAULT DETECTION")
    print("="*70)

    # Generate synthetic data
    np.random.seed(42)
    fs = 10240
    duration = 2.56

    # Normal data (100 samples)
    print("\nGenerating synthetic normal data...")
    normal_signals = []
    for i in range(100):
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.random.randn(len(t))
        signal_y = 0.1 * np.sin(2 * np.pi * 50 * t + np.pi/4) + 0.02 * np.random.randn(len(t))
        signal_z = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))
        normal_signals.append([signal_x, signal_y, signal_z])

    # Extract features from normal data
    extractor = VibrationFeatureExtractor(fs=fs)
    normal_features = []
    for signals in normal_signals:
        signal_x = preprocess_signal(signals[0], fs=fs)
        signal_y = preprocess_signal(signals[1], fs=fs)
        signal_z = preprocess_signal(signals[2], fs=fs)
        features = extract_features_from_multiaxis([signal_x, signal_y, signal_z], fs=fs)
        normal_features.append(features)

    X_normal = pd.DataFrame(normal_features)
    print(f"Normal data shape: {X_normal.shape}")

    # Generate test data (mix of normal and faults)
    print("\nGenerating test data (50 normal, 50 faulty)...")
    test_signals = []
    test_labels = []

    # 50 normal
    for i in range(50):
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.random.randn(len(t))
        signal_y = 0.1 * np.sin(2 * np.pi * 50 * t + np.pi/4) + 0.02 * np.random.randn(len(t))
        signal_z = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))
        test_signals.append([signal_x, signal_y, signal_z])
        test_labels.append(0)  # Normal

    # 50 faulty (bearing fault simulation)
    for i in range(50):
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                  0.3 * np.sin(2 * np.pi * 1234 * t) + \
                  0.1 * np.random.randn(len(t))
        signal_y = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                  0.25 * np.sin(2 * np.pi * 1234 * t) + \
                  0.1 * np.random.randn(len(t))
        signal_z = 0.15 * np.sin(2 * np.pi * 1234 * t) + \
                  0.05 * np.random.randn(len(t))
        test_signals.append([signal_x, signal_y, signal_z])
        test_labels.append(1)  # Faulty

    # Extract features from test data
    test_features = []
    for signals in test_signals:
        signal_x = preprocess_signal(signals[0], fs=fs)
        signal_y = preprocess_signal(signals[1], fs=fs)
        signal_z = preprocess_signal(signals[2], fs=fs)
        features = extract_features_from_multiaxis([signal_x, signal_y, signal_z], fs=fs)
        test_features.append(features)

    X_test = pd.DataFrame(test_features)
    y_test = np.array(test_labels)

    print(f"Test data shape: {X_test.shape}")
    print(f"Normal samples: {sum(y_test == 0)}, Faulty samples: {sum(y_test == 1)}")

    # Test different methods
    methods = ['isolation_forest', 'ocsvm', 'ensemble']

    for method in methods:
        print("\n" + "="*70)
        print(f"METHOD: {method.upper()}")
        print("="*70)

        # Initialize detector
        detector = AnomalyDetector(method=method, fs=fs)

        # Train on normal data
        print(f"\nTraining {method} on normal data...")
        detector.fit(X_normal, percentile_threshold=95)

        # Evaluate on test data
        print(f"\nEvaluating on test data...")
        results = detector.evaluate(X_test, y_test)

        # Save model
        model_path = Path(__file__).parent.parent / 'models' / f'anomaly_detector_{method}.pkl'
        model_path.parent.mkdir(exist_ok=True, parents=True)
        detector.save_model(model_path)

    # Test online detection
    print("\n" + "="*70)
    print("ONLINE DETECTION TEST")
    print("="*70)

    # Load one model for demo
    detector = AnomalyDetector(method='isolation_forest', fs=fs)
    model_path = Path(__file__).parent.parent / 'models' / 'anomaly_detector_isolation_forest.pkl'
    detector.load_model(model_path)

    # Test on a normal sample
    print("\nTest 1: Normal sample")
    result = detector.detect_online(test_signals[0])
    print(f"Is anomaly: {result['is_anomaly']}")
    print(f"Anomaly score: {result['anomaly_score']:.6f}")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Severity: {result['severity']}")

    # Test on a faulty sample
    print("\nTest 2: Faulty sample")
    result = detector.detect_online(test_signals[51])
    print(f"Is anomaly: {result['is_anomaly']}")
    print(f"Anomaly score: {result['anomaly_score']:.6f}")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Severity: {result['severity']}")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

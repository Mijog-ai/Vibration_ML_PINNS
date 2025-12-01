"""
Complete Example: Pump Fault Detection Pipeline

This script demonstrates the complete workflow:
1. Generate synthetic vibration data (replace with your real data)
2. Extract features
3. Train supervised classifier
4. Train unsupervised anomaly detector
5. Evaluate both models
6. Save models for deployment

For production use, replace synthetic data generation with your actual
pump vibration data loading code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from feature_extraction import extract_features_from_multiaxis, preprocess_signal
from supervised_pipeline import PumpFaultClassifier
from unsupervised_pipeline import AnomalyDetector

print("="*80)
print("COMPLETE PUMP FAULT DETECTION PIPELINE DEMONSTRATION")
print("="*80)

# Configuration
FS = 10240  # Sampling frequency in Hz
DURATION = 2.56  # Window duration in seconds
N_SAMPLES_PER_CLASS = 30
FAULT_CLASSES = ['Normal', 'Bearing Fault', 'Misalignment', 'Cavitation']

# ============================================================================
# STEP 1: DATA GENERATION (Replace with your actual data loading)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Generate Synthetic Vibration Data")
print("="*80)
print("NOTE: Replace this section with your actual data loading code")

np.random.seed(42)

def generate_fault_signal(fault_type, duration=DURATION, fs=FS):
    """Generate synthetic vibration signal for different fault types."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if fault_type == 'Normal':
        # Low vibration, mostly 1x rotation
        signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.random.randn(len(t))
        signal_y = 0.1 * np.sin(2 * np.pi * 50 * t + np.pi/4) + 0.02 * np.random.randn(len(t))
        signal_z = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))

    elif fault_type == 'Bearing Fault':
        # High-frequency impulses + harmonics
        signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                  0.3 * np.sin(2 * np.pi * 1234 * t) + \
                  0.1 * np.random.randn(len(t))
        signal_y = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                  0.25 * np.sin(2 * np.pi * 1234 * t) + \
                  0.1 * np.random.randn(len(t))
        signal_z = 0.15 * np.sin(2 * np.pi * 1234 * t) + \
                  0.05 * np.random.randn(len(t))

    elif fault_type == 'Misalignment':
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

    elif fault_type == 'Cavitation':
        # Broadband noise + low-frequency components
        signal_x = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                  0.2 * np.random.randn(len(t)) + \
                  0.1 * np.sin(2 * np.pi * 5 * t)
        signal_y = 0.1 * np.sin(2 * np.pi * 50 * t) + \
                  0.2 * np.random.randn(len(t)) + \
                  0.1 * np.sin(2 * np.pi * 7 * t)
        signal_z = 0.15 * np.random.randn(len(t))

    return {'X': signal_x, 'Y': signal_y, 'Z': signal_z}


# Generate dataset
signals_dict = {}
labels = {}
sample_id = 0

for fault_class in FAULT_CLASSES:
    for i in range(N_SAMPLES_PER_CLASS):
        signals_dict[sample_id] = generate_fault_signal(fault_class)
        labels[sample_id] = fault_class
        sample_id += 1

print(f"Generated {len(signals_dict)} samples")
print(f"Fault classes: {FAULT_CLASSES}")
print(f"Samples per class: {N_SAMPLES_PER_CLASS}")

# ============================================================================
# STEP 2: FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Extract Features from Vibration Signals")
print("="*80)

feature_list = []
label_list = []

for sample_id, signals in signals_dict.items():
    # Preprocess each axis
    signal_x = preprocess_signal(signals['X'], fs=FS)
    signal_y = preprocess_signal(signals['Y'], fs=FS)
    signal_z = preprocess_signal(signals['Z'], fs=FS)

    # Extract features
    features = extract_features_from_multiaxis(
        [signal_x, signal_y, signal_z], fs=FS
    )

    feature_list.append(features)
    label_list.append(labels[sample_id])

# Create DataFrame
df = pd.DataFrame(feature_list)
df['label'] = label_list

print(f"Extracted features shape: {df.shape}")
print(f"Number of features: {df.shape[1] - 1}")  # Excluding label
print(f"Feature names (first 10): {df.columns.tolist()[:10]}")

# ============================================================================
# STEP 3: SUPERVISED LEARNING - FAULT CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Train Supervised Classifier (Random Forest)")
print("="*80)

# Initialize classifier
classifier = PumpFaultClassifier(model_type='random_forest', fs=FS)

# Prepare data
X_train, X_test, y_train, y_test = classifier.prepare_data(
    df.copy(), feature_selection='kbest', n_features=30
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train with cross-validation
print("\nTraining Random Forest classifier...")
cv_score = classifier.train(X_train, y_train, cv=5)

# Evaluate
print("\n" + "-"*80)
print("Evaluation on Test Set:")
print("-"*80)
results_supervised = classifier.evaluate(X_test, y_test)

# Feature importance
print("\n" + "-"*80)
print("Top 10 Most Important Features:")
print("-"*80)
feature_importance = classifier.get_feature_importance(top_n=10)

# Save model
model_dir = Path(__file__).parent.parent / 'models'
model_dir.mkdir(exist_ok=True, parents=True)
classifier.save_model(model_dir / 'pump_classifier_demo.pkl')

# ============================================================================
# STEP 4: UNSUPERVISED LEARNING - ANOMALY DETECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Train Unsupervised Anomaly Detector")
print("="*80)

# Separate normal data for training
df_normal = df[df['label'] == 'Normal'].drop('label', axis=1)
print(f"Normal samples for training: {len(df_normal)}")

# Create test set with mix of normal and faults
df_test = df.copy()
y_test_anomaly = (df_test['label'] != 'Normal').astype(int).values
X_test_anomaly = df_test.drop('label', axis=1)

print(f"Test set: {len(X_test_anomaly)} samples")
print(f"  Normal: {sum(y_test_anomaly == 0)}")
print(f"  Faulty: {sum(y_test_anomaly == 1)}")

# Train Isolation Forest
print("\nTraining Isolation Forest anomaly detector...")
detector_if = AnomalyDetector(method='isolation_forest', fs=FS)
detector_if.fit(df_normal, percentile_threshold=95)

# Evaluate
print("\n" + "-"*80)
print("Evaluation: Isolation Forest")
print("-"*80)
results_if = detector_if.evaluate(X_test_anomaly, y_test_anomaly)

# Train One-Class SVM
print("\nTraining One-Class SVM anomaly detector...")
detector_ocsvm = AnomalyDetector(method='ocsvm', fs=FS)
detector_ocsvm.fit(df_normal, percentile_threshold=95)

# Evaluate
print("\n" + "-"*80)
print("Evaluation: One-Class SVM")
print("-"*80)
results_ocsvm = detector_ocsvm.evaluate(X_test_anomaly, y_test_anomaly)

# Train Ensemble
print("\nTraining Ensemble anomaly detector...")
detector_ensemble = AnomalyDetector(method='ensemble', fs=FS)
detector_ensemble.fit(df_normal, percentile_threshold=95)

# Evaluate
print("\n" + "-"*80)
print("Evaluation: Ensemble")
print("-"*80)
results_ensemble = detector_ensemble.evaluate(X_test_anomaly, y_test_anomaly)

# Save models
detector_if.save_model(model_dir / 'anomaly_detector_if_demo.pkl')
detector_ocsvm.save_model(model_dir / 'anomaly_detector_ocsvm_demo.pkl')
detector_ensemble.save_model(model_dir / 'anomaly_detector_ensemble_demo.pkl')

# ============================================================================
# STEP 5: COMPARISON & SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Results Summary")
print("="*80)

print("\nSUPERVISED CLASSIFICATION (Random Forest):")
print(f"  Accuracy:  {results_supervised['accuracy']:.4f}")
print(f"  F1-Score:  {results_supervised['f1_weighted']:.4f}")
print(f"  Cohen's Kappa: {results_supervised['cohen_kappa']:.4f}")

print("\nUNSUPERVISED ANOMALY DETECTION:")
comparison_df = pd.DataFrame({
    'Isolation Forest': {
        'Accuracy': results_if['accuracy'],
        'Precision': results_if['precision'],
        'Recall': results_if['recall'],
        'F1': results_if['f1'],
        'AUC': results_if['auc']
    },
    'One-Class SVM': {
        'Accuracy': results_ocsvm['accuracy'],
        'Precision': results_ocsvm['precision'],
        'Recall': results_ocsvm['recall'],
        'F1': results_ocsvm['f1'],
        'AUC': results_ocsvm['auc']
    },
    'Ensemble': {
        'Accuracy': results_ensemble['accuracy'],
        'Precision': results_ensemble['precision'],
        'Recall': results_ensemble['recall'],
        'F1': results_ensemble['f1'],
        'AUC': results_ensemble['auc']
    }
}).T

print("\nComparison of Anomaly Detection Methods:")
print(comparison_df.to_string())

# ============================================================================
# STEP 6: ONLINE PREDICTION DEMO
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Online Prediction Demonstration")
print("="*80)

# Generate a new test sample (bearing fault)
test_signal = generate_fault_signal('Bearing Fault')

# Test supervised classifier
print("\nSupervised Classifier Prediction:")
prediction = classifier.predict_online([
    test_signal['X'], test_signal['Y'], test_signal['Z']
])
print(f"  Predicted class: {prediction['class']}")
print(f"  Confidence: {prediction['probability']:.4f}")
print(f"  All probabilities:")
for cls, prob in prediction['probabilities'].items():
    print(f"    {cls:20s}: {prob:.4f}")

# Test anomaly detector
print("\nAnomaly Detector Prediction:")
result = detector_ensemble.detect_online([
    test_signal['X'], test_signal['Y'], test_signal['Z']
])
print(f"  Is anomaly: {result['is_anomaly']}")
print(f"  Anomaly score: {result['anomaly_score']:.6f}")
print(f"  Threshold: {result['threshold']:.6f}")
print(f"  Severity: {result['severity']}")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*80)
print("DEMONSTRATION COMPLETE!")
print("="*80)
print(f"\nTrained models saved to: {model_dir}")
print("\nNext steps:")
print("1. Replace synthetic data with your actual pump vibration data")
print("2. Adjust hyperparameters in configs/")
print("3. Deploy models for online monitoring")
print("4. See README.md and PUMP_FAULT_DETECTION_GUIDE.md for detailed documentation")
print("\n" + "="*80)

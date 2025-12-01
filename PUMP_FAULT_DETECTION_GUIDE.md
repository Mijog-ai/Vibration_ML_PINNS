# Complete Guide: Pump Fault Detection Using Accelerometer Data

## Table of Contents
1. [Overview](#overview)
2. [Signal Processing & Feature Extraction](#signal-processing--feature-extraction)
3. [Supervised Learning Pipeline](#supervised-learning-pipeline)
4. [Unsupervised Learning Pipeline](#unsupervised-learning-pipeline)
5. [Handling Variable Operating Conditions](#handling-variable-operating-conditions)
6. [Training & Validation Strategy](#training--validation-strategy)
7. [Deployment for Online Monitoring](#deployment-for-online-monitoring)
8. [Sensor Placement & Practical Tips](#sensor-placement--practical-tips)

---

## Overview

### Sensor Setup
- **Accelerometers**: Mounted on pump shell, near bearings and load-carrying structures
- **Measurement Directions**: Axial, Radial (X, Y, Z)
- **Sampling Rate**: 10.24 kHz (as per your data)
- **Operating Conditions**: Variable speed, pressure, displacement
- **Fault Classes**: Normal, Piston wear, Bearing faults, Misalignment, Cavitation, etc.

### Pipeline Goals
1. **Supervised**: Multi-class fault classification with known labels
2. **Unsupervised**: Early anomaly detection without requiring labeled fault data

---

## Signal Processing & Feature Extraction

### 1. Preprocessing Steps

#### 1.1 Signal Cleaning
```
Raw Signal → Detrending → High-pass Filter (>1 Hz) → Outlier Removal
```

**Why?**
- **Detrending**: Removes DC offset and slow drift
- **High-pass filter**: Eliminates low-frequency motion artifacts
- **Outlier removal**: Clipping, EMI spikes removal

#### 1.2 Segmentation
- **Window Length**:
  - **Stationary conditions**: 2-5 seconds (20,480-51,200 samples @ 10.24 kHz)
  - **Transient analysis**: 0.5-1 second
- **Overlap**: 50% (for online monitoring) to 0% (for training to avoid leakage)

**Why?**
- Stationarity assumption for FFT
- Balance between frequency resolution and computational cost

### 2. Time-Domain Features

Extract from each windowed segment:

| Feature | Formula | Why Important |
|---------|---------|---------------|
| **RMS** | √(1/N Σx²) | Overall energy, sensitive to all fault types |
| **Peak** | max(\|x\|) | Detects impulsive events (bearing spalls, cavitation) |
| **Crest Factor** | Peak/RMS | Bearing faults produce high peaks |
| **Kurtosis** | E[(x-μ)⁴]/σ⁴ | Sensitive to impulsive faults (>3 = spiky) |
| **Skewness** | E[(x-μ)³]/σ³ | Asymmetry (misalignment, shaft bow) |
| **Shape Factor** | RMS/Mean(\|x\|) | Waveform shape change |
| **Impulse Factor** | Peak/Mean(\|x\|) | Similar to crest factor |
| **Clearance Factor** | Peak/√(Mean(√\|x\|))² | More sensitive to early bearing damage |

**Extract for**: Each axis (X, Y, Z) separately

### 3. Frequency-Domain Features

Apply FFT to each window:

#### 3.1 Spectral Statistics
- **Peak Frequency**: Location of dominant peak
- **Peak Amplitude**: Magnitude at peak frequency
- **Spectral Centroid**: Σ(f·P(f))/ΣP(f) - "center of mass" of spectrum
- **Spectral Spread**: Spread around centroid
- **Spectral Entropy**: -Σ(P(f)·log(P(f))) - randomness measure

**Why?**
- Bearing faults generate harmonics at characteristic frequencies (BPFO, BPFI, BSF, FTF)
- Misalignment shows 1×, 2× running speed peaks
- Cavitation creates broadband noise

#### 3.2 Band Power Features
Divide spectrum into bands and compute power in each:

| Band | Frequency Range | Fault Type |
|------|-----------------|------------|
| **Low** | 1-10 Hz | Misalignment, imbalance, cavitation onset |
| **Mid** | 10-1000 Hz | Rotating component faults, bearing outer race |
| **High** | 1-5 kHz | Bearing inner race, high-frequency resonances |
| **Very High** | 5-10 kHz | Early bearing damage, lubrication issues |

**Rationale**:
- Different faults excite different frequency ranges
- Band ratios (High/Low) detect fault progression

#### 3.3 Harmonic Features
For rotating machinery at speed *f_rot*:
- Power at 1×, 2×, 3×, 4× *f_rot*
- Sideband power around bearing frequencies ± *f_rot*

**Why?**
- Imbalance: Strong 1× component
- Misalignment: Strong 2×, 3× components
- Bearing faults: Modulation sidebands

### 4. Time-Frequency Features

#### 4.1 Short-Time Fourier Transform (STFT)
- **Spectrogram**: 2D representation (time × frequency)
- **Use case**: Non-stationary signals, transient detection
- **Features**: Can extract as image for CNN input or compute statistics

#### 4.2 Wavelet Transform
- **Continuous Wavelet Transform (CWT)**: Multi-resolution analysis
- **Discrete Wavelet Transform (DWT)**: Decompose into detail/approximation coefficients
- **Wavelet Packet Decomposition (WPD)**: Energy in each sub-band

**Recommended Wavelets**:
- **Morlet**: Good for bearing fault detection (sinusoidal with Gaussian envelope)
- **Daubechies db4-db8**: General-purpose for DWT

**Why Wavelets?**
- Capture transient, impulsive events (bearing spalls)
- Better time-frequency localization than STFT for short-duration events

**Features from Wavelets**:
- Energy in detail coefficients D1-D5
- Entropy of detail coefficients
- Peak ratios between levels

#### 4.3 Envelope Analysis
```
Signal → Band-pass filter (high freq) → Hilbert transform → Extract envelope → FFT of envelope
```

**Why?**
- **Gold standard** for bearing fault detection
- Demodulates high-frequency bearing resonances
- Envelope spectrum shows clear bearing fault frequencies

**Implementation**:
1. Band-pass filter around bearing resonance (e.g., 2-10 kHz)
2. Apply Hilbert transform to get analytic signal
3. Extract envelope (magnitude of analytic signal)
4. FFT of envelope → look for BPFO, BPFI, BSF

### 5. Order Domain Features

For variable-speed operation:

- **Order Tracking**: Resample signal in angular domain (vs time)
- **Order Analysis**: FFT in order domain (multiples of shaft speed)

**Why?**
- Speed-independent feature extraction
- Critical when speed varies during measurement

### 6. Advanced Features

#### 6.1 Cyclostationary Analysis
- **Spectral Correlation**: Detects periodic modulation
- **Cyclic Frequency**: Identifies fault characteristic frequencies

**Why?**
- Robust to noise
- Detects early bearing faults with low SNR

#### 6.2 Entropy Measures
- **Approximate Entropy (ApEn)**: Regularity/predictability
- **Sample Entropy**: Improved ApEn (less bias)
- **Permutation Entropy**: Complexity measure

**Why?**
- Healthy machines have regular patterns
- Faults introduce complexity/randomness

---

## Supervised Learning Pipeline

### Goal
Multi-class classification: {Normal, Piston Wear, Bearing Fault, Misalignment, Cavitation, ...}

### Step 1: Data Preparation

```python
# For each operating condition (speed, pressure):
#   For each health state:
#     For each sensor (front/rear bearing, pump body):
#       For each axis (X, Y, Z):
#         - Load time-series data
#         - Segment into windows (no overlap for training)
#         - Extract features → Feature vector
#
# Result: DataFrame with columns:
#   [features...] | speed | pressure | sensor_loc | axis | health_state
```

### Step 2: Feature Engineering

#### 2.1 Recommended Feature Set (60-100 features per axis)
```
Time-domain (8 features × 3 axes = 24)
+ Frequency bands (4 bands × 3 axes = 12)
+ Spectral stats (5 features × 3 axes = 15)
+ Envelope features (4 bearing freqs × 3 axes = 12)
+ Wavelet energies (5 levels × 3 axes = 15)
+ Cross-axis features (coherence, correlation = 6)
────────────────────────────────────────────
≈ 84 features total
```

#### 2.2 Feature Selection
- **Filter Methods**: Correlation with target, mutual information
- **Wrapper Methods**: Recursive Feature Elimination (RFE)
- **Embedded Methods**: L1 regularization (Lasso), tree-based importance

**Why?**
- Reduce dimensionality
- Remove redundant/irrelevant features
- Improve generalization

### Step 3: Model Selection

#### Option A: Classical ML (Recommended for Small-Medium Datasets)

##### 1. **Random Forest (RF)**
```python
# Pros:
- Robust to noise, handles non-linear relationships
- Feature importance built-in
- Little hyperparameter tuning needed
- Handles multi-class naturally

# Cons:
- Can overfit with many correlated features
- Black-box (less interpretable than single tree)

# When to use:
- First baseline model
- 100-10k samples
- Mixed feature types
```

**Hyperparameters**:
- n_estimators: 100-500
- max_depth: 10-30
- min_samples_split: 2-10

##### 2. **Support Vector Machine (SVM)**
```python
# Pros:
- Excellent for high-dimensional spaces
- Effective with clear margin of separation
- Kernel trick for non-linear boundaries

# Cons:
- Slow training for large datasets
- Sensitive to feature scaling
- Requires careful hyperparameter tuning (C, gamma)

# When to use:
- <5k samples
- High feature-to-sample ratio
- Need probabilistic output (use SVC with probability=True)
```

**Recommended Kernel**: RBF (Gaussian) for most cases

##### 3. **Gradient Boosting (XGBoost/LightGBM)**
```python
# Pros:
- Often best performance on tabular data
- Handles missing values
- Built-in regularization

# Cons:
- Prone to overfitting if not tuned
- Longer training time

# When to use:
- >1k samples
- Maximize accuracy
- Have compute resources for tuning
```

##### 4. **k-Nearest Neighbors (k-NN)**
```python
# Pros:
- Simple, interpretable
- No training phase
- Naturally handles multi-class

# Cons:
- Slow prediction for large datasets
- Sensitive to feature scaling and curse of dimensionality
- Needs good distance metric

# When to use:
- Very small dataset (<1k)
- Baseline comparison
- Online learning (add new samples easily)
```

#### Option B: Deep Learning (For Large Datasets)

##### 1. **1D CNN on Raw Time-Series**
```python
Input: [batch, time_steps, channels]
       e.g., [32, 10240, 3]  # 1-sec window, X/Y/Z

Architecture:
Conv1D(64, kernel=16) → ReLU → MaxPool(4)
Conv1D(128, kernel=8) → ReLU → MaxPool(4)
Conv1D(256, kernel=4) → ReLU → MaxPool(4)
GlobalAvgPool → Dense(128) → Dropout(0.5) → Dense(num_classes)

# Pros:
- Learns features automatically
- Captures local temporal patterns
- No manual feature engineering

# Cons:
- Needs >10k samples
- Black-box
- Requires more compute

# When to use:
- Large dataset (>10k samples)
- Raw signal quality is good
- Want to avoid feature engineering
```

##### 2. **2D CNN on Spectrograms**
```python
Input: [batch, height, width, channels]
       e.g., [32, 128, 128, 3]  # Spectrogram for X/Y/Z

Architecture:
Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
Conv2D(64, 3×3) → ReLU → MaxPool(2×2)
Conv2D(128, 3×3) → ReLU → MaxPool(2×2)
Flatten → Dense(256) → Dropout(0.5) → Dense(num_classes)

# Pros:
- Leverages image classification techniques
- Transfer learning possible (ResNet, VGG pre-trained)
- Visualizable (Grad-CAM)

# Cons:
- Spectrogram computation overhead
- Loses some temporal resolution

# When to use:
- Time-frequency patterns important
- Transfer learning desired
- >5k samples
```

##### 3. **LSTM / GRU for Sequence Modeling**
```python
Input: [batch, time_steps, features]
       e.g., [32, 100, 84]  # 100 time windows, 84 features each

Architecture:
LSTM(128, return_sequences=True)
LSTM(64, return_sequences=False)
Dense(64) → Dropout(0.3) → Dense(num_classes)

# Pros:
- Captures long-term dependencies
- Good for sequential patterns

# Cons:
- Slow training
- Prone to overfitting
- Needs careful tuning

# When to use:
- Long sequences with temporal dependencies
- Fault progression modeling
- >5k sequences
```

##### 4. **Hybrid: CNN + LSTM**
```python
# Extract local features with CNN, then model temporal with LSTM
Conv1D layers → LSTM layers → Dense

# When to use:
- Best of both worlds
- Complex temporal patterns
- Large dataset
```

### Step 4: Class Imbalance Handling

Vibration data often has:
- Many normal samples
- Few fault samples (especially rare faults)

**Strategies**:
1. **Class Weighting**: Set `class_weight='balanced'` in sklearn
2. **SMOTE**: Synthetic Minority Over-sampling (for tabular features)
3. **Augmentation**: Time-series augmentation (jittering, scaling, time-warping)
4. **Focal Loss**: For deep learning (focuses on hard examples)
5. **Ensemble**: Train separate classifiers for each fault vs. normal

### Step 5: Model Training Workflow

```python
# Pseudocode
for each_operating_condition in [all_speeds, all_pressures]:
    # Option 1: Train condition-specific models
    model = RandomForest()
    X_train, y_train = get_condition_data(each_operating_condition)
    model.fit(X_train, y_train)

    # Option 2: Train global model with condition as feature
    # (see "Handling Variable Operating Conditions" section)
```

### Step 6: Interpretability & Diagnostics

- **Feature Importance**: From RF, XGBoost
- **SHAP Values**: Explain predictions
- **Confusion Matrix**: Which faults are confused?
- **Per-Class Precision/Recall**: Critical for rare faults

---

## Unsupervised Learning Pipeline

### Goal
Detect anomalies (early faults) without labeled fault data

### When to Use Unsupervised?
- **Insufficient labeled fault data**
- **Novel/unknown fault types**
- **Early warning system** (before fault categorized)
- **Normal operation baseline** available

### Approach 1: Reconstruction-Based (Deep Learning)

#### 1.1 **Autoencoder (AE)**

```python
Architecture:
Input (raw or features) → Encoder → Latent (bottleneck) → Decoder → Reconstruction

Encoder:  Dense(128) → Dense(64) → Dense(32)  # Compress
Latent:   Dense(16)                           # Bottleneck
Decoder:  Dense(32) → Dense(64) → Dense(128) → Dense(input_dim)

Training: Only on NORMAL data
Loss: MSE(input, reconstruction)

Anomaly Score: Reconstruction error for new sample
Threshold: Set at 95th-99th percentile of training reconstruction errors
```

**Pros**:
- Simple, effective
- Can use raw signals or features
- Visualize latent space

**Cons**:
- May not capture all normal variations
- Sensitive to hyperparameters

**When to use**:
- >2k normal samples
- Want to learn complex normal patterns

#### 1.2 **Variational Autoencoder (VAE)**

```python
# Similar to AE but with probabilistic latent space
# Latent space follows N(0, 1) distribution

Anomaly Score: Reconstruction error + KL divergence
```

**Pros**:
- More robust than AE (regularized latent space)
- Better generalization

**When to use**:
- Same as AE but with more data (>5k normal samples)

#### 1.3 **LSTM Autoencoder** (for time-series)

```python
Encoder: LSTM(128) → LSTM(64) → LSTM(32, return_sequences=False)
Latent:  Dense(16)
Decoder: RepeatVector(time_steps) → LSTM(32) → LSTM(64) → LSTM(128) → TimeDistributed(Dense(features))

# Reconstructs entire sequence
```

**When to use**:
- Sequential anomalies (temporal patterns)
- Fault progression detection

### Approach 2: Distance-Based

#### 2.1 **One-Class SVM (OC-SVM)**

```python
from sklearn.svm import OneClassSVM

model = OneClassSVM(kernel='rbf', nu=0.05)  # nu = expected outlier fraction
model.fit(X_normal)  # Only normal data

y_pred = model.predict(X_test)  # +1 = normal, -1 = anomaly
```

**Pros**:
- Works with small datasets
- No assumptions on data distribution
- Effective for high-dimensional features

**Cons**:
- Sensitive to nu and gamma hyperparameters
- Slow for large datasets

**When to use**:
- 500-5k normal samples
- Extracted features (not raw signals)

#### 2.2 **Isolation Forest**

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05, n_estimators=200)
model.fit(X_normal)

y_pred = model.predict(X_test)  # +1 = normal, -1 = anomaly
anomaly_scores = model.score_samples(X_test)  # Lower = more anomalous
```

**Pros**:
- Fast training and prediction
- Handles high-dimensional data
- No need to tune kernel

**Cons**:
- Less effective if anomalies are clustered

**When to use**:
- First baseline for anomaly detection
- >1k normal samples
- Fast deployment needed

#### 2.3 **Local Outlier Factor (LOF)**

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_normal)

y_pred = lof.predict(X_test)
```

**Pros**:
- Detects local density anomalies
- Works with complex geometries

**Cons**:
- Slow for large datasets
- Sensitive to n_neighbors

**When to use**:
- Anomalies in low-density regions
- <10k samples

### Approach 3: Clustering-Based

#### 3.1 **k-Means Distance**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(X_normal)

# Anomaly score = distance to nearest cluster center
distances = kmeans.transform(X_test).min(axis=1)
threshold = np.percentile(distances_train, 95)
anomalies = distances > threshold
```

**Pros**:
- Simple, fast
- Interpretable (which cluster?)

**Cons**:
- Assumes spherical clusters
- Need to choose k

**When to use**:
- Multiple normal operating modes
- Exploratory analysis

#### 3.2 **Gaussian Mixture Model (GMM)**

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X_normal)

# Anomaly score = negative log-likelihood
log_likelihood = gmm.score_samples(X_test)
anomalies = log_likelihood < threshold
```

**Pros**:
- Probabilistic
- Handles elliptical clusters

**When to use**:
- Multiple operating conditions with smooth transitions

#### 3.3 **DBSCAN for Outlier Detection**

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_normal)

# Samples with label -1 are outliers
```

**When to use**:
- Noisy training data
- Unknown number of normal modes

### Approach 4: Statistical Methods

#### 4.1 **Mahalanobis Distance**

```python
from scipy.spatial.distance import mahalanobis

# Compute covariance on normal data
mean = X_normal.mean(axis=0)
cov = np.cov(X_normal, rowvar=False)
cov_inv = np.linalg.inv(cov)

# Anomaly score for each test sample
distances = [mahalanobis(x, mean, cov_inv) for x in X_test]
threshold = chi2.ppf(0.95, df=X_test.shape[1])  # Chi-square threshold
anomalies = distances > threshold
```

**Pros**:
- Accounts for feature correlations
- Statistically grounded threshold

**Cons**:
- Assumes Gaussian distribution
- Fails if features are collinear (singular covariance matrix)

**When to use**:
- Approximately Gaussian features
- Small-medium datasets

#### 4.2 **Statistical Process Control (SPC)**

Monitor time-domain features over time:

```python
# For each feature (RMS, kurtosis, etc.):
mean = feature_train.mean()
std = feature_train.std()

# Control limits
UCL = mean + 3*std  # Upper Control Limit
LCL = mean - 3*std  # Lower Control Limit

# Alarm if feature exceeds limits
```

**Pros**:
- Simple, interpretable
- Real-time monitoring

**Cons**:
- Univariate (doesn't capture feature interactions)
- Needs stable operating conditions

**When to use**:
- Continuous monitoring
- Simple baseline
- Operator dashboards

### Recommended Unsupervised Strategy

**Hybrid Approach**:
1. **Isolation Forest** for initial screening (fast, robust)
2. **Autoencoder** for complex pattern learning
3. **SPC** on key features (RMS, kurtosis) for real-time monitoring

**Ensemble Anomaly Detection**:
```python
# Combine scores from multiple methods
score_total = w1*score_IForest + w2*score_AE + w3*score_OCSVM
anomaly = score_total > threshold
```

---

## Handling Variable Operating Conditions

### Challenge
Vibration patterns change with speed, pressure, load → Need condition-aware models

### Strategy 1: Separate Models per Condition

**Approach**:
- Discretize operating conditions (e.g., low/medium/high speed)
- Train separate classifier for each condition

**Pros**:
- Simple
- Optimal per-condition performance

**Cons**:
- Needs labeled data for ALL conditions
- Poor generalization to unseen conditions
- Model management complexity

**When to use**:
- Few discrete operating points
- Abundant data per condition

### Strategy 2: Condition as Feature

**Approach**:
- Include speed, pressure as input features
- Train single global model

**Implementation**:
```python
features = [
    time_domain_features,
    frequency_domain_features,
    speed,           # RPM
    pressure,        # Bar
    displacement,    # %
]
```

**Pros**:
- Single model
- Generalizes to new conditions (interpolation)

**Cons**:
- Model complexity increases
- Condition variables need scaling

**When to use**:
- Continuous range of operating conditions
- Large dataset

### Strategy 3: Condition-Normalized Features

**Approach**:
- Compute features relative to baseline for that condition

**Example**:
```python
# Pre-compute baseline RMS for each speed
baseline_RMS[speed] = RMS_normal_data_at_speed(speed)

# Normalized feature
RMS_normalized = RMS_current / baseline_RMS[current_speed]
```

**Pros**:
- Condition-independent features
- Robust to operating point changes

**Cons**:
- Needs normal baseline for each condition
- Assumption: fault signature scales with condition

**When to use**:
- Wide range of operating conditions
- Difficulty getting fault data at all conditions

### Strategy 4: Order Domain Analysis

**Approach**:
- Convert time-domain signals to angle-domain (order tracking)
- Features computed in orders (multiples of shaft rotation) not Hz

**Why?**
- Bearing fault frequencies are proportional to speed
- E.g., BPFO = N_balls × (1 - d/D·cosα) × f_rot
- Order analysis makes features speed-invariant

**Implementation**:
```python
# Resample signal w.r.t. shaft angle (requires tachometer or speed signal)
angle_signal = resample_by_angle(time_signal, speed_profile)

# FFT in order domain
order_spectrum = fft(angle_signal)

# Extract order-based features
power_at_order_2 = order_spectrum[2]  # 2× running speed
```

**When to use**:
- Speed varies during measurement
- Rotating machinery faults
- Have speed reference signal

### Strategy 5: Transfer Learning / Domain Adaptation

**Approach**:
- Train model on condition A (source domain)
- Adapt to condition B (target domain) with few samples

**Methods**:
- Fine-tuning (for deep learning)
- Domain adversarial training
- Maximum Mean Discrepancy (MMD)

**When to use**:
- Large data at some conditions, sparse at others
- Deep learning models

### Recommended Strategy

**For Supervised**:
- Use **Strategy 2 (condition as feature)** + **Strategy 3 (normalization)**
- Extract both absolute and normalized features
- Let model learn which to use

**For Unsupervised**:
- **Strategy 3 (normalization)** essential
- Build separate normal baselines for each condition
- Monitor normalized anomaly scores

---

## Training & Validation Strategy

### 1. Data Splitting Principles

#### Avoid Data Leakage!
❌ **WRONG**:
```python
# Split random windows from same recording
X = extract_windows(recording, overlap=50%)
X_train, X_test = train_test_split(X, test_size=0.2)
# ⚠️ Leakage: train and test windows from same recording are correlated!
```

✅ **CORRECT**:
```python
# Split by recording (time-series), then extract windows
recordings = [rec1, rec2, rec3, ..., rec100]
train_recs, test_recs = train_test_split(recordings, test_size=0.2)

X_train = extract_windows(train_recs, overlap=0%)    # No overlap for training
X_test = extract_windows(test_recs, overlap=50%)     # Overlap OK for testing
```

### 2. Recommended Split Strategy

#### For Independent Recordings:
```
60% Training | 20% Validation | 20% Test
```

#### For Time-Series (Sequential Data):
```
Chronological split (no shuffling):
├─ Training: First 60% of timeline
├─ Validation: Next 20%
└─ Test: Last 20%
```

**Why chronological?**
- Simulates deployment (predict future from past)
- Avoids temporal leakage

### 3. Cross-Validation Design

#### Option A: Stratified K-Fold (for independent samples)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Fold {fold}: {score}")
```

**Use when**:
- Samples are independent (no temporal/spatial correlation)
- Balanced folds needed

#### Option B: Group K-Fold (for correlated samples)

```python
from sklearn.model_selection import GroupKFold

# Groups = recording ID, sensor location, or operating condition
groups = [1,1,1,2,2,2,3,3,3, ...]  # Recording IDs

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=groups):
    # Ensures all windows from same recording stay in same fold
    ...
```

**Use when**:
- Multiple windows from same recording
- Want to avoid leakage

#### Option C: Time-Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    # Always trains on past, validates on future
    ...
```

**Use when**:
- Sequential data
- Want to simulate online deployment

### 4. Hyperparameter Tuning

#### Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1_weighted',  # Or 'accuracy', 'precision', etc.
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

#### Random Search (faster for large spaces)

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=50,  # Number of random combinations
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)
```

### 5. Evaluation Metrics

#### For Supervised Classification:

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| **Precision** | TP/(TP+FP) | Cost of false alarms high |
| **Recall** | TP/(TP+FN) | Cost of missed faults high |
| **F1-Score** | 2·Prec·Rec/(Prec+Rec) | Balance prec/rec |
| **F2-Score** | 5·Prec·Rec/(4·Prec+Rec) | Recall more important |
| **Cohen's Kappa** | Agreement beyond chance | Multi-class, imbalanced |
| **Matthews Corr.** | Correlation coef | Imbalanced binary |

**For Vibration Fault Detection**:
- **Prioritize Recall**: Missing a fault is worse than false alarm
- Use **F2-Score** or **Recall@Precision≥0.8**
- Report **per-class metrics** (some faults harder to detect)

#### For Unsupervised Anomaly Detection:

**Challenges**:
- No labels for validation
- Need labeled test set to evaluate

**Metrics** (if test labels available):
- **True Positive Rate (TPR)**: Fault detection rate
- **False Positive Rate (FPR)**: False alarm rate
- **ROC-AUC**: Area under ROC curve (vary threshold)
- **Precision@k**: Precision for top-k anomalies

**Practical Approach**:
```python
# Train on normal data
model.fit(X_normal_train)

# Evaluate on labeled test set (normal + faults)
anomaly_scores = model.score_samples(X_test)
y_true = [0,0,0,...,1,1,1]  # 0=normal, 1=fault

# ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_true, -anomaly_scores)
auc = roc_auc_score(y_true, -anomaly_scores)

# Choose threshold: balance TPR/FPR
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
```

**If No Labels Available**:
- **Visual Inspection**: Plot anomaly scores over time, investigate high-score samples
- **Expert Review**: Show top anomalies to domain experts
- **Consistency**: Ensure stable scores on repeated normal data

### 6. Model Validation Checklist

✅ **Before Deployment**:
- [ ] No data leakage (check split strategy)
- [ ] Cross-validation score stable across folds
- [ ] Test set performance close to validation
- [ ] Per-class metrics acceptable for all faults
- [ ] Model tested on new sensor data (not in training)
- [ ] Model tested on different pump (if generalizing)
- [ ] Feature distributions similar train/test (use PCA, t-SNE to visualize)
- [ ] False positive rate acceptable for operations

---

## Deployment for Online Monitoring

### 1. System Architecture

```
Sensor → Data Acquisition → Preprocessing → Feature Extraction → Model → Decision Logic → Alarm
         (10.24 kHz)        (Filtering)     (Real-time)         (Inference) (Thresholds)
```

### 2. Window Length & Overlap

#### Window Length Selection

| Consideration | Recommendation |
|--------------|----------------|
| **Frequency Resolution** | Δf = fs / N → For 1 Hz resolution: N = 10,240 samples (1 sec @ 10.24 kHz) |
| **Stationarity** | Shorter windows (0.5-2 sec) for variable conditions |
| **Statistical Reliability** | Longer windows (2-5 sec) for stable statistics |
| **Latency** | Shorter windows (0.5-1 sec) for fast response |

**Recommended**:
- **Default**: 2.56 sec (26,214 samples) → 0.39 Hz resolution
- **Fast Response**: 1 sec (10,240 samples) → 1 Hz resolution

#### Overlap Selection

| Use Case | Overlap | Why |
|----------|---------|-----|
| **Online Monitoring** | 50-75% | Smooth updates, don't miss transients |
| **Training** | 0% | Avoid data leakage |
| **High-Speed Detection** | 90% | Near-continuous monitoring |

**Recommended**: 50% overlap (update every 1.28 sec for 2.56 sec window)

### 3. Real-Time Processing Pipeline

```python
# Pseudocode for online monitoring

buffer = CircularBuffer(size=window_size * (1 + overlap))

while True:
    # Acquire new data
    new_samples = sensor.read(batch_size)
    buffer.append(new_samples)

    if buffer.is_full():
        # Extract window
        window = buffer.get_window()

        # Preprocess
        window = high_pass_filter(window, cutoff=1Hz)
        window = remove_outliers(window)

        # Feature extraction
        features = extract_features(window)  # 84-dim vector

        # Normalize (using training statistics)
        features_norm = (features - mean_train) / std_train

        # Predict
        if supervised_mode:
            prediction = model.predict(features_norm)  # Fault class
            probability = model.predict_proba(features_norm)

            if prediction != 'Normal' and max(probability) > 0.8:
                raise_alarm(prediction, probability)

        elif unsupervised_mode:
            anomaly_score = model.score_samples(features_norm)

            if anomaly_score < threshold:
                raise_alarm('Anomaly', anomaly_score)

        # Slide window
        buffer.slide(shift=int(window_size * (1 - overlap)))

    time.sleep(acquisition_period)
```

### 4. Decision Logic & Thresholds

#### Single-Threshold Approach
```python
if anomaly_score > threshold:
    alarm = True
```

**Problem**: Single transient can trigger false alarm

#### Improved: Persistence Check
```python
# Alarm only if anomaly persists
consecutive_anomalies = 0

if anomaly_score > threshold:
    consecutive_anomalies += 1
else:
    consecutive_anomalies = 0

if consecutive_anomalies >= N_persist:  # e.g., N_persist = 3
    alarm = True
```

#### Multi-Level Alarms
```python
if anomaly_score > critical_threshold:
    alarm = 'CRITICAL - Stop operation'
elif anomaly_score > warning_threshold:
    alarm = 'WARNING - Schedule maintenance'
elif anomaly_score > caution_threshold:
    alarm = 'CAUTION - Monitor closely'
else:
    alarm = 'NORMAL'
```

**Recommended Thresholds** (for anomaly detection):
- **Caution**: 90th percentile of training scores
- **Warning**: 95th percentile
- **Critical**: 99th percentile

#### Adaptive Thresholds
```python
# Update threshold based on recent history (moving window)
recent_scores = scores[-1000:]  # Last 1000 windows
threshold = np.percentile(recent_scores, 95)
```

**Use when**: Operating conditions drift over time

### 5. Handling False Positives

**Strategies**:
1. **Persistence**: Require N consecutive anomalies
2. **Voting**: Ensemble of models (alarm if ≥2/3 agree)
3. **Multi-Sensor Fusion**: Confirm anomaly across multiple sensors
4. **Context Checks**:
   - Ignore alarms during startup/shutdown
   - Check if operating condition changed
5. **Expert-in-the-Loop**: Flag for review before stopping equipment

### 6. Latency Considerations

**End-to-End Latency**:
```
Window acquisition (1-2 sec)
+ Preprocessing (10-50 ms)
+ Feature extraction (50-200 ms)
+ Model inference (1-100 ms)
+ Decision logic (1 ms)
────────────────────────────
Total: 1.1 - 2.4 sec
```

**For Critical Applications**:
- Use **0.5 sec windows** → Reduce to 0.6-1 sec latency
- **Optimize feature extraction** (pre-compute FFT plans)
- **Model optimization** (quantization, pruning for deep learning)
- **Edge deployment** (avoid network latency)

### 7. Data Logging & Visualization

**Log for Each Prediction**:
```python
log_entry = {
    'timestamp': datetime.now(),
    'features': features,
    'prediction': prediction,
    'anomaly_score': score,
    'operating_condition': {'speed': rpm, 'pressure': bar},
    'raw_window': window  # Optional: for post-analysis
}
```

**Dashboard Elements**:
1. **Real-Time Plot**: Anomaly score over time
2. **Feature Trends**: RMS, kurtosis, peak frequency
3. **Alarm Log**: Recent alarms with severity
4. **Spectrogram**: Live frequency spectrum
5. **Operating Conditions**: Speed, pressure, temperature

### 8. Model Update Strategy

**Challenge**: Model degrades as pump ages or conditions change

**Solutions**:
1. **Scheduled Retraining**: Monthly/quarterly with new data
2. **Online Learning**: Incrementally update model (k-NN, OC-SVM)
3. **Active Learning**: Operator labels flagged anomalies → retrain
4. **Ensemble**: Combine old and new models (weight by recency)

### 9. Edge vs. Cloud Deployment

#### Edge (Local Controller)
**Pros**:
- Low latency
- No network dependency
- Data privacy

**Cons**:
- Limited compute (use lightweight models)
- Harder to update

**Recommended Models**: Random Forest, Isolation Forest, lightweight CNN

#### Cloud
**Pros**:
- Powerful compute (complex models, ensemble)
- Easy to update
- Centralized monitoring (multiple pumps)

**Cons**:
- Network latency
- Requires connectivity

**Recommended**: **Hybrid**
- Edge: Fast anomaly detection (Isolation Forest)
- Cloud: Deep analysis, model training, trend analysis

---

## Sensor Placement & Practical Tips

### 1. Optimal Sensor Locations

#### For Bearing Fault Detection
- **Primary**: On bearing housing (as close as possible to bearing)
- **Secondary**: On pump casing adjacent to bearing
- **Avoid**: Flexible structures, far from fault source

**Rule of Thumb**: Signal attenuates ~20 dB per structural junction

#### For Pump-Specific Faults

| Fault Type | Best Sensor Location |
|------------|---------------------|
| **Bearing Faults** | Bearing housing (radial/axial) |
| **Misalignment** | Near coupling, both radial directions |
| **Imbalance** | Mid-span of shaft, radial |
| **Cavitation** | Pump inlet/outlet, volute casing |
| **Piston Wear** | Cylinder block, axial direction |
| **Valve Issues** | Valve plate, axial |

**Minimum Configuration**: 3 accelerometers
- Front bearing housing (radial X, Y, axial Z)
- Rear bearing housing (radial X, Y, axial Z)
- Pump body/volute (radial X, Y, axial Z)

**Recommended**: 6-9 sensors total (redundancy + full coverage)

### 2. Mounting Methods

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Stud Mount** | Best frequency response (up to 10 kHz) | Permanent, drilling required | Fixed installation |
| **Adhesive (Epoxy)** | Good response (up to 7 kHz), no drilling | Permanent | Semi-permanent |
| **Magnet** | Removable, easy | Limited freq (up to 2 kHz), poor at high freq | Temporary, surveys |
| **Wax** | Temporary, cheap | Very limited | Quick tests |

**For Your Application** (up to 5 kHz needed):
- **Permanent monitoring**: Stud mount (drill + screw)
- **Test bench**: Strong magnet or adhesive

**Critical**:
- Clean surface (grind, degrease)
- Flat mounting surface
- Tighten stud to spec (10-20 Nm)
- Check resonance frequency of sensor (should be >5× max freq of interest)

### 3. Sensor Specifications

**For Pump Vibration**:
- **Type**: IEPE (ICP) accelerometers
- **Sensitivity**: 10-100 mV/g
  - High sensitivity (100 mV/g): Low vibration (<10 g)
  - Low sensitivity (10 mV/g): High vibration (>50 g)
- **Frequency Range**: 1 Hz - 10 kHz (minimum)
- **Measurement Range**: ±50 g (typical), ±500 g (high-shock)
- **Temperature Range**: -40°C to +120°C (industrial)

**Recommended Brands**: PCB Piezotronics, Brüel & Kjær, Kistler, Dytran

### 4. Data Acquisition Setup

#### Sampling Rate
- **Nyquist**: fs ≥ 2 × f_max
- **Practical**: fs ≥ 2.56 × f_max (to avoid aliasing with real-world signals)
- **Your Setup**: 10.24 kHz → Good up to 4 kHz

**If Bearing Fault Frequencies > 4 kHz**:
- Increase sampling to 20-50 kHz
- Or use envelope analysis (demodulate high-freq carrier)

#### Anti-Aliasing Filter
- **Essential**: Hardware low-pass filter before ADC
- **Cutoff**: 0.4 × fs (e.g., 4 kHz for 10.24 kHz sampling)
- **Order**: 4th-order Butterworth or better

#### ADC Resolution
- **Minimum**: 16-bit (provides 96 dB dynamic range)
- **Recommended**: 24-bit for high dynamic range

### 5. Noise Reduction Techniques

#### Electrical Noise
- **Use IEPE sensors**: Built-in amplifier → robust to cable noise
- **Shielded cables**: Ground shield at one end only (avoid ground loops)
- **Cable routing**: Away from power cables, motors, VFDs
- **Differential input**: If available on DAQ
- **Grounding**: Single-point grounding system

#### Mechanical Noise
- **Isolate from external vibration**:
  - Mount pump on isolation pads
  - Ensure rigid mounting for sensors
- **Temperature compensation**: Use sensors with built-in compensation
- **Avoid strain-sensitive mounting**: Don't mount on thin, flexible panels

#### Signal Noise (Post-Processing)
- **High-pass filter**: Remove DC and low-freq drift (<1 Hz)
- **Notch filter**: Remove electrical line noise (50/60 Hz + harmonics)
- **Averaging**: Average multiple spectra (reduce variance)

### 6. Environmental Considerations

#### Temperature
- **Sensor rating**: Ensure sensors rated for pump environment
- **Cable rating**: Avoid cable breakdown near hot surfaces
- **Thermal drift**: Calibrate at operating temperature

#### Moisture/Contamination
- **IP rating**: IP67 or higher for wet environments
- **Cable sealing**: Use glands, heat shrink
- **Sensor protection**: Boots, covers for sensor connector

#### Vibration from Adjacent Equipment
- **Isolate**: Ensure measured vibration is from target pump
- **Reference sensor**: Place sensor on ground/foundation to measure background

### 7. Calibration & Verification

#### Initial Setup
1. **Sensitivity check**: Apply known vibration (shaker table) or use back-to-back calibration
2. **Frequency response**: Verify flat response in 1 Hz - 5 kHz range
3. **Axis alignment**: Ensure X/Y/Z aligned with pump axes (radial/axial)
4. **Phase check**: Verify direction (use impact hammer, check sign)

#### Periodic Checks
- **Weekly**: Visual inspection of sensors, cables
- **Monthly**: Compare readings across sensors (check for drift)
- **Annually**: Full recalibration (send to lab or use portable calibrator)

### 8. Common Mistakes to Avoid

❌ **Don't**:
- Mount on flexible structures (panels, thin walls)
- Use long cables (>10 m) without IEPE sensors
- Ignore cable routing near EMI sources
- Use magnetic mount for permanent monitoring above 2 kHz
- Forget to ground sensors/DAQ
- Mount sensor with loose coupling
- Ignore sensor resonance frequency

✅ **Do**:
- Mount as close to source as possible
- Use stud mount for high frequencies
- Keep cables short, shielded, away from power
- Check sensor polarity/direction
- Document sensor locations (take photos)
- Perform sanity checks (compare sensors, check spectra)

---

## Summary: Quick Decision Guide

### Supervised Learning
**Small Dataset (<1k samples)**:
- Features: Time + Frequency (60-80 features)
- Model: Random Forest or SVM
- Validation: Stratified 5-fold CV

**Medium Dataset (1k-10k samples)**:
- Features: Time + Frequency + Wavelet (80-100 features)
- Model: XGBoost or Random Forest
- Validation: Group K-Fold (by recording)

**Large Dataset (>10k samples)**:
- Features: Raw signals or spectrograms
- Model: 1D CNN or 2D CNN
- Validation: Hold-out test set (20%)

### Unsupervised Learning
**Small Dataset (<1k normal samples)**:
- Model: One-Class SVM or Mahalanobis Distance
- Threshold: 95th percentile

**Medium Dataset (1k-5k normal samples)**:
- Model: Isolation Forest (baseline) + OC-SVM
- Ensemble: Average anomaly scores

**Large Dataset (>5k normal samples)**:
- Model: Autoencoder or LSTM-Autoencoder
- Threshold: 99th percentile reconstruction error

### Variable Operating Conditions
- **Few conditions**: Train separate models per condition
- **Many conditions**: Normalize features + add speed/pressure as features
- **Speed varies**: Use order tracking

### Deployment
- **Window**: 2.56 sec, 50% overlap
- **Decision**: 3 consecutive anomalies
- **Update**: Retrain monthly or after 1k new samples

---

## Next Steps

1. **Explore your data**: Load sample files, plot time-series + spectra
2. **Implement feature extraction**: Start with basic time/frequency features
3. **Baseline model**: Random Forest on features (supervised) or Isolation Forest (unsupervised)
4. **Iterate**: Add features, tune hyperparameters, try deep learning if data sufficient
5. **Deploy**: Start with offline batch processing, then move to online monitoring

**I can help you implement any part of this pipeline. What would you like to start with?**

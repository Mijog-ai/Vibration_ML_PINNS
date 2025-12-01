================================================================================
VIBRATION-BASED PUMP FAULT DETECTION USING MACHINE LEARNING
================================================================================

This repository contains a complete machine learning pipeline for detecting
faults in pumps using accelerometer vibration data.

QUICK START:
-----------
1. Install dependencies:
   pip install -r requirements.txt

2. Run the complete example:
   python examples/complete_example.py

3. See detailed documentation:
   - README.md - Full project documentation with examples
   - PUMP_FAULT_DETECTION_GUIDE.md - Comprehensive technical guide

REPOSITORY CONTENTS:
-------------------

📁 src/                      - Source code modules
   ├── feature_extraction.py        - Extract features from vibration signals
   ├── supervised_pipeline.py       - Fault classification (Random Forest, SVM, XGBoost)
   └── unsupervised_pipeline.py     - Anomaly detection (Isolation Forest, One-Class SVM)

📁 examples/                 - Example scripts
   └── complete_example.py          - End-to-end demonstration

📁 front housing/            - Sample vibration data
   ├── FRF_*.txt                    - Frequency Response Function data
   └── Coherence_*.txt              - Coherence measurements

📁 models/                   - Saved trained models (created after training)

📁 data/                     - Your vibration data goes here

📁 notebooks/                - Jupyter notebooks for interactive analysis

FEATURES:
---------
✓ Comprehensive feature extraction (time, frequency, wavelets, envelope analysis)
✓ Multiple ML algorithms (Random Forest, SVM, XGBoost, Isolation Forest, One-Class SVM)
✓ Both supervised and unsupervised approaches
✓ Real-time online monitoring support
✓ Production-ready code with full documentation

SUPPORTED FAULT TYPES:
---------------------
✓ Bearing faults
✓ Misalignment
✓ Imbalance
✓ Cavitation
✓ Piston wear
✓ Custom fault types

SENSOR SETUP:
------------
- Accelerometers: IEPE/ICP piezoelectric
- Sampling rate: 10.24 kHz (configurable)
- Mounting: Near bearings and load-carrying structures
- Directions: Axial and radial (X, Y, Z)

For detailed information, see README.md and PUMP_FAULT_DETECTION_GUIDE.md

================================================================================

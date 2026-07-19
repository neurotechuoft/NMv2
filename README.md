# NMv2
version two of Neurotech's NeuronMove project

# Parkinson's Tremor Detection Ensemble

This repository contains an end-to-end machine learning pipeline for detecting Parkinsonian tremors using tri-axial accelerometer data. The system utilizes an ensemble of Convolutional Neural Networks (CNNs) for feature extraction and Random Forest classifiers for final prediction.

## Project Overview

The objective of this project is to accurately differentiate between Parkinson's tremors and normal control movements. To achieve high accuracy and robustness across different patients and sensor profiles, the pipeline dynamically weights classes to handle severe data imbalance and utilizes a multi-model voting ensemble.

**Datasets Used:**
* `jiehu-rima`
* `kaggle-arushna`
* `kaggle-yunji`
*(Note: The `parkinsons-home` dataset was excluded due to high noise profiles negatively impacting ensemble performance).*

**Ensemble Performance (POOLED):**
* **Accuracy:** ~85.1%
* **Macro F1:** ~0.808
* **Precision:** ~0.788
* **Recall:** ~0.657

## Repository Structure

* `data_filtering.py`: Preprocesses raw accelerometer data (drift removal, bandpass filtering, windowing).
* `updated_train_processed.py`: Trains the feature-extracting CNNs (Shallow, Standard, DeepWide) and the final Random Forest classifier.
* `evaluate_ensemble.py`: Evaluates the trained models, calculates performance metrics, and dynamically tunes the activation thresholds.
* `updated_main.py`: The production inference script designed for edge devices (e.g., Raspberry Pi) to execute live predictions.
* `artifacts/`: Directory containing trained model weights, evaluation metrics, and `best_thresholds.csv`.

## Step-by-Step Workflow

### 1. Data Preparation
Place your raw accelerometer datasets into the `data/` directory. Run the filtering script to clean and window the data:
```python data_filtering.py```
### 2. Data Processing 
Run:
```
python3 kaggle_preprocess.py \
  --data_dir "Kaggle Set (Yunji)/patient_tremor_data" \
  --id_csv   "Kaggle Set (Yunji)/tremorPatientID.csv" \
  --out_dir  artifacts/kaggle-yunji

python3 Jiehu_preprocess.py \
  --csv_path "Jiehu Set (Rima)/df_all_timesteps.csv" \
  --out_dir  artifacts/jiehu-rima

python3 parkinsons_at_home_preprocess.py \
  --data_dir "formatted data" \
  --out_dir  artifacts/parkinsons-home
```
### 3. Model Training
```python updated_train_processed.py```

### Evaluation
```
python evaluate_ensemble.py
```

### Inference
WIP

### References
WIP

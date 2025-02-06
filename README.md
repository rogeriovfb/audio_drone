# Drone Fault Classification Project

## Overview
This project implements a pipeline for drone fault classification using audio recordings under various fault conditions. The solution includes feature extraction, spectrogram generation, and machine learning techniques such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

### Dataset
The dataset used in this project is sourced from the paper:  
**"SOUND-BASED DRONE FAULT CLASSIFICATION USING MULTITASK LEARNING"**  
*Authors: Wonjun Yi, Jung-Woo Choi, Jae-Woo Lee*  
Published by the Korea Advanced Institute of Science and Technology (KAIST).

- Paper: [arXiv:2304.11708](https://arxiv.org/abs/2304.11708)  
- Dataset: [Zenodo Dataset](https://zenodo.org/record/7779574#.ZCOvfXZBwQ8)

The dataset includes audio recordings of three drone models (A, B, C) in six maneuvering directions (F, B, L, R, C, CC) and nine fault conditions (N, MF1–MF4, PC1–PC4). Noise from various environments, such as construction sites and sports complexes, is added to simulate real-world conditions. 

![Dataset Distribution](figures/dataset_distribution.png)

---

## Getting Started

### Step 1: Download and Set Up the Dataset
1. Download the dataset from [Zenodo](https://zenodo.org/record/7779574#.ZCOvfXZBwQ8).
2. Extract the dataset to a desired location.
3. Update the `data_dir` variable in `utils.py` with the dataset path:
   ```python
   data_dir = "C:/path_to_your_dataset/Audio_drones"

### Step 2: Preprocessing

In this step, two datasets are prepared:
1. **Feature-Based Dataset**: For traditional machine learning models.
2. **CNN-Based Dataset**: Using spectrogram images.

#### Feature Extraction
Extract key audio features (e.g., MFCC, spectral centroid, spectral contrast) for traditional machine learning algorithms.

- **Run the Script**:  
  ```bash
  python Preprocessing/extract_features.py

The extracted features will be saved as audio_drone_features_extended.csv in the project root directory.

####Spectrogram Generation
Generate mel spectrogram images for CNN-based models.

- **Run the Script**:  
  ```bash
  python Preprocessing/create_spectrograms.py
  
Spectrogram images will be saved in the Audio_drones_spectrograms directory, organized as follows:
- By Drone Type: Subdirectories A, B, and C.
- By Dataset Split: Subdirectories train, valid, and test.

## Running the Models

This section provides instructions for running the three main approaches used in this project: classical machine learning algorithms, convolutional neural networks (CNNs), and recurrent neural networks (RNNs). Each approach is located in a dedicated directory:

- **Classical Machine Learning**: Located in the `classic_machine_learning` folder.
- **Convolutional Neural Networks (CNNs)**: Scripts for CNN models are in the `cnn` directory.
- **Recurrent Neural Networks (RNNs)**: Scripts for RNN models can be found in the `rnn` directory.

Detailed instructions for running the models will be provided soon.

---

## Results

The table below summarizes the performance metrics for the models evaluated in this study. Metrics include accuracy, recall, and F1-score for fault classification.

| **Model**           | **Accuracy** | **Recall** | **F1-Score** |
|---------------------|--------------|------------|--------------|
| AdaBoost            | 0.42         | 0.42       | 0.42         |
| Decision Tree       | 0.45         | 0.45       | 0.45         |
| Naive Bayes         | 0.29         | 0.28       | 0.28         |
| KNN                 | 0.58         | 0.58       | 0.58         |
| QDA                 | 0.53         | 0.49       | 0.49         |
| Logistic Regression | 0.36         | 0.36       | 0.37         |
| XGBOOST             | 0.63         | 0.63       | 0.63         |
| SVM                 | 0.68         | 0.68       | 0.68         |
| Random Forest       | 0.26         | 0.26       | 0.24         |
| GRU                 | 0.88         | 0.88       | 0.88         |



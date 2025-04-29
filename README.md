# 🧠 Predicting Age from Brain MRI Features

This project focuses on accurately **predicting a person's age based on their brain MRI data** using advanced machine learning techniques. Since the human brain undergoes structural changes as we age, MRI scans carry important patterns that can help estimate chronological age. The goal of this project is to develop a **robust regression model** that can generalize well on unseen data by processing and learning from these structural MRI features.

---

## 🚀 Overview

We applied a comprehensive pipeline that includes:
- Handling missing values
- Detecting and removing outliers
- Selecting relevant features from high-dimensional data
- Applying and optimizing regression models
- Building a final ensemble model using stacking

---

## 🧰 Tools & Technologies

- **Python** 🐍
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn** (for visualizations)
- **Isolation Forest**
- **Support Vector Regression (SVR)**
- **KNeighborsRegressor**
- **StackingRegressor**
- **PCA & KNNImputer**
- **Mutual Information for Feature Selection**

---

## 📊 Data Preprocessing

### 🔍 Outlier Detection

1. **Scaling**: Applied `RobustScaler` to reduce the effect of outliers.
2. **Missing Value Handling**: Used `KNNImputer` to fill missing values.
3. **Dimensionality Reduction**: Reduced features to 2D using PCA for visualization.
4. **Outlier Detection**: Implemented `IsolationForest` on PCA-transformed data.
5. **Results**: Identified and removed **61 outliers** from the original **1212 samples**.

<p align="center">
  <img src="Predicting Age from Brain MRI Features.png" alt="Outlier Detection with Isolation Forest" width="600"/>
</p>

---

### 🧩 Missing Value Imputation (Post-Outlier Removal)

- Standardized data using `StandardScaler`.
- Re-applied `KNNImputer` to address any residual missing values.

---

## 🧠 Feature Selection

The dataset originally contained **832 features**. We refined this using a multi-step approach:

1. **Mutual Information**: Selected top **600 features** most informative for age prediction.
2. **Correlation with Target**: Retained features with correlation coefficient > **0.09**, resulting in **204 features**.
3. **Inter-Feature Correlation**: Removed features with pairwise correlation > **0.9**, leaving us with a final set of **174 features**.

---

## 🤖 Regression Models

### ✅ Best Model: **SVR (RBF Kernel)**

- Tuned using `GridSearchCV`
- Best Parameters:
  - `C = 55.9`
  - `epsilon = 0.0001`

### 🔁 Final Model: **Stacking Regressor**

- **Base Learners**:
  - SVR (RBF Kernel)
  - KNeighborsRegressor (with `n_neighbors = 3`, `weights = 'distance'`)
- **Meta Learner**:
  - SVR (Linear Kernel)

This ensemble model provided the **best generalization** performance.

---

## 📈 Results

- Accurate age prediction using only MRI-derived structural features.
- Robust performance thanks to thoughtful feature engineering and ensemble modeling.
- Demonstrated the potential of MRI-based biomarkers in age prediction and neurodegenerative research.

---

## 📁 Project Structure

brain-age-prediction/
├── data/                   # Processed datasets
├── notebooks/              # Jupyter notebooks for each stage
├── models/                 # Saved model weights and pipelines
├── src/                    # Core Python scripts
│   ├── preprocessing.py        # Data cleaning and preprocessing
│   ├── feature_selection.py    # Feature selection techniques
│   ├── modeling.py             # Model training and evaluation
│   └── utils.py                # Utility functions
├── README.md              # Project description
└── requirements.txt       # Dependencies


---

## 📌 Future Work

- Experiment with deep learning models (e.g., CNNs for raw MRI scans).
- Validate model on external datasets.
- Investigate biological interpretability of selected features.

---

## 🙌 Acknowledgements

This project was inspired by the intersection of neuroscience and machine learning. Huge thanks to the open datasets and scikit-learn contributors that made this exploration possible.

---

## 🧪 Installation

```bash
https://github.com/k-abhi1/-Predicting-Age-from-Brain-MRI-Features
cd -Predicting-Age-from-Brain-MRI-Features
git clone https://github.com/k-abhi1/Predicting-Age-from-Brain-MRI-Features.git


---

Let me know if you want to include plots or performance metrics like MAE/RMSE, or if you want a version with Jupyter notebooks hosted on Google Colab!



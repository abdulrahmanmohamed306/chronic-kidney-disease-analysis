Chronic Kidney Disease Prediction & Analysis
An end-to-end data science project focused on cleaning medical records, exploring clinical patterns, and building a machine learning model to predict chronic kidney disease (CKD).

Overview
This project processes patient health data to handle missing values, uncover hidden correlations between clinical features, and train a robust classification model to assist in early detection.

Workflow
Data Cleaning & Imputation: Handled missing data using KNNImputer for numerical columns and mode imputation for categorical features.

Exploratory Data Analysis (EDA): Visualized target class distributions (ckd vs notckd), correlation heatmaps, and feature distributions using Seaborn and Matplotlib.

Preprocessing & Modeling: Applied LabelEncoder for categorical variables, StandardScaler for feature normalization, and trained a RandomForestClassifier.

Evaluation: Assessed model performance using accuracy metrics, confusion matrices, and 5-fold cross-validation to ensure stability.

Top Feature Importances
The Random Forest model identified these key clinical metrics as the most influential for prediction:

sg (Specific Gravity)

hemo (Hemoglobin)

pcv (Packed Cell Volume)

sc (Serum Creatinine)

rc (Red Blood Cell Count)

Tech Stack
Python

Pandas & NumPy

Scikit-Learn

Matplotlib & Seaborn

# Pima Indians Diabetes Risk Prediction
This project focuses on building a robust machine learning pipeline to predict the onset of diabetes based on diagnostic measurements. The analysis covers the full data science lifecycle, from exploratory data analysis (EDA) and rigorous data cleaning to model optimization and performance evaluation.

## Project Overview
The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to diagnostically predict whether or not a patient has diabetes based on specific medical indicators.
 
## Key Features Used:
Diagnostic: Glucose, Blood Pressure, Insulin, BMI, Skin Thickness.
Demographic: Age, Number of Pregnancies.
History: Diabetes Pedigree Function.

## Technical Workflow
### 1. Data Cleaning & Imputation
A critical part of this project was identifying that several columns contained 0 values that were biologically impossible (e.g., Blood Pressure or BMI).
The Solution: Instead of dropping these rows, I replaced 0 with NaN and imputed the median of the valid data. This preserved the dataset size while improving statistical accuracy.
### 2. Feature Scaling
Since algorithms like SVM and KNN are distance-based, I implemented StandardScaler to ensure all features were on the same scale, preventing features with larger ranges from dominating the model.
### 3. Hyperparameter Tuning
To move beyond default performance, I utilized RandomizedSearchCV for ensemble models (Random Forest and XGBoost). This allowed for the discovery of optimal tree depth, estimators, and learning rates.

## Model Performance
After testing and tuning multiple classification algorithms, the results were compared to identify the most reliable predictor:
Model     Accuracy (%)
XGBoost~75.00%
Logistic Regression~74.07%
SVM~71.30%
Random Forest~70.37%
- Note: While accuracy is a primary metric, the project also focuses on Recall to minimize False Negatives, which is vital in a medical context.

## How to Use
- Clone the repository: git clone https://github.com/YOUR_USERNAME/Diabetes-Risk-Prediction.git
- Install dependencies: pip install pandas numpy matplotlib seaborn scikit-learn xgboost
- Run the Jupyter Notebook: Diabetes_Risk_Analysis.ipynb

## Conclusion
The analysis demonstrates that ensemble methods like XGBoost provide the strongest predictive power for this dataset. The project highlights the importance of domain-specific data cleaning (handling impossible zeros) and the impact of feature scaling on model reliability.

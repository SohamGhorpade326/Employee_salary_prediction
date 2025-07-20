# =============================================================================
# FILE 1: model_training.py
# PURPOSE: To train the model and save it. Run this file only ONCE.
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import joblib # For saving the model

print("Starting model training...")

# --- 1. Load and Prepare Data ---
# Load the dataset
df = pd.read_csv('adult 3.csv')

# Replace '?' with NaN and drop rows with missing values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Separate target variable (y) from features (X)
X = df.drop('income', axis=1)
y = df['income']

# Convert target variable to binary (1 for >50K, 0 for <=50K)
y = y.apply(lambda x: 1 if x.strip() == '>50K' else 0)

# --- 2. Define Preprocessing ---
# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines for numerical and categorical features
# StandardScaler for numerical data, OneHotEncoder for categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# --- 3. Create and Train the Model Pipeline ---
# Define the model with the best performance
model = GradientBoostingClassifier(random_state=42)

# Create the full pipeline, including preprocessing and the model
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', model)])

# Train the pipeline on the ENTIRE dataset
full_pipeline.fit(X, y)

# --- 4. Save the Pipeline ---
# The saved file will contain the fitted preprocessor and the trained model
joblib.dump(full_pipeline, 'model.joblib')

print("Model training complete and pipeline saved to 'model.joblib'")

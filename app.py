import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Simulated dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Features
y = 3 * X.squeeze() + 4 + np.random.randn(100) * 2  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow experiment
mlflow.set_experiment("Simple Regression Experiment")

with mlflow.start_run():
    # Model creation and training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Model and metrics logged to MLflow")

# Viewing logs
print("Run this command to view logs:")
print("mlflow ui")

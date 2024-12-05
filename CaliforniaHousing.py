import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the California housing dataset
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

# Check the columns in the DataFrame
print("Columns in the dataset:", data.columns)

# Provide statistics for the specified columns
statistics = data[["AveRooms", "AveBedrms", "AveOccup", "Population"]].describe()
print(statistics)

# Split the dataset into features and target variable
# Check if 'MedHouseVal' is in the DataFrame
if 'MedHouseVal' in california_housing.target_names:
    X = data  # Features
    y = california_housing.target  # Target variable
else:
    print("Target variable 'MedHouseVal' not found in the dataset.")
    exit()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# ... existing code ...

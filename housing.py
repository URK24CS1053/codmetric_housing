# housing_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("housing.csv")  # replace with your CSV filename

# 2. Inspect the dataset
print(df.head())
print(df.info())

# 3. Handle missing values
# 'total_bedrooms' has missing values; fill with median
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# 4. Encode categorical variables
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# 5. Define features (X) and target (y)
target_column = 'median_house_value'
X = df.drop(target_column, axis=1)
y = df[target_column]

# 6. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# 10. Visualize predictions vs actual values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], color='red', linestyle='--')  # diagonal line
plt.show()

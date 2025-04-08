# regression_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Part 2: Data Preparation
# ----------------------------

# Load the cleaned dataset (generated from your earlier web scraping scripts)
df = pd.read_csv("cleaned_ebay_deals.csv", dtype=str)

# Remove rows with missing values in essential columns
df = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"])

# Convert numeric columns from strings to numbers
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")

# Function to convert shipping string values to numeric.
# For example, if the shipping is labeled as "Free shipping", we assign 0.0;
# otherwise, extract the numeric value.
def convert_shipping(x):
    if isinstance(x, str):
        if "free" in x.lower():
            return 0.0
        cleaned = re.sub(r'[^\d.-]', '', x.strip())
        return float(cleaned) if cleaned != "" else np.nan
    return x

df["shipping"] = df["shipping"].apply(convert_shipping)
df = df.dropna(subset=["shipping"])

# Plot the distribution of discount_percentage
plt.figure(figsize=(8, 6))
sns.histplot(df["discount_percentage"], bins=30, kde=True)
plt.title("Distribution of Discount Percentage")
plt.xlabel("Discount Percentage")
plt.ylabel("Frequency")
plt.show()

# ----------------------------
# Part 3: Binning and Balancing
# ----------------------------

# Create a temporary column "discount_bin" that divides discount_percentage into:
# - Low: 0% to 10%
# - Medium: 10% to 30%
# - High: greater than 30%
def assign_discount_bin(discount):
    if discount <= 10:
        return "Low"
    elif discount <= 30:
        return "Medium"
    else:
        return "High"

df["discount_bin"] = df["discount_percentage"].apply(assign_discount_bin)

# Display counts for each bin
print("Discount bin counts:")
print(df["discount_bin"].value_counts())

# Balance the dataset by under-sampling each bin to the number of samples in the smallest bin
min_count = df["discount_bin"].value_counts().min()
df_balanced = df.groupby("discount_bin").apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
print("Balanced dataset shape:", df_balanced.shape)

# Remove the discount_bin column because our target remains the original continuous discount percentage.
df_balanced = df_balanced.drop(columns=["discount_bin"])

# ----------------------------
# Part 4: Regression Modeling
# ----------------------------

# Use the following features: price, original_price, and shipping
features = ["price", "original_price", "shipping"]
X = df_balanced[features]
y = df_balanced["discount_percentage"]

# Split the data into 80% training and 20% testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model.
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions on the test set.
y_pred = model.predict(X_test)

# Evaluate the model using MAE, MSE, RMSE, and R² Score.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Regression Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Explanation:
# MAE is the average absolute difference between predictions and actual values.
# MSE is the average squared difference, which penalizes larger errors more.
# RMSE is the square root of MSE and gives error in the same units as the discount percentage.
# R² indicates the proportion of the variance in the target explained by the model.

# ----------------------------
# Part 5: Visual Evaluation
# ----------------------------

# Scatter plot: Actual vs. Predicted Discount Percentage.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Discount Percentage")
plt.ylabel("Predicted Discount Percentage")
plt.title("Actual vs Predicted Discount Percentage")
plt.show()

# Residual plot: Error distribution (Actual - Predicted).
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Discount Percentage")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# apply_model.py
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to convert the shipping string to numeric.
def convert_shipping(x):
    if isinstance(x, str):
        if "free" in x.lower():
            return 0.0
        cleaned = re.sub(r'[^\d.-]', '', x.strip())
        return float(cleaned) if cleaned != "" else np.nan
    return x

# Load the cleaned dataset.
df = pd.read_csv("cleaned_ebay_deals.csv", dtype=str)

# Remove rows with missing essential values.
df = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"])

# Convert the price, original_price, and discount_percentage columns to numeric.
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")

# Process the shipping column.
df["shipping"] = df["shipping"].apply(convert_shipping)
df = df.dropna(subset=["shipping"])

# For consistency, create the discount_bin column and balance the dataset (as done previously).
def assign_discount_bin(discount):
    if discount <= 10:
        return "Low"
    elif discount <= 30:
        return "Medium"
    else:
        return "High"

df["discount_bin"] = df["discount_percentage"].apply(assign_discount_bin)
min_count = df["discount_bin"].value_counts().min()
df_balanced = df.groupby("discount_bin").apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
df_balanced = df_balanced.drop(columns=["discount_bin"])

# Define the features.
features = ["price", "original_price", "shipping"]
X = df_balanced[features]
y = df_balanced["discount_percentage"]

# Split and train a Linear Regression model (using the same procedure as in regression_model.py).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Part 6: Applying the Model to Incomplete Data
# ----------------------------

# Simulate incomplete data by removing the discount_percentage column.
df_incomplete = df_balanced.drop(columns=["discount_percentage"])

# Randomly select 20 products from the incomplete dataset.
df_subset = df_incomplete.sample(n=20, random_state=42)

# Use the trained model to predict the discount percentage for these products.
X_new = df_subset[features]
predicted_discount = model.predict(X_new)

# Add the predictions to the DataFrame.
df_subset["Predicted_Discount (%)"] = predicted_discount

# Create and display the final result table with the selected columns.
result_table = df_subset[["title", "price", "original_price", "shipping", "Predicted_Discount (%)"]]
print("Predicted Discount for 20 Randomly Selected Products:")
print(result_table.to_string(index=False))

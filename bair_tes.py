# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # For scaling data
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
file_path = "uneguiData.csv"  # Path to your uploaded file
data = pd.read_csv(file_path)

# Data cleaning and preprocessing
# Convert "Үнэ" column to numeric by removing commas
data["Үнэ"] = data["Үнэ"].str.replace(",", "").astype(int)

# Add new features
data["Тагтны тоо"] = data["Тагт"].str.extract(r"(\d+)").fillna(0).astype(int)
data["9-өөс доош давхар"] = (data["Барилгын давхар"] <= 9).astype(int)

# Drop unnecessary columns
data_cleaned = data.drop(columns=["ID", "Огноо", "Тагт"])

# Encode categorical variables
categorical_columns = [
    "Байршил",
    "Барилгын явц",
    "Гараж",
    "Төлбөрийн нөхцөл",
    "Хаалга",
    "Цонх",
    "Шал",
]
data_encoded = pd.get_dummies(
    data_cleaned, columns=categorical_columns, drop_first=True
)

# Split data into features (X) and target (y)
X = data_encoded.drop(columns=["Үнэ"])  # All columns except "Үнэ" for feature set
y = data_encoded["Үнэ"]  # Target variable

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# Plot actual vs predicted prices with improvements
# Plot actual vs predicted prices using all data
def plot_actual_vs_predicted_prices_full(data, model, scaler):
    plt.figure(figsize=(12, 6))

    # Scale all data features
    X_all = data.drop(columns=["Үнэ"])
    X_all_scaled = scaler.transform(X_all)

    # Predict on all data
    y_all_pred = model.predict(X_all_scaled)

    # Plot actual prices
    plt.plot(
        range(len(data)),
        data["Үнэ"] / 1_000_000,
        label="Actual Price (in million)",
        color="blue",
        marker="o",
        markersize=2,
        linestyle="-",
    )

    # Plot predicted prices
    plt.plot(
        range(len(data)),
        y_all_pred / 1_000_000,
        label="Predicted Price (in million)",
        color="red",
        marker="x",
        markersize=2,
        linestyle="--",
    )

    plt.title("Actual vs Predicted Prices for All Data")
    plt.xlabel("Properties (All Rows)")
    plt.ylabel("Price (Million ₮)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


# Call the function with the entire dataset
plot_actual_vs_predicted_prices_full(data_encoded, model, scaler)

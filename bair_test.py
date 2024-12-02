# Import libraries
import pandas as pd
import numpy as np  # Add this line

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt  # For plotting graphs
import openpyxl  # To handle Excel files

# Load dataset
file_path = "uneguiData.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Data cleaning and preprocessing
# Convert "Үнэ" column to numeric by removing commas
data["Үнэ"] = data["Үнэ"].str.replace(",", "").astype(int)

# Add new features based on rules
data["Тагтны тоо"] = data["Тагт"].str.extract(r"(\d+)").fillna(0).astype(int)
data["9-өөс доош давхар"] = (data["Барилгын давхар"] <= 9).astype(int)

# Drop unnecessary columns
data_cleaned = data.drop(columns=["ID", "Огноо", "Тагт"])

# Encode categorical variables into numerical values
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
X = data_encoded.drop(columns=["Үнэ"])
y = data_encoded["Үнэ"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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


# Function to check coefficients of the model
def check_location_coefficients(model, X_train):
    coefficients = pd.DataFrame(
        {"Feature": X_train.columns, "Coefficient": model.coef_}
    ).sort_values(by="Coefficient", ascending=False)
    print("\n--- Feature Coefficients ---")
    print(coefficients)


# Function to check location significance
def check_location_significance(X, y):
    # Ensure all data is numeric
    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_columns) > 0:
        print("Non-numeric columns found:", non_numeric_columns)
        X = pd.get_dummies(X, drop_first=True)
        print("Converted non-numeric columns to numeric using get_dummies.")

    if not np.issubdtype(y.dtype, np.number):
        y = pd.to_numeric(y, errors="coerce")
        print("Converted y to numeric.")

    # Ensure all data is correctly converted to numpy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Add constant for intercept
    X_with_constant = sm.add_constant(X)

    # Fit OLS model
    stats_model = sm.OLS(y, X_with_constant).fit()

    print("\n--- Статистик шинжилгээний дүн ---")
    print(stats_model.summary())  # Display full results


# Check coefficients
check_location_coefficients(model, X_train)

# Check significance
check_location_significance(X_train, y_train)


# Function to align new data to match training data structure
def align_features(new_data, reference_data):
    """
    Align the features of new data to match the training data structure.
    Missing columns will be added with default value 0.
    Columns not in the training data will be removed.
    """
    missing_cols = set(reference_data.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0  # Add missing columns with default value 0
    # Reorder columns to match the training data
    new_data = new_data[reference_data.columns]
    return new_data


# Example: Predict for new data
new_data = pd.DataFrame(
    {
        "Байршил": ["СБД"],  # Example district
        "Ашиглалтанд орсон он": [2024],
        "Барилгын давхар": [5],
        "Талбай": [50],
        "Хэдэн давхарт": [4],
        "Цонхны тоо": [4],
        "Барилгын явц": ["Ашиглалтад орсон"],
        "Гараж": ["Байхгүй"],
        "Тагтны тоо": [2],
        "9-өөс доош давхар": [1],
        "Төлбөрийн нөхцөл": ["Лизинггүй"],
        "Хаалга": ["Төмөр"],
        "Цонх": ["Мод"],
        "Шал": ["Паркет"],
    }
)
new_data_file = "new_data.xlsx"  # Replace with your file path
new_data = pd.read_excel(new_data_file)
new_data_encoded = pd.get_dummies(
    new_data,
    columns=[
        "Байршил",
        "Барилгын явц",
        "Гараж",
        "Төлбөрийн нөхцөл",
        "Хаалга",
        "Цонх",
        "Шал",
    ],
)
new_data_aligned = align_features(new_data_encoded, X)

# Predict the price for new data
predicted_prices = model.predict(new_data_aligned)

# Add predicted prices to the new_data dataframe
new_data["Таамагласан үнэ"] = predicted_prices


# Plot actual vs predicted prices
def plot_actual_vs_predicted_prices(new_data):
    plt.figure(figsize=(12, 8))

    # Convert prices to millions for easier display
    new_data["Үнэ (сая)"] = new_data["Үнэ"] / 1_000_000
    new_data["Таамагласан үнэ (сая)"] = new_data["Таамагласан үнэ"] / 1_000_000

    # Plot actual prices
    plt.plot(
        new_data.index,
        new_data["Үнэ (сая)"],
        label="Зарын үнэ (сая төгрөг)",
        marker="o",
        linestyle="-",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )

    # Plot predicted prices
    plt.plot(
        new_data.index,
        new_data["Таамагласан үнэ (сая)"],
        label="Таамагласан үнэ (сая төгрөг)",
        marker="x",
        linestyle="--",
        color="red",
        linewidth=2,
        alpha=0.8,
    )

    # Highlight differences for some points (add annotations)
    for i in range(0, len(new_data), max(1, len(new_data) // 10)):
        plt.annotate(
            f"{new_data['Үнэ (сая)'].iloc[i]:.1f}M\n{new_data['Таамагласан үнэ (сая)'].iloc[i]:.1f}M",
            (new_data.index[i], new_data["Үнэ (сая)"].iloc[i]),
            textcoords="offset points",
            xytext=(10, 10),
            ha="center",
            fontsize=9,
            arrowprops=dict(facecolor="gray", arrowstyle="->", alpha=0.5),
        )

    # Styling
    plt.title(
        "Зарын үнэ ба Таамагласан үнэ",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Орон сууцны үнэ тооцоолох модел", fontsize=14)
    plt.ylabel("Үнэ (сая төгрөг)", fontsize=14)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()


# Call the function
plot_actual_vs_predicted_prices(new_data)


# Save updated dataframe to a new Excel file
new_data.to_excel("new_data_with_predictions.xlsx", index=False)
print("New data with predictions saved to 'new_data_with_predictions.xlsx'")

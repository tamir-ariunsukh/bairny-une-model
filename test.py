# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "uneguiData.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Data cleaning and preprocessing
# Convert "Үнэ" column to numeric by removing commas
data["Үнэ"] = data["Үнэ"].str.replace(",", "").astype(int)

# Drop unnecessary columns
data_cleaned = data.drop(columns=["ID", "Огноо"])

# Encode categorical variables into numerical values
categorical_columns = [
    "Байршил",
    "Барилгын явц",
    "Гараж",
    "Тагт",
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

# Train the Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
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
        "Тагт": ["2 тагттай"],
        "Төлбөрийн нөхцөл": ["Лизинггүй"],
        "Хаалга": ["Төмөр"],
        "Цонх": ["Мод"],
        "Шал": ["Паркет"],
    }
)

# Encode new data
new_data_encoded = pd.get_dummies(
    new_data,
    columns=[
        "Байршил",
        "Барилгын явц",
        "Гараж",
        "Тагт",
        "Төлбөрийн нөхцөл",
        "Хаалга",
        "Цонх",
        "Шал",
    ],
)

# Align new data to match training data structure
new_data_aligned = align_features(new_data_encoded, X)

# Predict the price
predicted_price = model.predict(new_data_aligned)
print(f"Predicted Price: {predicted_price[0]}")

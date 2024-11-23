# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load and preprocess dataset
def load_and_preprocess_data(file_path):
    """
    Load dataset, clean and preprocess it for training.
    """
    data = pd.read_csv(file_path)

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

    return data_encoded


# Train the model
def train_model(X, y):
    """
    Train a Linear Regression model and return it.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5

    print(f"Model Performance:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")

    return model, X_train


# Align new data features
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


# Predict for new data
def predict_new_data(new_data, model, reference_data):
    """
    Preprocess, align, and predict price for new data.
    """
    # Encode new data
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
    new_data_encoded = pd.get_dummies(new_data, columns=categorical_columns)

    # Align with training data structure
    new_data_aligned = align_features(new_data_encoded, reference_data)

    # Predict the price
    predicted_price = model.predict(new_data_aligned)
    return predicted_price[0]


# Main workflow
file_path = "uneguiData.csv"  # Update with your file path
data = load_and_preprocess_data(file_path)

# Split data into features (X) and target (y)
X = data.drop(columns=["Үнэ"])
y = data["Үнэ"]

# Train the model
model, reference_data = train_model(X, y)

# Example new data
new_data = pd.DataFrame(
    {
        "Байршил": ["БГД"],  # Example district
        "Ашиглалтанд орсон он": [2024],
        "Барилгын давхар": [5],
        "Талбай": [30],
        "Хэдэн давхарт": [4],
        "Цонхны тоо": [3],
        "Барилгын явц": ["Ашиглалтад орсон"],
        "Гараж": ["Байхгүй"],
        "Тагт": ["2 тагттай"],
        "Төлбөрийн нөхцөл": ["Лизинггүй"],
        "Хаалга": ["Төмөр"],
        "Цонх": ["Мод"],
        "Шал": ["Паркет"],
    }
)

# Predict the price
predicted_price = predict_new_data(new_data, model, X)
print(f"Predicted Price: {predicted_price}")

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

# Load dataset
file_path = "uneguiData.csv"  # Update with your file path
data = pd.read_csv(file_path)


@app.route("/", methods=["GET", "POST"])
def index():
    # Get unique values for dropdowns
    unique_years = sorted(data["Ашиглалтанд орсон он"].dropna().unique().tolist())
    unique_floors = sorted(data["Барилгын давхар"].dropna().unique().tolist())
    unique_progress = sorted(data["Барилгын явц"].dropna().unique().tolist())
    unique_garage = sorted(data["Гараж"].dropna().unique().tolist())
    unique_location = sorted(data["Байршил"].dropna().unique().tolist())
    unique_balcony = sorted(data["Тагт"].dropna().unique().tolist())
    unique_payment = sorted(data["Төлбөрийн нөхцөл"].dropna().unique().tolist())
    unique_door = sorted(data["Хаалга"].dropna().unique().tolist())
    unique_window = sorted(data["Цонх"].dropna().unique().tolist())
    unique_floor_type = sorted(data["Шал"].dropna().unique().tolist())

    filtered_data = None  # Initialize filtered_data

    if request.method == "POST":
        selected_year = request.form.get("year")
        selected_floor = request.form.get("floor")
        selected_progress = request.form.get("progress")
        selected_garage = request.form.get("garage")
        selected_location = request.form.get("location")
        selected_balcony = request.form.get("balcony")
        area = request.form.get("area")  # Numeric input for area
        selected_payment = request.form.get("payment")
        selected_door = request.form.get("door")
        num_floors = request.form.get(
            "num_floors"
        )  # Numeric input for number of floors
        selected_window = request.form.get("window")
        num_windows = request.form.get("num_windows")  # Dropdown for number of windows
        selected_floor_type = request.form.get("floor_type")

        # Here you can implement filtering logic based on selected values
        # For now, just displaying the selected values as filtered_data
        filtered_data = {
            "year": selected_year,
            "floor": selected_floor,
            "progress": selected_progress,
            "garage": selected_garage,
            "location": selected_location,
            "balcony": selected_balcony,
            "area": area,
            "payment": selected_payment,
            "door": selected_door,
            "num_floors": num_floors,
            "window": selected_window,
            "num_windows": num_windows,
            "floor_type": selected_floor_type,
        }

    return render_template(
        "index.html",
        unique_years=unique_years,
        unique_floors=unique_floors,
        unique_progress=unique_progress,
        unique_garage=unique_garage,
        unique_location=unique_location,
        unique_balcony=unique_balcony,
        unique_payment=unique_payment,
        unique_door=unique_door,
        unique_window=unique_window,
        unique_floor_type=unique_floor_type,
        filtered_data=filtered_data,
    )


@app.route("/predict", methods=["POST"])
def predict():
    selected_year = request.form.get("year")
    selected_floor = request.form.get("floor")
    selected_progress = request.form.get("progress")
    selected_garage = request.form.get("garage")
    selected_location = request.form.get("location")
    selected_balcony = request.form.get("balcony")
    area = request.form.get("area")
    selected_payment = request.form.get("payment")
    selected_door = request.form.get("door")
    num_floors = request.form.get("num_floors")
    selected_window = request.form.get("window")
    num_windows = request.form.get("num_windows")
    selected_floor_type = request.form.get("floor_type")

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
            "Ашиглалтанд орсон он": [selected_year],
            "Барилгын давхар": [selected_floor],
            "Барилгын явц": [selected_progress],
            "Гараж": [selected_garage],
            "Байршил": [selected_location],
            "Тагт": [selected_balcony],
            "Талбай": [area],
            "Төлбөрийн нөхцөл": [selected_payment],
            "Хаалга": [selected_door],
            "Хэдэн давхарт": [num_floors],
            "Цонхны тоо": [num_windows],
            "Цонх": [selected_window],
            "Шал": [selected_floor_type],
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

    return jsonify({"predicted_price": "{:,.0f}".format(predicted_price[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

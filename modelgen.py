import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# === 1. Өгөгдөл боловсруулах ===

IMG_SIZE = 64
data_dir = "dataset"
categories = ["EyesOpen", "EyesClosed"]

data, labels = [], []

for category in categories:
    category_path = os.path.join(data_dir, category)
    class_label = categories.index(category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(class_label)
        except Exception as e:
            print(f"Алдаа: {e}")

data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

datagen.fit(X_train)

# === 2. Моделийн архитектур ===

model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Conv2D(256, (3, 3), activation="relu"),  # Нэмсэн давхарга
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),  # Нэмсэн давхарга
        Dense(len(categories), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# === Callback тохируулах ===
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
model_checkpoint = ModelCheckpoint(
    "bbest_eye_state_model.keras", save_best_only=True, monitor="val_loss", verbose=1
)

# === 3. Сургалт ===
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stop, model_checkpoint],
    verbose=1,
)

# === 4. Үр дүнг графикаар дүрслэх ===
plt.figure(figsize=(12, 6))

# Алдагдал
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Сургалтын алдагдал")
plt.plot(history.history["val_loss"], label="Шалгалтын алдагдал")
plt.legend()
plt.title("Алдагдал")

# Нарийвчлал
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Сургалтын нарийвчлал")
plt.plot(history.history["val_accuracy"], label="Шалгалтын нарийвчлал")
plt.legend()
plt.title("Нарийвчлал")

plt.show()

print("Модел хадгалагдсан: 'bbest_eye_state_model.keras'")

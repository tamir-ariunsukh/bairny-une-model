import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# === best_eye_state_model module ачааллах ===
try:
    eye_model = load_model("best_eye_state_model.keras")
    print("Загвар амжилттай ачаалагдлаа.")
except Exception as e:
    raise Exception(f"Загвар ачаалахад алдаа гарлаа: {e}")

# === Mediapipe-ийн тохиргоо ===
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

# === Зураг унших ===
image_path = "test.png"  # Шалгах зургийн зам
frame = cv2.imread(image_path)
if frame is None:
    raise Exception("Зураг олдсонгүй. Замыг зөв шалгана уу.")

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)


# === Толгойн байрлалыг тооцоолох функц ===
def calculate_head_position(landmarks):
    nose_tip = landmarks[1]  # Хамрын үзүүр
    chin = landmarks[152]  # Эрүүний үзүүр
    left_cheek = landmarks[234]  # Зүүн хацар
    right_cheek = landmarks[454]  # Баруун хацар

    vertical_difference = abs(nose_tip.y - chin.y)
    horizontal_difference = abs(left_cheek.x - right_cheek.x)

    return vertical_difference, horizontal_difference


# === Нүдний төлөв шалгах ===
def check_eye_status(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Загварт тохируулах хэмжээ
    input_frame = np.expand_dims(resized_frame, axis=(0, -1)) / 255.0
    prediction = eye_model.predict(input_frame, verbose=0)[0][0]
    return "Eyes Open" if prediction > 0.5 else "Eyes Closed"


# === Илрүүлэлт хийх ===
head_status = "Head Up (Awake)"
eye_status = "Eyes Open"

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,  # Шаардлагатай атрибутыг энд оруулна
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
        )
        vertical_diff, horizontal_diff = calculate_head_position(
            face_landmarks.landmark
        )
        if vertical_diff > 0.1:  # Толгой доош хазайсан
            head_status = "Head Down (Sleeping)"
        elif horizontal_diff < 0.05:  # Толгой хажуу тийш хазайсан
            head_status = "Head Tilted (Sleeping)"
        break

eye_status = check_eye_status(frame)

# === Нийт төлөв тодорхойлох ===
status = (
    "Sleeping"
    if head_status in ["Head Down (Sleeping)", "Head Tilted (Sleeping)"]
    or eye_status == "Eyes Closed"
    else "Awake"
)

# === Төлөв дэлгэц дээр харуулах ===
cv2.putText(
    frame,
    f"{head_status} | {eye_status}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2,
)
cv2.putText(
    frame,
    f"Overall Status: {status}",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2,
)

cv2.imshow("Static Image Detector", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

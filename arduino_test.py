import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# === Загвар ачаалах ===
try:
    eye_model = load_model("best_eye_state_model.keras")
    print("Загвар амжилттай ачаалагдлаа.")
except Exception as e:
    raise Exception(f"Загвар ачаалахад алдаа гарлаа: {e}")

# === Mediapipe-ийн тохиргоо ===
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)


# === Нүдний төлөв шалгах функц ===
def check_eye_status(frame, eye_coords):
    try:
        x_min = max(0, int(min([coord.x for coord in eye_coords]) * frame.shape[1]) - 5)
        x_max = min(
            frame.shape[1],
            int(max([coord.x for coord in eye_coords]) * frame.shape[1]) + 5,
        )
        y_min = max(0, int(min([coord.y for coord in eye_coords]) * frame.shape[0]) - 5)
        y_max = min(
            frame.shape[0],
            int(max([coord.y for coord in eye_coords]) * frame.shape[0]) + 5,
        )

        eye_region = frame[y_min:y_max, x_min:x_max]
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        resized_eye = cv2.resize(gray_eye, (64, 64))
        input_eye = np.expand_dims(resized_eye, axis=(0, -1)) / 255.0
        prediction = eye_model.predict(input_eye, verbose=0)[0][1]
        return prediction > 0.5  # True бол "Eyes Open", False бол "Eyes Closed"
    except Exception as e:
        print(f"Нүдний төлөвийг шалгах алдаа: {e}")
        return False


# === Зураг унших ===
image_path = "test.png"  # Шалгах зургийн зам
frame = cv2.imread(image_path)
if frame is None:
    raise Exception("Зураг олдсонгүй. Замыг зөв шалгана уу.")

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)

# === Илрүүлэлт хийх ===
head_status = "Head Up (Awake)"
eye_status = "Unknown"
status = "Unknown"

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
        )
        landmarks = face_landmarks.landmark

        # Зүүн болон баруун нүдний координат
        left_eye_coords = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eye_coords = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]

        left_eye_open = check_eye_status(frame, left_eye_coords)
        right_eye_open = check_eye_status(frame, right_eye_coords)

        # Толгойн байрлалыг шалгах
        nose_tip = landmarks[1]
        chin = landmarks[152]
        vertical_difference = abs(nose_tip.y - chin.y)
        if vertical_difference < 0.1:
            head_status = "Head Down (Sleeping)"
        else:
            head_status = "Head Up (Awake)"

        # Нийт төлөв тодорхойлох
        if left_eye_open or right_eye_open:
            eye_status = "Eyes Open"
            status = "Awake"
        else:
            eye_status = "Eyes Closed"
            status = "Sleeping"

        break  # Нэг нүүр илрүүлээд зогсоох

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

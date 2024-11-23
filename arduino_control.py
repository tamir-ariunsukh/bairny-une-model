import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import serial
import time

# === Arduino-той холболт ===
try:
    arduino = serial.Serial(port="COM1", baudrate=9600, timeout=0.1)
    time.sleep(2)  # Arduino-г эхлүүлэх хугацаа
    print("Arduino амжилттай холбогдлоо.")
except Exception as e:
    print(f"Arduino холболт амжилтгүй: {e}")
    arduino = None  # Arduino байхгүй тохиолдолд код үргэлжлэх боломжтой болгоно

# === best_eye_state_model module ачааллах ===
try:
    eye_model = load_model("best_eye_state_model.keras")
    print("Загвар амжилттай ачаалагдлаа.")
except Exception as e:
    raise Exception(f"Загвар ачаалахад алдаа гарлаа: {e}")

# === Mediapipe-ийн тохиргоо ===
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# === Камераг эхлүүлэх ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Камер нээгдэхгүй байна. Камераа шалгана уу.")


# === Толгойн байрлалыг тооцоолох функц ===
def calculate_head_position(landmarks):
    nose_tip = landmarks[1]  # Хамрын үзүүр
    chin = landmarks[152]  # Эрүүний үзүүр
    left_cheek = landmarks[234]  # Зүүн хацар
    right_cheek = landmarks[454]  # Баруун хацар

    vertical_difference = abs(nose_tip.y - chin.y)
    horizontal_difference = abs(left_cheek.x - right_cheek.x)

    return vertical_difference, horizontal_difference


# === Arduino руу дохио илгээх функц ===
def send_to_arduino(command):
    if arduino:
        try:
            arduino.write(bytes(command, "utf-8"))
            time.sleep(0.05)
        except Exception as e:
            print(f"Arduino-д өгөгдөл илгээхэд алдаа гарлаа: {e}")


# === Бодит цагийн илрүүлэлт ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Камераас зураг уншиж чадсангүй.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # === Толгойн байрлалыг шалгах ===
    head_status = "Head Up (Awake)"
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
            )
            vertical_diff, horizontal_diff = calculate_head_position(
                face_landmarks.landmark
            )
            if vertical_diff > 0.1:  # Толгой доош хазайсан
                head_status = "Head Down (Sleeping)"
                break
            elif horizontal_diff < 0.05:  # Толгой хажуу тийш хазайсан
                head_status = "Head Tilted (Sleeping)"
                break

    # === Нүдний төлөв шалгах ===
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Загварт тохируулах хэмжээ
    input_frame = np.expand_dims(resized_frame, axis=(0, -1)) / 255.0
    prediction = eye_model.predict(input_frame, verbose=0)[0][0]

    eye_status = "Eyes Open" if prediction > 0.5 else "Eyes Closed"

    # === Нийт төлөв тодорхойлох ===
    if (
        head_status in ["Head Down (Sleeping)", "Head Tilted (Sleeping)"]
        or eye_status == "Eyes Closed"
    ):
        status = "Sleeping"
        send_to_arduino("1")  # Arduino руу унтаж байгааг илгээх
    else:
        status = "Awake"
        send_to_arduino("0")  # Arduino руу сэрүүн байгааг илгээх

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
    cv2.imshow("Combined Detector", frame)

    # === 'q' дарвал гарах ===
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

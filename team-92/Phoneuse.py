import cv2
import pygame
import pyttsx3
import time
import threading
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure this file exists

# Initialize Pygame for sound alert
pygame.mixer.init()
alert_sound = "alert.wav"  # Ensure this file exists
pygame.mixer.music.load(alert_sound)

# Initialize text-to-speech
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

# Function to play alert
def speak_warning():
    tts_engine.say("Avoid phone while driving. Focus on driving.")
    tts_engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)

# Class IDs to detect (Phones, Screens, Remotes, Other Devices)
DETECTED_CLASS_IDS = {38, 43, 65, 66, 67, 73, 30, 42}  # Added 30, 42
ALWAYS_DETECT_IDS = {67}  # Detect at all confidence levels (0.00 - 0.99)

last_alert_time = 0  # Prevent spam alerts
last_detection_time = 0
device_visible = False
device_boxes = []  # Store all detected device boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    results = model(frame)
    object_detected = False
    device_boxes.clear()

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])

            # Always detect IDs 28 & 67 (ignore confidence), others need confidence > 0.4
            if class_id in ALWAYS_DETECT_IDS or (class_id in DETECTED_CLASS_IDS and confidence > 0.4):
                object_detected = True
                last_detection_time = current_time
                device_visible = True  # Keep the device visible for 3 seconds
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Adjust bounding box to better fit edges
                padding = 5
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)

                device_boxes.append((x1, y1, x2, y2))

                # Alert every 5 seconds
                if current_time - last_alert_time > 5:
                    pygame.mixer.music.play()
                    threading.Thread(target=speak_warning, daemon=True).start()
                    last_alert_time = current_time

    # Keep detected devices visible for 3 seconds
    if device_visible and (current_time - last_detection_time <= 3):
        for (x1, y1, x2, y2) in device_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Device Detected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        device_visible = False  # Hide after 3 seconds

    cv2.imshow("Device Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' press
        break

cap.release()
cv2.destroyAllWindows()

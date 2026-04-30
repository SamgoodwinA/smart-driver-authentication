import cv2
import time
import pyttsx3
import pygame
import numpy as np

# Load pre-trained Haar cascades for face and eyes detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize pygame for playing sound
pygame.mixer.init()

# Load the alert sound (ensure alert.wav is present)
alert_sound_path = 'alert.wav'
try:
    alert_sound = pygame.mixer.Sound(alert_sound_path)
except Exception as e:
    print(f"Error loading sound: {e}")
    alert_sound = None

# Start webcam with optimized settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduced buffer size for less lag

# Variables to track sleep detection
face_detected_start = None  
eye_closed_start = None    
last_alert_time = 0        
sleep_detected = False

def adjust_gamma(image, gamma=1.0):
    """Apply gamma correction."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_preprocessing(frame):
    """Apply contrast enhancement and gamma correction if needed."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    processed_l = clahe.apply(l_channel)
    lab[:, :, 0] = processed_l
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    avg_brightness = cv2.mean(gray)[0]

    if avg_brightness < 60:
        return adjust_gamma(enhanced, 1.6)
    elif avg_brightness > 180:
        return adjust_gamma(enhanced, 0.8)
    return enhanced

def is_blurry(image, threshold=80):
    """Check if image is blurry using variance of Laplacian."""
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < threshold

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # Skip every alternate frame to improve performance
    if frame_count % 2 != 0:
        cv2.imshow("Sleep Detection", frame)
        continue

    processed_frame = adaptive_preprocessing(frame)
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(120, 120), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        if face_detected_start is None:
            face_detected_start = current_time
        face_duration = current_time - face_detected_start

        # Select largest detected face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        roi_gray = gray[y:y + h, x:x + w]

        # Only check eyes if face has been detected for at least 3 seconds
        if face_duration >= 3:
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))

            valid_eyes = [eye for eye in eyes if eye[1] < h / 2 and eye[2] > 15 and eye[3] > 15]

            # Ensure eyes are aligned (avoid misdetections)
            if len(valid_eyes) >= 2 and abs(valid_eyes[0][1] - valid_eyes[1][1]) > 25:
                valid_eyes = valid_eyes[:1]

            # Check if eyes are closed
            if len(valid_eyes) < 2 and not is_blurry(processed_frame):
                if eye_closed_start is None:
                    eye_closed_start = current_time
                eye_closed_duration = current_time - eye_closed_start

                if eye_closed_duration >= 3:
                    if not sleep_detected and (current_time - last_alert_time) > 10:
                        if alert_sound:
                            try:
                                alert_sound.play()
                            except Exception as e:
                                print(f"Error playing sound: {e}")

                        print("Person may be sleeping.")
                        engine.say("Warning, stop the car and take rest.")
                        engine.runAndWait()
                        sleep_detected = True
                        last_alert_time = current_time
            else:
                eye_closed_start = None
                sleep_detected = False

        # Draw rectangles
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if face_duration >= 3:
            for (ex, ey, ew, eh) in valid_eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Display face detection timer
        cv2.putText(frame, f"Face Detected: {int(face_duration)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        face_detected_start = None
        eye_closed_start = None
        sleep_detected = False

    # Display info
    cv2.putText(frame, 'Detection: Face & Eyes', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sleep Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

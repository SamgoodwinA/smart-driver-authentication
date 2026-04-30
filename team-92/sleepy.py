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

# Load the alert sound (make sure alert.wav is in the same directory or provide full path)
alert_sound_path = 'alert.wav'  # Update this path if necessary
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
face_detected_start = None  # NEW: Track face detection start time
eye_closed_start = None     # NEW: Track eye closure start time
last_alert_time = 0         # Cooldown tracker
sleep_detected = False

def adjust_gamma(image, gamma=1.0):
    """Improved performance version"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_preprocessing(frame):
    """Optimized preprocessing pipeline"""
    # Convert to LAB and apply faster CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Use faster CLAHE parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    processed_l = clahe.apply(l_channel)
    lab[:,:,0] = processed_l
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Quick brightness check
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    avg_brightness = cv2.mean(gray)[0]
    
    # Apply gamma only if needed
    if avg_brightness < 60:
        return adjust_gamma(enhanced, 1.6)  # Reduced gamma for speed
    elif avg_brightness > 180:
        return adjust_gamma(enhanced, 0.8)  # Adjusted gamma
    return enhanced

def is_blurry(image, threshold=80):
    """Optimized blur check"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    
    # Skip frame processing to reduce lag (every other frame)
    if frame_count % 2 != 0:  # CHANGED: Process every 2nd frame instead of 3rd
        cv2.imshow("Sleep Detection", frame)
        continue

    # Start processing
    processed_frame = adaptive_preprocessing(frame)
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(gray, 
                                        scaleFactor=1.1,  # CHANGED: Faster detection
                                        minNeighbors=4,   # CHANGED: Reduced from 5
                                        minSize=(120, 120),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    
    # NEW: Face detection timing logic
    if len(faces) > 0:
        if face_detected_start is None:
            face_detected_start = current_time
        face_duration = current_time - face_detected_start
        
        # Select largest face only
        face = max(faces, key=lambda f: f[2]*f[3])
        x, y, w, h = face
        roi_gray = gray[y:y+h, x:x+w]
        
        # Only check eyes after 3 seconds of continuous face detection
        if face_duration >= 3:
            # Detect eyes with optimized parameters
            eyes = eye_cascade.detectMultiScale(roi_gray,
                                              scaleFactor=1.1,  # CHANGED: Faster
                                              minNeighbors=3,   # CHANGED: Reduced
                                              minSize=(25, 25))
            
            # Fast eye validation
            valid_eyes = []
            for (ex, ey, ew, eh) in eyes:
                if ey < h/2 and ew > 15 and eh > 15:
                    valid_eyes.append((ex, ey, ew, eh))
            
            # Check eye alignment quickly
            if len(valid_eyes) >= 2:
                eye1_y = valid_eyes[0][1]
                eye2_y = valid_eyes[1][1]
                if abs(eye1_y - eye2_y) > 25:
                    valid_eyes = valid_eyes[:1]

            # Sleep detection logic
            if len(valid_eyes) < 2 and not is_blurry(processed_frame):  
                if eye_closed_start is None:
                    eye_closed_start = current_time
                eye_closed_duration = current_time - eye_closed_start
                
                if eye_closed_duration >= 3:
                    if not sleep_detected and (current_time - last_alert_time) > 10:
                        if alert_sound is not None:
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
        if face_duration >= 3:  # Only show eyes after 3 seconds
            for (ex, ey, ew, eh) in valid_eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        
        # NEW: Display face detection timer
        cv2.putText(frame, f"Face Detected: {int(face_duration)}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
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
import cv2 
import pickle
import numpy as np
import smtplib
import os
from email.message import EmailMessage
import mimetypes
import time
import geopy
import geocoder
import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
import threading
import dlib
from sklearn.neighbors import KNeighborsClassifier

# Email Configuration (Original)
EMAIL_SENDER = "asamgoodwin@gmail.com"
EMAIL_PASSWORD = "khyo uetw qrka vtoe"
EMAIL_RECEIVER = "shebron333@gmail.com"
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# Enhanced Face Recognition Configuration
FACE_DETECTOR = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
SHAPE_PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
FACE_RECOGNIZER = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
FACE_MATCH_THRESHOLD = 0.58  # Optimized threshold

# Global variables (Original)
lock = threading.Lock()
training_in_progress = False
new_face_name = ""
face_approved = False
face_rejected = False
system_active = True
last_reject_alert = 0
approval_expiry = 0
rejection_start_time = 0
last_unknown_detection = 0
classified_name = ""

# Video configuration for performance
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
TARGET_FPS = 25

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray)
    if avg < 60:
        return adjust_gamma(image, 1.8)
    elif avg > 180:
        return adjust_gamma(image, 0.7)
    return image

def is_blurry(image, threshold=95):
    image = adaptive_brightness(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"📷 Blur Check: Variance={variance} (Threshold={threshold})")
    return variance < threshold

def capture_clear_photo(video, max_attempts=8, blur_threshold=95):
    best_image = None
    best_confidence = 0
    time.sleep(0.5)

    for attempt in range(1, max_attempts + 1):
        with lock:
            video.grab()
            ret, frame = video.retrieve()
        if not ret:
            print("❌ Error: Could not capture frame.")
            return None
        
        frame = adaptive_brightness(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Fast face detection with confidence
        faces = FACE_DETECTOR(rgb, 0)
        
        if len(faces) == 0:
            print(f"⚠️ No face detected in attempt {attempt}, skipping...")
            continue

        # Get best face with confidence
        face = max(faces, key=lambda f: f.confidence)
        if face.confidence < 0.85:
            continue

        # Check blur on face region only
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
        face_roi = frame[y:y+h, x:x+w]
        if is_blurry(face_roi, blur_threshold):
            print(f"⚠️ Blurry face detected (Attempt {attempt}/{max_attempts})")
            continue

        print(f"✅ Clear photo captured on attempt {attempt}!")
        return frame
        
    print("⚠️ Failed to capture clear photo after attempts")
    return None

def get_gps_location():
    try:
        g = geocoder.ip("me")
        if g.latlng:
            latitude, longitude = g.latlng
            location_url = f"https://www.google.com/maps?q={latitude},{longitude}"
            return f"📍 Vehicle GPS Location: {latitude}, {longitude}\n🔗 Google Maps: {location_url}"
        else:
            return "⚠️ GPS Location Unavailable"
    except Exception as e:
        print("❌ Error fetching GPS location:", e)
        return "⚠️ GPS Location Error"

def send_email_with_photo(image_path, is_rejection=False):
    try:
        msg = EmailMessage()
        msg["Subject"] = "ALERT! Unknown Face Detected in Vehicle - Action Needed" if not is_rejection else "FOLLOW-UP ALERT! Unauthorized Driver Still Present"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        gps_info = get_gps_location()
        content = f"An unknown person is driving the vehicle.\n\n{gps_info}"
        
        if not is_rejection:
            content += "\n\nReply with:\n- 'APPROVE' to allow access\n- 'REJECT' to deny access\n- 'ADD [Name]' to register as new user"
        
        msg.set_content(content)

        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            maintype, subtype = mimetypes.guess_type(image_path)[0].split("/")
            msg.add_attachment(img_data, maintype=maintype, subtype=subtype, filename="unknown_person.jpg")

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("📧 Alert sent: Unknown face detected with clear photo and GPS location" + (" (Follow-up)" if is_rejection else ""))
        os.remove(image_path)
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")

def check_for_approval():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT, timeout=10)
        mail.login(EMAIL_SENDER, EMAIL_PASSWORD)
        mail.select("inbox")

        time_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
        status, messages = mail.search(None, 'UNSEEN')
        if status != "OK":
            return None

        email_ids = messages[0].split()

        for email_id in reversed(email_ids):
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            if status != "OK":
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    email_date = parsedate_to_datetime(msg["Date"])
                    if email_date.tzinfo is None:
                        email_date = email_date.replace(tzinfo=timezone.utc)

                    if email_date < time_threshold:
                        continue

                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")

                    if subject.lower().startswith("re:"):
                        subject = subject[3:].strip()

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            if content_type == "text/plain":
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()

                    reply_content = ""
                    lines = body.splitlines()
                    for line in lines:
                        if line.strip().lower().startswith("on ") and "wrote:" in line.lower():
                            break
                        reply_content += line + "\n"

                    reply_upper = reply_content.upper().strip()
                    if "REJECT" in reply_upper:
                        return False
                    elif "APPROVE" in reply_upper:
                        return True
                    elif "ADD" in reply_upper:
                        parts = reply_content.split()
                        if len(parts) > 1:
                            return ("ADD", ' '.join(parts[1:]))
        return None
    except Exception as e:
        print(f"❌ Error checking for approval: {str(e)}")
        return None

def get_face_embedding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        faces = FACE_DETECTOR(rgb, 0)
        if len(faces) == 0:
            return None, None
        
        face = max(faces, key=lambda f: f.confidence)
        if face.confidence < 0.8:
            return None, None
        
        shape = SHAPE_PREDICTOR(rgb, face.rect)
        embedding = FACE_RECOGNIZER.compute_face_descriptor(rgb, shape)
        return np.array(embedding), face.rect
    except Exception as e:
        print(f"❌ Embedding error: {str(e)}")
        return None, None

def capture_training_photos(name):
    global training_in_progress, FACES, LABELS, system_active, video
    print(f"🎬 Starting training photo capture for {name}")
    
    embeddings = []
    count = 0
    last_face_time = time.time()

    try:
        training_in_progress = True
        
        while system_active and count < 50:
            current_time = time.time()
            if current_time - last_face_time > 120:
                print("🕒⛔ Training closed: No face detected for 2 minutes")
                break

            with lock:
                ret, frame = video.read()
            if not ret:
                continue

            frame = adaptive_brightness(frame)
            embedding, face = get_face_embedding(frame)
            
            if embedding is not None:
                last_face_time = current_time
                embeddings.append(embedding)
                count += 1

                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.putText(frame, f"Captured: {count}/50", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Training Capture", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        if len(embeddings) >= 20:
            FACES = np.vstack((FACES, embeddings)) if len(FACES) > 0 else np.array(embeddings)
            LABELS += [name] * len(embeddings)
            
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(LABELS, f)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(FACES, f)
            
            print(f"✅🎉 Successfully added {name} with {len(embeddings)} samples!")
        else:
            print("❌ Insufficient training samples captured")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
    finally:
        training_in_progress = False
        cv2.destroyWindow("Training Capture")

def email_check_thread():
    global face_approved, face_rejected, last_reject_alert, approval_expiry, rejection_start_time, last_unknown_detection, classified_name
    print("📧 Checking for email response...")
    start_time = time.time()
    
    while time.time() - start_time < 120 and not face_rejected:
        response = check_for_approval()
        if response is not None:
            if response == True:
                print("✅ Face approved. No alerts for 24 hours.")
                approval_expiry = time.time() + 86400
                face_approved = True
                face_rejected = False
                return
            elif response == False:
                print("❌ Face rejected. Starting 5-minute alert loop.")
                face_rejected = True
                last_reject_alert = time.time()
                rejection_start_time = time.time()
                last_unknown_detection = time.time()
                return
            elif isinstance(response, tuple) and response[0] == "ADD":
                new_face_name = response[1]
                print(f"🆕 Received ADD command for: {new_face_name}")
                training_thread = threading.Thread(target=capture_training_photos, args=(new_face_name,))
                training_thread.start()
                return
        
        if classified_name == "Recognized":
            print("✅ Recognized face detected - stopping email check")
            return
        
        time.sleep(10)
    
    if time.time() - start_time >= 120:
        print("⏳ No response received within 2 minutes")

def capture_and_send_email(check_email=True):
    global last_alert_time, face_detected_time, photo_capture_time
    print("📸 Capturing a clear photo and sending alert...")

    image_path = "unknown_person.jpg"
    clear_photo = capture_clear_photo(video)

    if clear_photo is not None:
        cv2.imwrite(image_path, clear_photo)
        send_email_with_photo(image_path, is_rejection=face_rejected)
        last_alert_time = time.time()
        face_detected_time = None
        photo_capture_time = None
        if check_email and not face_rejected:
            threading.Thread(target=email_check_thread).start()
    else:
        print("❌ Failed to capture a clear photo with face, skipping alert")

# Initialize face data
try:
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    FACES = np.array(FACES)
except:
    LABELS = []
    FACES = np.empty((0, 128), dtype=np.float32)

knn = KNeighborsClassifier(n_neighbors=3)
if len(LABELS) > 0:
    knn.fit(FACES, LABELS)

# Optimized video capture
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
video.set(cv2.CAP_PROP_FPS, TARGET_FPS)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Original timing parameters
waiting_period = 3
photo_wait_time = 5
alert_cooldown = 60
face_return_threshold = 2
REJECTION_TIMEOUT = 299

# Original state variables
face_detected_time = None
photo_capture_time = None
last_alert_time = None
known_face_visible_time = None
last_face_loss_time = None
predicted_name = ""
classified_name = ""

print(f"🔹 System Initialized")
print(f"🔹 Existing faces: {len(set(LABELS))} registered users")

# Main recognition loop with original timing
while system_active:
    if training_in_progress:
        print("⏳ Training in progress, system paused...")
        while training_in_progress:
            time.sleep(1)
        continue
        
    if face_approved and time.time() > approval_expiry:
        face_approved = False
        print("🕒 Approval period expired, resuming normal operations")

    if face_rejected:
        current_time = time.time()
        time_since_last_unknown = current_time - last_unknown_detection
        time_since_last_alert = current_time - last_reject_alert
        
        if time_since_last_unknown >= REJECTION_TIMEOUT:
            print("🔄⏸️ Rejection paused - no unauthorized face detected")
            face_rejected = False
        elif time_since_last_alert >= 300:
            print("🔄 Sending follow-up rejection alert...")
            capture_and_send_email(check_email=False)
            last_reject_alert = current_time
            last_unknown_detection = current_time
        elif classified_name == "Recognized":
            print("🔄✅ Rejection canceled - authorized face recognized")
            face_rejected = False

    with lock:
        ret, frame = video.read()
    if not ret:
        print("❌ Error: Could not capture frame.")
        time.sleep(0.1)
        continue

    frame = adaptive_brightness(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Fast face detection
    faces = FACE_DETECTOR(rgb, 0)
    
    face_still_detected = False
    
    if len(faces) > 0:
        face = max(faces, key=lambda f: f.confidence)
        if face.confidence < 0.8:
            continue
        
        rect = face.rect
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        face_still_detected = True
        
        # Original classification timing logic
        if face_detected_time is None and last_face_loss_time is not None:
            if (time.time() - last_face_loss_time) < face_return_threshold:
                photo_capture_time = None
                
        if face_detected_time is None:
            face_detected_time = time.time()
            classified_name = "Classifying..."

        elapsed_time = time.time() - face_detected_time

        # Maintain 3-second classification period
        if elapsed_time < waiting_period:
            classified_name = "Classifying..."
        else:
            embedding, _ = get_face_embedding(frame)
            
            if embedding is not None and len(LABELS) > 0:
                distances = np.linalg.norm(FACES - embedding, axis=1)
                best_match_distance = np.min(distances)
            else:
                best_match_distance = FACE_MATCH_THRESHOLD + 0.1

            print(f"🔍 Face match distance: {best_match_distance:.4f} | Threshold: {FACE_MATCH_THRESHOLD}")
            
            if best_match_distance < FACE_MATCH_THRESHOLD:
                classified_name = "Recognized"
                known_face_visible_time = time.time()
                photo_capture_time = None
                face_approved = False
                face_rejected = False
            else:
                classified_name = "Unknown"
                last_unknown_detection = time.time()
                if not face_rejected:
                    if last_alert_time is None or time.time() - last_alert_time >= alert_cooldown:
                        if photo_capture_time is None:
                            photo_capture_time = time.time()
                        elif time.time() - photo_capture_time >= photo_wait_time:
                            capture_and_send_email()
                            photo_capture_time = None

        # Update display name
        if face_still_detected:
            predicted_name = classified_name
        elif known_face_visible_time and time.time() - known_face_visible_time < 3:
            predicted_name = classified_name
        else:
            predicted_name = "Classifying..."

        # Original visualization
        label = "Recognized" if predicted_name == "Recognized" else ("Unknown" if predicted_name == "Unknown" else "Classifying...")
        color = (0, 255, 0) if label == "Recognized" else (0, 0, 255) if label == "Unknown" else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    else:
        if face_detected_time is not None:
            last_face_loss_time = time.time()
        face_detected_time = None
        classified_name = ""
        known_face_visible_time = None
        photo_capture_time = None

    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) == ord("q"):
        system_active = False
        break

video.release()
cv2.destroyAllWindows()
print("🔹 System shutdown complete")

# Original system documentation
class RejectionHandler:
    STATE_ACTIVE = 1
    STATE_PAUSED = 2
    STATE_CANCELED = 3
    
    def __init__(self):
        self.state = self.STATE_PAUSED
        self.timers = {
            'last_unknown': 0,
            'last_alert': 0
        }

def debug_timers(): pass
def system_health_check(): pass
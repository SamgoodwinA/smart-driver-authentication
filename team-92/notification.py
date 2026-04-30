from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from win32com.client import Dispatch

# Email configuration
EMAIL_SENDER = "asamgoodwin@gmail.com"  # Change to your email
EMAIL_PASSWORD = "khyo uetw qrka vtoe"  # Use an app password (for Gmail)
EMAIL_RECEIVER = "asamgoodwin@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Function to send an email alert
def send_alert_email():
    subject = "Security Alert: Unknown Person Detected!"
    body = "An unknown person is accessing the vehicle. Please check immediately."
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Security alert email sent!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function for voice notification
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load face detection and training data
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Reshape face data if needed
FACES = np.array(FACES)
if FACES.ndim == 1:
    FACES = FACES.reshape(-1, 1)

print('Shape of Faces matrix:', FACES.shape)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Attendance log configuration
COL_NAMES = ['NAME', 'TIME']
logged_names = set()
unknown_alert_sent = False  # To prevent multiple alerts for unknown persons

# Initialize video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture video frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = gray[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict the person
        output = knn.predict(resized_img)
        predicted_name = str(output[0])

        if predicted_name not in LABELS:  # If face is not recognized
            predicted_name = "Unknown"

            # Send email alert only once
            if not unknown_alert_sent:
                send_alert_email()
                speak("Unknown person detected. Security alert sent.")
                unknown_alert_sent = True  # Prevent duplicate alerts

        # Log attendance for known people only
        if predicted_name != "Unknown" and predicted_name not in logged_names:
            logged_names.add(predicted_name)
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            attendance_file = f"Attendance/Attendance_{date}.csv"
            file_exists = os.path.isfile(attendance_file)

            with open(attendance_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow([predicted_name, timestamp])

            speak(f"Attendance logged for {predicted_name}")

        # Draw rectangle and label on detected face
        color = (0, 0, 255) if predicted_name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(frame, predicted_name, (x, y - 15),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Face Recognition", frame)

    # Quit if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()

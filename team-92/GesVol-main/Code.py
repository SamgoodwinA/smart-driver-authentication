import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import math
from collections import deque
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# Initialize Mediapipe Hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.9, min_tracking_confidence=0.9)  # High accuracy

# Open webcam
cap = cv2.VideoCapture(0)

# Stabilization: Store previous landmarks to filter jitter
landmark_buffer = deque(maxlen=3)
call_active = False  # Track call state
volume_control_active = False  # Track volume control state

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    h, w, _ = frame.shape
    
    # **Apply Noise Reduction for Stability**
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.bilateralFilter(frame, 5, 75, 75)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            if len(lm_list) >= 21:
                # **Hand Landmark Key Points**
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]
                middle_tip = lm_list[12]
                ring_tip = lm_list[16]
                pinky_tip = lm_list[20]

                # **Stabilization: Store last 3 positions**
                landmark_buffer.append(lm_list)
                if len(landmark_buffer) == 3:
                    avg_positions = np.mean(landmark_buffer, axis=0).astype(int)
                    lm_list = [tuple(pos) for pos in avg_positions]

                # **Calculate Distances**
                thumb_index_distance = math.dist(thumb_tip, index_tip)  # Volume control
                thumb_middle_distance = math.dist(thumb_tip, middle_tip)  # Volume activation
                pinky_thumb_distance = math.dist(pinky_tip, thumb_tip)  # Pick Call
                index_ring_distance = math.dist(index_tip, ring_tip)  # Cancel Call

                # **Activate/Deactivate Volume Control (Thumb + Middle joined)**
                if thumb_middle_distance < 40 and not volume_control_active:
                    volume_control_active = True  # Enable volume control

                elif thumb_middle_distance < 40 and volume_control_active:
                    volume_control_active = False  # Disable volume control (lock volume)

                # **Adjust Volume if Active**
                if volume_control_active:
                    vol_level = np.interp(thumb_index_distance, [20, 200], [min_vol, max_vol])
                    volume.SetMasterVolumeLevel(vol_level, None)

                    # Display Volume
                    vol_percent = int(np.interp(thumb_index_distance, [20, 200], [0, 100]))
                    cv2.putText(frame, f'Volume: {vol_percent}%', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # **Call Handling Gestures**
                if pinky_thumb_distance < 40 and not call_active:
                    pyautogui.press("volumeup")  # Simulating call pickup
                    cv2.putText(frame, 'Picking Call', (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    call_active = True

                elif index_ring_distance < 40 and call_active:
                    pyautogui.press("volumedown")  # Simulating call hang-up
                    cv2.putText(frame, 'Hanging Call', (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    call_active = False

            # Draw hand landmarks with larger points
            for lm in lm_list:
                cv2.circle(frame, lm, 10, (0, 0, 255), -1)  # Increased size

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import time
import os
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

latest_result = None
prev_x, prev_y = None, None
movement_direction = "Stationary"

shape_x, shape_y = 320, 240
shape_color = (0, 0, 255)
shape_radius = 30

def receive_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=receive_result
)

if not os.path.exists(MODEL_PATH):
    print(f"Critical Error: '{MODEL_PATH}' not found!")
    exit()

detector = vision.HandLandmarker.create_from_options(options)
 
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit()
 
print("Hand Tracking Started! Press 'q' to quit.")
 
def detect_gesture(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]
    
    fingers_extended = []
    for i in range(1, 5):
        if hand_landmarks[tip_ids[i]].y < hand_landmarks[pip_ids[i]].y:
            fingers_extended.append(True)
        else:
            fingers_extended.append(False)
            
    extended_count = sum(fingers_extended)
    thumb_extended = abs(hand_landmarks[tip_ids[0]].x - hand_landmarks[pip_ids[0]].x) > 0.04

    if thumb_extended and extended_count == 0:
        if hand_landmarks[4].y < hand_landmarks[2].y:
            return "Thumbs Up"

    if extended_count >= 3:
        return "Open"
    elif extended_count == 0 and not thumb_extended:
        return "Closed Fist"
    else:
        return "Partial"
 
while True:
    success, frame = cap.read()
    
    if not success:
        time.sleep(1)
        continue
 
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp)
    
    current_result = latest_result
    gesture = "No hand detected"
    movement_direction = "Stationary"
    
    if current_result and current_result.hand_landmarks:
        hand_landmarks = current_result.hand_landmarks[0]
        
        if len(current_result.handedness) > 0:
            hand_label = current_result.handedness[0][0].category_name
        else:
            hand_label = "Unknown"
        
        gesture = detect_gesture(hand_landmarks)
        wrist_x, wrist_y = int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h)
        
        if prev_x is not None and prev_y is not None:
            dx = wrist_x - prev_x
            dy = wrist_y - prev_y
            threshold = 8
            
            if abs(dx) > abs(dy):
                if dx > threshold: movement_direction = "Right"
                elif dx < -threshold: movement_direction = "Left"
            else:
                if dy > threshold: movement_direction = "Down"
                elif dy < -threshold: movement_direction = "Up"
        
        prev_x, prev_y = wrist_x, wrist_y
        shape_x, shape_y = wrist_x, wrist_y
        
        if gesture == "Thumbs Up":
            shape_color = (0, 255, 0)
        elif gesture == "Open":
            shape_color = (255, 0, 0)
        else:
            shape_color = (0, 0, 255)
            
        x_dist = hand_landmarks[0].x - hand_landmarks[17].x
        y_dist = hand_landmarks[0].y - hand_landmarks[17].y
        z_dist = hand_landmarks[0].z - hand_landmarks[17].z
        hand_size_metric = math.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
        
        shape_radius = int(hand_size_metric * 400)
        shape_radius = max(10, min(shape_radius, 150))
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        for start_idx, end_idx in connections:
            start_lm = hand_landmarks[start_idx]
            end_lm = hand_landmarks[end_idx]
            pt1 = (int(start_lm.x * w), int(start_lm.y * h))
            pt2 = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(frame, pt1, pt2, (200, 200, 200), 1)

        for i, lm in enumerate(hand_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), cv2.FILLED)
        
        fingertip_ids = [4, 8, 12, 16, 20]
        for tip_id in fingertip_ids:
            lm = hand_landmarks[tip_id]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.putText(frame, str(tip_id), (x - 5, y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        wrist = hand_landmarks[0]
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(frame, f"{hand_label} Hand", (wrist_x - 40, wrist_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        prev_x, prev_y = None, None

    cv2.circle(frame, (shape_x, shape_y), shape_radius, shape_color, cv2.FILLED)
    cv2.circle(frame, (shape_x, shape_y), shape_radius, (0, 0, 0), 2)
    
    cv2.rectangle(frame, (5, 5), (310, 135), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (310, 135), (255, 255, 255), 1)
    
    status_color = (0, 255, 0) if gesture != "No hand detected" else (0, 0, 255)
    cv2.putText(frame, f"Status: {gesture}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, f"Movement: {movement_direction}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Shape Size: {shape_radius}px", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
    cv2.putText(frame, "Press 'q' to Quit", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
 
    cv2.imshow("Dynamic Hand Interaction System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
detector.close()
cap.release()
cv2.destroyAllWindows()
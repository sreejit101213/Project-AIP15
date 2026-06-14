import cv2
import mediapipe as mp
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Force Python to look in the script's actual directory for the task file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR,'hand_landmarker.task')

# Global variable to store live detection updates
latest_result = None

# Callback function when the model finishes an async frame detection
def receive_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global latest_result
    latest_result = result

# 1. Initialize the detector with the absolute file path
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=receive_result
)
detector = vision.HandLandmarker.create_from_options(options)
 
# Keeping your camera index at 1
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not access the webcam at index 1. Try changing it back to 0 if this fails.")
    exit()
 
print("Hand Tracking Started! Press 'q' to quit.")
 
def detect_gesture(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]
    extended = 0
    
    if abs(hand_landmarks[tip_ids[0]].x - hand_landmarks[pip_ids[0]].x) > 0.04:
        extended += 1
    
    for i in range(1, 5):
        if hand_landmarks[tip_ids[i]].y < hand_landmarks[pip_ids[i]].y:
            extended += 1
    
    if extended >= 4:
        return "Open"
    elif extended <= 1:
        return "Closed Fist"
    else:
        return "Partial"
 
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from camera.")
        break
 
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Package frame with a unique millisecond timestamp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp)
    
    gesture = "No hand detected"
    
    # 3. Read data from the live stream result callback
    if latest_result and latest_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
            if idx < len(latest_result.handedness):
                hand_label = latest_result.handedness[idx][0].category_name
            else:
                hand_label = "Unknown"
            
            gesture = detect_gesture(hand_landmarks)
            
            # Draw tracking points
            fingertip_ids = [4, 8, 12, 16, 20]
            for tip_id in fingertip_ids:
                lm = hand_landmarks[tip_id]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 10, (255, 0, 255), cv2.FILLED)
                cv2.putText(frame, str(tip_id), (x - 5, y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw wrist text label
            wrist = hand_landmarks[0]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f"{hand_label} Hand", (wrist_x - 40, wrist_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    status_color = (0, 255, 0) if gesture in ["Open", "Closed Fist"] else (0, 165, 255)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
 
    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
detector.close()
cap.release()
cv2.destroyAllWindows()
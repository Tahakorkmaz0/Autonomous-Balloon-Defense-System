import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

# --- CONFIGURATION ---
# 1. Device Configuration (CUDA support)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device used: {device}")

# 2. Load Model
# Note: Ensure 'justbalon.pt' is located in the 'models' directory.
model_path = os.path.join("models", "justbalon.pt")
# If running directly from src, handle path fallback
if not os.path.exists(model_path):
    model_path = "justbalon.pt" # Fallback for local testing

try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def precision_color_analysis(crop_img):
    """
    Performs precise color analysis in HSV space.
    Filters out background noise, white objects, and low-saturation pixels.
    Returns: status (FRIENDLY/HOSTILE), color (BGR tuple)
    """
    # Convert ROI to HSV color space
    hsv_roi = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    # Split channels
    h = hsv_roi[:, :, 0]
    s = hsv_roi[:, :, 1]
    v = hsv_roi[:, :, 2]

    # --- FILTER 1: WHITE & NOISE REJECTION ---
    # Increased Saturation threshold from 50 to 90.
    # This effectively filters out white/grey balloons and pale background objects.
    valid_mask = (s > 90) & (v > 40)

    # Get Hue values of valid pixels
    valid_h = h[valid_mask]

    # If not enough colorful pixels are found (e.g., object is black/white)
    if len(valid_h) < 10: 
        return None, None

    # --- SCORING ---
    # Red Hue Ranges (OpenCV HSV: 0-180) -> Wraps around 0 and 180
    red_score = np.sum(((valid_h < 20) | (valid_h > 160)))

    # Blue Hue Ranges
    blue_score = np.sum((valid_h > 90) & (valid_h < 130))

    # --- FILTER 2: REFLECTION HANDLING ---
    # If valid pixels are too few (likely just a reflection), ignore the object.
    # Threshold set to 50 pixels.
    if red_score < 50 and blue_score < 50:
        return None, None

    # --- DECISION LOGIC (IFF) ---
    if red_score > blue_score:
        return "HOSTILE", (0, 0, 255) # Red Bounding Box
    elif blue_score > red_score:
        return "FRIENDLY", (255, 0, 0) # Blue Bounding Box
    
    # If scores are equal or dominant color is undefined (Green, Yellow, etc.)
    return None, None

# 3. Initialize Camera
cap = cv2.VideoCapture(0)

print("System Started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Object Detection (YOLOv11)
    results = model.predict(frame, device=device, verbose=False, conf=0.5)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Crop Region of Interest (ROI)
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                # Perform Logic-Based Color Analysis
                status, color = precision_color_analysis(roi)

                # --- FILTER 3: VISUALIZATION CONTROL ---
                # If status is None (White, Yellow, Green, or Noise), DO NOT DRAW.
                if status is None:
                    continue

                label = f"{status} {conf:.2f}" 
                
                # Draw Bounding Box & Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Text Background for readability
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Autonomous Tracking System (Red/Blue Only)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
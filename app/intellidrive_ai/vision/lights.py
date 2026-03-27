import cv2
import numpy as np
from typing import Optional

class TrafficLightClassifier:
    """
    Classifies the state of a traffic light (Red, Yellow/Orange, Green)
    based on the vertical position of the most active light in the bbox.
    """
    def __init__(self):
        # Color ranges in HSV (simplified)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([90, 255, 255])

    def classify(self, frame: np.ndarray, bbox: list) -> str:
        """
        Classifies traffic light state from a cropped frame.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Clip bbox to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return "UNKNOWN"
            
        crop = frame[y1:y2, x1:x2]
        h_crop, w_crop = crop.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Masking for colors
        mask_red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask_green = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Divide into vertical slots: Top (Red), Middle (Yellow), Bottom (Green)
        red_roi = mask_red[0:h_crop//3, :]
        yellow_roi = mask_yellow[h_crop//3:2*h_crop//3, :]
        green_roi = mask_green[2*h_crop//3:, :]
        
        # Fallback: Check for bright spots (White-ish) if color mask is weak
        # This helps in night scenes where the light center is overexposed
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        red_score = cv2.countNonZero(red_roi)
        yellow_score = cv2.countNonZero(yellow_roi)
        green_score = cv2.countNonZero(green_roi)
        
        # Brightness fallback scores
        red_bright = cv2.countNonZero(bright_mask[0:h_crop//3, :])
        yellow_bright = cv2.countNonZero(bright_mask[h_crop//3:2*h_crop//3, :])
        green_bright = cv2.countNonZero(bright_mask[2*h_crop//3:, :])
        
        # Boost scores with brightness if they are in the right position
        red_score += red_bright * 0.5
        yellow_score += yellow_bright * 0.5
        green_score += green_bright * 0.5
        
        scores = {"RED": red_score, "YELLOW": yellow_score, "GREEN": green_score}
        best_color = max(scores, key=scores.get)
        
        if scores[best_color] > 5: # Threshold for activation
            return best_color
        
        return "UNKNOWN"

if __name__ == "__main__":
    # Test with mockup
    lib = TrafficLightClassifier()
    print("TrafficLightClassifier initialized.")

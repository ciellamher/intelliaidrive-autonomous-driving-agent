import cv2
import sys
import os
import hashlib
import math
from ultralytics import YOLO

class DualYOLODetector:
    def __init__(self, base_model_path='../yolov8n.pt', sign_model_path='../streetsignsense-yolo12n.pt'):
        # Base model for general objects (cars, people)
        self.base_model = YOLO(base_model_path)
        # Class 9 is traffic light in COCO
        self.base_classes = {
            2: 'car', 0: 'person', 9: 'traffic light',
            1: 'bicycle', 3: 'motorcycle', 5: 'bus', 7: 'truck'
        }
        
        # Custom model for traffic signs and lights
        self.sign_model = YOLO(sign_model_path)
        self.sign_classes = self.sign_model.names
        print("Dual-model active: Base (yolov8n) + Custom (streetsignsense-yolo12n)")
            
        self.history = {} # {track_id: [points]}

    def get_color(self, label):
        hash_obj = hashlib.md5(label.encode('utf-8'))
        return tuple(int(hash_obj.hexdigest()[i:i+2], 16) for i in (0, 2, 4))

    def detect_and_draw(self, frame):
        annotated_frame = frame.copy()
        class_centers = {}
        
        # 1. Track general objects with base model
        results_base = self.base_model.track(frame, persist=True, verbose=False)[0]
        if results_base.boxes.id is not None:
            boxes = results_base.boxes.xyxy.cpu().numpy()
            track_ids = results_base.boxes.id.int().cpu().numpy()
            class_ids = results_base.boxes.cls.int().cpu().numpy()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if int(class_id) in self.base_classes:
                    x1, y1, x2, y2 = map(int, box)
                    label = self.base_classes[int(class_id)]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    if track_id not in self.history:
                        self.history[track_id] = []
                    self.history[track_id].append(center)
                    if len(self.history[track_id]) > 30:
                        self.history[track_id].pop(0)
                        
                    color = self.get_color(label)
                    
                    # Draw track lines (history)
                    if len(self.history[track_id]) > 1:
                        for i in range(1, len(self.history[track_id])):
                            cv2.line(annotated_frame, self.history[track_id][i-1], self.history[track_id][i], color, 2)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} #{track_id}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                    if label not in class_centers:
                        class_centers[label] = []
                    class_centers[label].append(center)

        # 2. Detect traffic signs with custom model
        if self.sign_model:
            results_sign = self.sign_model.predict(frame, verbose=False)[0]
            if len(results_sign.boxes) > 0:
                s_boxes = results_sign.boxes.xyxy.cpu().numpy()
                s_class_ids = results_sign.boxes.cls.int().cpu().numpy()
                s_scores = results_sign.boxes.conf.cpu().numpy()
                
                for box, class_id, score in zip(s_boxes, s_class_ids, s_scores):
                    x1, y1, x2, y2 = map(int, box)
                    label = self.sign_classes[int(class_id)]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    color = self.get_color(label)
                    
                    # Draw simple box for signs
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} ({score:.2f})", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                    if label not in class_centers:
                        class_centers[label] = []
                    class_centers[label].append(center)

        # Connect objects of the same class
        for label, centers in class_centers.items():
            color = self.get_color(label)
            if len(centers) > 1:
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        pt1 = centers[i]
                        pt2 = centers[j]
                        dist = math.dist(pt1, pt2)
                        cv2.line(annotated_frame, pt1, pt2, color, 1, cv2.LINE_AA)
                        mid_x = (pt1[0] + pt2[0]) // 2
                        mid_y = (pt1[1] + pt2[1]) // 2
                        cv2.putText(annotated_frame, f"{int(dist)}px", (mid_x, mid_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return annotated_frame

def main(video_path="../data/video.mp4"):
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(video_path):
        video_path_resolved = os.path.join(base_dir, video_path)
    else:
        video_path_resolved = video_path

    if not os.path.exists(video_path_resolved):
        # fallback to sample 3
        video_path_resolved = os.path.join(base_dir, "../data/sample3.mp4")
        if not os.path.exists(video_path_resolved):
            print(f"Error: {video_path_resolved} not found.")
            return

    print(f"Starting standalone app detection on {video_path_resolved}...")
    
    # Initialize detector
    base_path = os.path.join(base_dir, '../yolov8n.pt')
    sign_path = os.path.join(base_dir, '../streetsignsense-yolo12n.pt')
    detector = DualYOLODetector(base_model_path=base_path, sign_model_path=sign_path)
    
    cap = cv2.VideoCapture(video_path_resolved)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    window_name = 'FlowCast Standalone App'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python3" to true' ''')
    except:
        pass

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        annotated_frame = detector.detect_and_draw(frame)
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Execution finished.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/video.mp4"
    main(path)

import argparse
import sys
import yaml
import cv2
import time
import torch
import numpy as np
import os

# FlowCast AI Modules (now inside app/)
from intellidrive_ai.vision.detect import DetectionEngine
from intellidrive_ai.vision.track import ByteTracker
from intellidrive_ai.vision.trajectories import TrajectoryEngine
from intellidrive_ai.vision.lights import TrafficLightClassifier
from intellidrive_ai.models.agent import RLAgent

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)

def vision_pipeline_stream(config, source):
    """
    Generator that processes video and yields (frame, analytics).
    """
    print("Initializing Vision Pipeline components...")
    
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    has_frame, first_frame = cap.read()
    if not has_frame:
        print("Failed to read video source.")
        return
    
    h, w = first_frame.shape[:2]
    
    device = config.get('system', {}).get('device', 'auto')
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
    engine = DetectionEngine(
        model_id=config['vision']['model_id'],
        sign_model_id="streetsignsense-yolo12n.pt",
        confidence=0.15,
        imgsz=480,
        classes=None,
        device=device
    )
    
    light_classifier = TrafficLightClassifier()
    agent = RLAgent(device=device)
    tracker = ByteTracker(track_buffer=10)
    
    SPEED_LIMIT_MAP = {
        3: 5, 4: 10, 5: 20, 6: 30, 7: 40, 8: 50, 9: 60,
        10: 70, 11: 80, 12: 90, 13: 100, 14: 110, 15: 120, 16: 130
    }
    current_speed_limit = "--"
    
    primary_skip = 2
    secondary_skip = 10
    frame_count = 0
    sign_dets = []
    tracked = []
    
    COLOR_PALETTE = {
        0: (255, 128, 0), 2: (0, 255, 255), 3: (100, 200, 255),
        5: (255, 0, 127), 7: (0, 0, 255), 9: (0, 255, 0), 11: (255, 255, 255)
    }
    
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        timestamp = time.time()
        is_infer = False
        detections_to_track = []
        
        if frame_count % primary_skip == 0:
            detections_to_track += engine.detect_frame(frame)
            is_infer = True
            
        if frame_count % secondary_skip == 0:
            sign_dets = engine.detect_frame(frame, use_signs=True)
            detections_to_track += sign_dets
            for d in sign_dets:
                if d['class_id'] in SPEED_LIMIT_MAP:
                    current_speed_limit = str(SPEED_LIMIT_MAP[d['class_id']])
            is_infer = True
            
        if is_infer:
            tracked = tracker.update(detections_to_track, timestamp)
        else:
            tracker.predict(timestamp)
            tracked = tracker.update([], timestamp)
            
        frame_count += 1
        
        # UI State
        traffic_light_status = "UNKNOWN"
        active_counts = {"Person": 0, "Vehicle": 0}
        object_centers = []
        min_hazard_dist = 1.0 # Default: safe
        self_x = int(w * 0.45)
        
        for t_obj in tracked:
            obj_id, det_class = t_obj['ID'], t_obj['class_id']
            bbox = t_obj['bbox']
            
            if det_class == 0: active_counts["Person"] += 1
            elif det_class in [2, 3, 5, 7]: active_counts["Vehicle"] += 1
            
            cx, cy = t_obj['x_center'], bbox[3] # BOTTOM-edge
            
            if cy > h * 0.92: continue # Hood mask
            
            dist = np.sqrt((cx/w - 0.45)**2 + (cy/h - 1.0)**2)
            
            if det_class in [0, 2, 3, 5, 7]:
                min_hazard_dist = min(min_hazard_dist, dist)
                # Targeting line (More visible: thickness 2)
                if dist < 0.6:
                    line_color = (0, 0, 255) if dist < 0.3 else (0, 165, 255)
                    cv2.line(frame, (self_x, h-120), (int(cx), int(cy)), line_color, 2)
            
            if det_class == 9: # Traffic Light
                traffic_light_status = light_classifier.classify(frame, bbox)
                cv2.line(frame, (self_x, h-120), (int(cx), int(cy)), (255, 255, 255), 2, cv2.LINE_AA)
            
            object_centers.append((int(cx), int(t_obj['y_center'])))
            
            # Drawing BBoxes
            color = COLOR_PALETTE.get(det_class, (200, 200, 200))
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            label_text = f"ID:{obj_id}"
            if det_class == 0: label_text = "Person"
            elif det_class in [2,3,5,7]: label_text = "Vehicle"
            elif det_class == 9: label_text = "Traffic Light"
            cv2.putText(frame, f"{label_text} #{obj_id}", (int(bbox[0]), int(bbox[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Radar Connectivity
        for i in range(len(object_centers)):
            for j in range(i + 1, len(object_centers)):
                p1, p2 = object_centers[i], object_centers[j]
                if (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 < 350**2:
                    cv2.line(frame, p1, p2, (150, 150, 150), 1)
        
        # 3. Decision Logic (RETAIN USER SETTINGS: 60/55)
        hazard_pct = int((1.0 - min_hazard_dist) * 100)
        
        agent_action = "GO"
        if hazard_pct >= 60:
            agent_action = "STOP"
        elif hazard_pct >= 55:
            agent_action = "SLOW"
            
        if traffic_light_status == "RED": agent_action = "STOP"
        elif traffic_light_status == "YELLOW": agent_action = "SLOW"
        elif traffic_light_status == "GREEN" and hazard_pct < 85: agent_action = "GO"
            
        # 4. PREMIUM UI HUD (LARGER & MORE VISIBLE)
        inf_fps = 1.0 / (time.time() - loop_start)
        
        status_color = (0, 255, 0)
        if agent_action == "STOP": status_color = (0, 0, 255)
        elif agent_action == "SLOW": status_color = (0, 165, 255)
        
        # Action Card (ENLARGED)
        cv2.rectangle(frame, (10, 10), (220, 110), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (220, 110), status_color, 3)
        cv2.putText(frame, agent_action, (35, 95), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(frame, "INTENT", (35, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Dashboard Glass Bar (ENLARGED)
        bar_x = 235
        cv2.rectangle(frame, (bar_x, 10), (w - 10, 110), (15, 15, 15), -1)
        cv2.rectangle(frame, (bar_x, 10), (w - 10, 110), (120, 120, 120), 2)
        
        # Speed Limit Sign (ENLARGED)
        cv2.circle(frame, (bar_x + 60, 60), 40, (255, 255, 255), -1)
        cv2.circle(frame, (bar_x + 60, 60), 40, (0, 0, 255), 6) # Thick red border
        cv2.putText(frame, current_speed_limit, (bar_x + 38, 75), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(frame, "LIMIT", (bar_x + 35, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Traffic Light Indicator (ENLARGED)
        lt_circ = (60, 60, 60)
        if traffic_light_status == "RED": lt_circ = (0, 0, 255)
        elif traffic_light_status == "YELLOW": lt_circ = (0, 255, 255)
        elif traffic_light_status == "GREEN": lt_circ = (0, 255, 0)
        cv2.circle(frame, (bar_x + 145, 60), 25, lt_circ, -1)
        cv2.putText(frame, "LIGHT", (bar_x + 125, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Metrics Center (ENLARGED)
        cv2.putText(frame, f"FPS: {inf_fps:.1f}", (bar_x + 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"HAZARD: {hazard_pct}%", (bar_x + 200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"VEHICLES: {active_counts['Vehicle']}  PEOPLE: {active_counts['Person']}", (bar_x + 200, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if frame_count % 10 == 0:
            print(f"Frames {frame_count} Processed | Real-time FPS: {inf_fps:.1f}")

        yield frame, {"fps": inf_fps}

    cap.release()

def run_vision_pipeline(config, source):
    for frame, analytics in vision_pipeline_stream(config, source):
        cv2.imshow('IntelliDrive UI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--source", type=str, default="data/sample6.mp4")
    args = parser.parse_args()
    
    # Try multiple paths for config
    config_paths = [args.config, "config.yaml", "../config.yaml", "app/config.yaml"]
    config = None
    for p in config_paths:
        if os.path.exists(p):
            config = load_config(p)
            print(f"Loaded config: {p}")
            break
            
    if not config:
        print("Fatal: Could not find config.yaml", file=sys.stderr)
        sys.exit(1)
            
    # Try multiple paths for source
    source_file = args.source
    if args.source and not args.source.isdigit():
        source_paths = [args.source, "data/sample6.mp4", "../data/sample6.mp4", "sample6.mp4", "../sample6.mp4"]
        for p in source_paths:
            if os.path.exists(p):
                source_file = p
                break
            
    if source_file:
        run_vision_pipeline(config, source_file)

if __name__ == "__main__":
    main()

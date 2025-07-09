from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import time
import numpy as np
from collections import defaultdict
import pickle
import os

# Select GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO model with speed optimizations
model = YOLO("best.pt")
model.to(device)

# Configure YOLO for better detection (relaxed parameters)
model.overrides['verbose'] = False
model.overrides['imgsz'] = 640
model.overrides['conf'] = 0.3  # Reduced from 0.6 for better recall
model.overrides['iou'] = 0.5  # Slightly increased for better NMS
model.overrides['max_det'] = 50  # Increased from 30

# Load video
video_path = "15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)

# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps

print(f"Video FPS: {fps}")
print(f"Target frame time: {frame_time:.4f} seconds")
print("Starting detection with strict ID management and re-identification...")

# Enhanced Deep SORT with better re-identification
tracker = DeepSort(
    max_age=150,  # Longer track persistence for players going off-screen
    n_init=3,  # Faster track confirmation
    max_cosine_distance=0.3,  # Stricter appearance matching for better re-ID
    nn_budget=100,  # Larger budget for feature storage
    max_iou_distance=0.7  # Better spatial matching
)

# Performance tracking
frame_count = 0
start_time = time.time()

# Relaxed filtering parameters
MIN_BBOX_AREA = 400
MIN_BBOX_WIDTH = 15
MIN_BBOX_HEIGHT = 30
MAX_BBOX_AREA_RATIO = 0.2
MIN_ASPECT_RATIO = 1.0
MAX_ASPECT_RATIO = 5.0
MIN_CONFIDENCE = 0.4

# STRICT ID MANAGEMENT
MAX_TOTAL_IDS = 26  # HARD LIMIT: 22 players + 4 referees
active_track_ids = set()  # Currently active track IDs
all_track_ids = set()  # All track IDs ever created
dormant_tracks = {}  # Store features of tracks that went off-screen


# Enhanced re-identification system
class ReIDManager:
    def __init__(self):
        self.id_features = {}  # Store average features for each ID
        self.id_positions = {}  # Store last known positions
        self.id_appearance_history = {}  # Store appearance history
        self.id_last_seen = {}  # Frame when ID was last seen
        self.pending_reids = {}  # Tracks pending re-identification
        self.feature_buffer_size = 10  # Number of features to average
        self.reid_threshold = 0.25  # Similarity threshold for re-ID
        self.position_weight = 0.3  # Weight for position in re-ID decision
        self.appearance_weight = 0.7  # Weight for appearance in re-ID decision

    def extract_features(self, frame, bbox):
        """Extract simple appearance features from bounding box"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Resize to standard size
        roi_resized = cv2.resize(roi, (64, 128))

        # Extract color histogram features
        roi_hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([roi_hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([roi_hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([roi_hsv], [2], None, [60], [0, 256])

        # Normalize histograms
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-10)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-10)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-10)

        # Extract texture features (LBP-like)
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        texture_features = []

        # Simple texture descriptor
        for i in range(0, gray.shape[0], 16):
            for j in range(0, gray.shape[1], 16):
                block = gray[i:i + 16, j:j + 16]
                if block.size > 0:
                    texture_features.append(np.std(block))

        texture_features = np.array(texture_features)
        if len(texture_features) == 0:
            texture_features = np.zeros(32)
        else:
            # Pad or truncate to fixed size
            if len(texture_features) < 32:
                texture_features = np.pad(texture_features, (0, 32 - len(texture_features)))
            else:
                texture_features = texture_features[:32]

        # Combine features
        features = np.concatenate([hist_h, hist_s, hist_v, texture_features])
        return features

    def update_id_features(self, track_id, frame, bbox):
        """Update feature database for a track ID"""
        features = self.extract_features(frame, bbox)
        if features is None:
            return

        if track_id not in self.id_features:
            self.id_features[track_id] = []
            self.id_appearance_history[track_id] = []

        self.id_features[track_id].append(features)
        self.id_appearance_history[track_id].append(frame_count)

        # Keep only recent features
        if len(self.id_features[track_id]) > self.feature_buffer_size:
            self.id_features[track_id] = self.id_features[track_id][-self.feature_buffer_size:]
            self.id_appearance_history[track_id] = self.id_appearance_history[track_id][-self.feature_buffer_size:]

        # Update position and last seen
        x1, y1, x2, y2 = bbox
        self.id_positions[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.id_last_seen[track_id] = frame_count

    def calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0

        # Normalize features
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(features1, features2) / (norm1 * norm2)

    def find_best_match(self, new_features, new_position, max_age_frames=150):
        """Find best matching dormant ID for re-identification"""
        best_match_id = None
        best_similarity = 0.0

        for track_id, feature_list in self.id_features.items():
            # Skip if ID is currently active
            if track_id in active_track_ids:
                continue

            # Skip if too old
            if frame_count - self.id_last_seen.get(track_id, 0) > max_age_frames:
                continue

            # Calculate average features
            avg_features = np.mean(feature_list, axis=0)

            # Calculate appearance similarity
            appearance_sim = self.calculate_similarity(new_features, avg_features)

            # Calculate position similarity (closer = more similar)
            last_pos = self.id_positions.get(track_id, (0, 0))
            pos_distance = np.sqrt((new_position[0] - last_pos[0]) ** 2 +
                                   (new_position[1] - last_pos[1]) ** 2)
            pos_similarity = 1.0 / (1.0 + pos_distance / 100)  # Normalize position distance

            # Combined similarity
            combined_sim = (appearance_sim * self.appearance_weight +
                            pos_similarity * self.position_weight)

            if combined_sim > best_similarity and combined_sim > self.reid_threshold:
                best_similarity = combined_sim
                best_match_id = track_id

        return best_match_id, best_similarity

    def attempt_reidentification(self, frame, new_tracks):
        """Attempt to re-identify tracks that may be returning players"""
        reidentified = {}

        for track in new_tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # Skip if this ID is already known
            if track_id in all_track_ids:
                continue

            # Skip if we've reached the ID limit
            if len(all_track_ids) >= MAX_TOTAL_IDS:
                continue

            # Extract features for this track
            l, t, r, b = map(int, track.to_ltrb())
            bbox = (l, t, r, b)
            features = self.extract_features(frame, bbox)
            position = ((l + r) // 2, (t + b) // 2)

            if features is not None:
                # Try to find a match
                match_id, similarity = self.find_best_match(features, position)

                if match_id is not None:
                    print(f"Re-identified track {track_id} as previous ID {match_id} (similarity: {similarity:.3f})")
                    reidentified[track_id] = match_id

        return reidentified


# Initialize re-identification manager
reid_manager = ReIDManager()

# Enhanced stability tracking
detection_history = {}
STABILITY_BUFFER_SIZE = 8
MIN_STABILITY_SCORE = 0.15

# Background subtraction for motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=20, detectShadows=False
)


def is_blank_space(x1, y1, x2, y2, frame_gray, threshold=10):
    """More lenient blank space check"""
    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return True

    std_dev = np.std(roi)
    if std_dev < threshold:
        return True

    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    max_bin = np.max(hist)
    if max_bin > 0.9 * roi.size:
        return True

    return False


def calculate_detection_stability(x1, y1, x2, y2, track_id):
    """More lenient stability calculation"""
    if track_id not in detection_history:
        detection_history[track_id] = []

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1
    area = width * height

    detection_history[track_id].append({
        'center': (center_x, center_y),
        'size': (width, height),
        'area': area,
        'frame': frame_count
    })

    if len(detection_history[track_id]) > STABILITY_BUFFER_SIZE:
        detection_history[track_id] = detection_history[track_id][-STABILITY_BUFFER_SIZE:]

    history = detection_history[track_id]
    if len(history) < 2:
        return 0.3

    centers = np.array([h['center'] for h in history])
    position_variance = np.var(centers, axis=0)
    position_stability = 1.0 / (1.0 + np.mean(position_variance) / 100)

    areas = np.array([h['area'] for h in history])
    area_variance = np.var(areas)
    size_stability = 1.0 / (1.0 + area_variance / 20000)

    frame_diffs = np.diff([h['frame'] for h in history])
    temporal_consistency = np.mean(frame_diffs <= 8)

    stability = (position_stability * 0.4 + size_stability * 0.3 + temporal_consistency * 0.3)
    return min(stability, 1.0)


def validate_detection(x1, y1, x2, y2, conf, frame_area, motion_mask=None, frame_gray=None):
    """Relaxed validation to reduce false negatives"""
    width = x2 - x1
    height = y2 - y1
    area = width * height

    if area < MIN_BBOX_AREA:
        return False, f"Too small: {area} < {MIN_BBOX_AREA}"
    if width < MIN_BBOX_WIDTH or height < MIN_BBOX_HEIGHT:
        return False, f"Dimensions too small: {width}x{height}"

    area_ratio = area / frame_area
    if area_ratio > MAX_BBOX_AREA_RATIO:
        return False, f"Too large: {area_ratio:.2f} > {MAX_BBOX_AREA_RATIO}"

    aspect_ratio = height / width if width > 0 else 0
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False, f"Bad aspect ratio: {aspect_ratio:.2f}"

    if conf < MIN_CONFIDENCE:
        return False, f"Low confidence: {conf:.2f} < {MIN_CONFIDENCE}"

    if frame_gray is not None:
        roi_gray = frame_gray[y1:y2, x1:x2]
        if roi_gray.size > 0:
            texture_std = np.std(roi_gray)
            if texture_std < 12:
                return False, f"Low texture: {texture_std:.2f} < 12"

            sobel_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            edge_density = np.mean(edge_magnitude)
            if edge_density < 8:
                return False, f"Low edge density: {edge_density:.2f} < 8"

            hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
            hist_normalized = hist / np.sum(hist)
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            if entropy < 2.0:
                return False, f"Low entropy: {entropy:.2f} < 2.0"

    if motion_mask is not None and frame_count > 30:
        roi_motion = motion_mask[y1:y2, x1:x2]
        motion_ratio = np.sum(roi_motion > 0) / area
        motion_threshold = max(0.02, 0.08 - (conf - 0.3) * 0.1)

        if motion_ratio < motion_threshold:
            return False, f"No motion: {motion_ratio:.3f} < {motion_threshold:.3f}"

    return True, "Valid"


def update_id_tracking(tracks, frame):
    """Enhanced ID tracking with strict limits and re-identification"""
    global active_track_ids, all_track_ids

    current_active = set()

    # First pass: Update existing tracks
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            current_active.add(track_id)

            # Update re-identification features
            l, t, r, b = map(int, track.to_ltrb())
            reid_manager.update_id_features(track_id, frame, (l, t, r, b))

            # Add to all_track_ids if not already there (respecting limit)
            if track_id not in all_track_ids and len(all_track_ids) < MAX_TOTAL_IDS:
                all_track_ids.add(track_id)
                print(f"New ID created: {track_id} (Total: {len(all_track_ids)}/{MAX_TOTAL_IDS})")

    # Second pass: Attempt re-identification for new tracks
    if len(all_track_ids) < MAX_TOTAL_IDS:
        reidentified = reid_manager.attempt_reidentification(frame, tracks)

        # Apply re-identification results
        for new_id, old_id in reidentified.items():
            if old_id not in current_active:  # Only if old ID is not currently active
                # Update tracking (this would need tracker modification in real implementation)
                # For now, just update our records
                current_active.add(old_id)
                all_track_ids.add(old_id)
                print(f"Re-identified: {new_id} -> {old_id}")

    # Clean up tracks that are no longer active
    inactive_tracks = all_track_ids - current_active
    for track_id in inactive_tracks:
        if track_id in reid_manager.id_last_seen:
            frames_since_seen = frame_count - reid_manager.id_last_seen[track_id]
            if frames_since_seen > 300:  # Remove very old tracks
                print(f"Removing old track: {track_id}")
                reid_manager.id_features.pop(track_id, None)
                reid_manager.id_positions.pop(track_id, None)
                reid_manager.id_last_seen.pop(track_id, None)

    active_track_ids = current_active


def log_detection_info(detections, frame_count):
    """Log detection information for debugging"""
    if frame_count % 60 == 0:
        print(f"Frame {frame_count}: {len(detections)} raw detections")
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            width, height = x2 - x1, y2 - y1
            area = width * height
            print(f"  Det {i}: conf={conf:.3f}, area={area:.0f}, aspect={height / width:.2f}")


def get_numeric_id(track_id):
    """Convert track_id to numeric value for comparison"""
    try:
        return int(track_id)
    except (ValueError, TypeError):
        # If track_id is not a simple integer string, extract numeric part
        import re
        match = re.search(r'\d+', str(track_id))
        if match:
            return int(match.group())
        return 999  # Default high value for unknown IDs


prev_frame_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_start_time = time.time()

    display_frame = frame.copy()
    frame_area = frame.shape[0] * frame.shape[1]

    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Motion detection
    motion_mask = None
    if frame_count > 30:
        motion_mask = background_subtractor.apply(frame)

        if prev_frame_gray is not None:
            frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)
            _, frame_diff_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            frame_diff_mask = cv2.morphologyEx(frame_diff_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.bitwise_or(motion_mask, frame_diff_mask)
    else:
        background_subtractor.apply(frame)

    prev_frame_gray = current_frame_gray.copy()

    # YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Debug logging
    if detections is not None and len(detections) > 0:
        log_detection_info(detections, frame_count)

    dets_for_tracker = []
    filtered_count = 0
    raw_detection_count = len(detections) if detections is not None else 0

    # Process detections
    if detections is not None and len(detections) > 0:
        h_orig, w_orig = frame.shape[:2]

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            # Quick blank space check
            if is_blank_space(x1, y1, x2, y2, current_frame_gray):
                filtered_count += 1
                continue

            is_valid, reason = validate_detection(
                x1, y1, x2, y2, conf, frame_area,
                motion_mask=motion_mask,
                frame_gray=current_frame_gray
            )

            if not is_valid:
                filtered_count += 1
                # Debug: Show filtered detections in red
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(display_frame, f"FILT:{reason[:10]}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                continue

            width = x2 - x1
            height = y2 - y1
            dets_for_tracker.append(([x1, y1, width, height], conf, 'player'))

            # Debug: Show valid detections in green
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(display_frame, f"VALID:{conf:.2f}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Update tracker
    tracks = tracker.update_tracks(dets_for_tracker, frame=frame)

    # Update ID tracking with re-identification
    update_id_tracking(tracks, frame)

    # Display tracks
    displayed_tracks = 0
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        numeric_id = get_numeric_id(track_id)
        l, t, r, b = map(int, track.to_ltrb())

        # Calculate stability
        stability = calculate_detection_stability(l, t, r, b, track_id)

        # Only display stable tracks
        if stability < MIN_STABILITY_SCORE:
            continue

        displayed_tracks += 1

        # Color coding based on numeric ID range
        if numeric_id <= 22:
            color = (0, 255, 0)  # Green for players (ID 1-22)
        elif numeric_id <= 26:
            color = (0, 255, 255)  # Yellow for referees (ID 23-26)
        else:
            color = (0, 0, 255)  # Red for excess IDs (should not happen)

        # Thicker boxes for better visibility
        cv2.rectangle(display_frame, (l, t), (r, b), color, 3)

        # Enhanced label with type indication
        player_type = "P" if numeric_id <= 22 else "R" if numeric_id <= 26 else "X"
        text = f"{player_type}{track_id} S:{stability:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Background for text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.rectangle(display_frame, (l, t - 30), (l + text_size[0] + 10, t - 5), color, -1)
        cv2.putText(display_frame, text, (l + 5, t - 12), font, font_scale, (0, 0, 0), thickness)

        # Center point
        center_x = (l + r) // 2
        center_y = (t + b) // 2
        cv2.circle(display_frame, (center_x, center_y), 4, color, -1)

    # Enhanced statistics display
    cv2.putText(display_frame, f"Raw Detections: {raw_detection_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Valid for Tracking: {len(dets_for_tracker)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Filtered Out: {filtered_count}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Active Tracks: {len(active_track_ids)}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Displayed: {displayed_tracks}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display_frame, f"Total IDs: {len(all_track_ids)}/{MAX_TOTAL_IDS}",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(display_frame, f"Frame: {frame_count}",
                (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ID limit warning
    if len(all_track_ids) >= MAX_TOTAL_IDS:
        cv2.putText(display_frame, "ID LIMIT REACHED!",
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Timing control
    processing_time = time.time() - frame_start_time
    target_time = start_time + (frame_count * frame_time)
    current_time = time.time()
    if current_time < target_time:
        time.sleep(target_time - current_time)

    cv2.imshow("Enhanced Player Detection with Re-ID and Strict ID Limits", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Periodic logging
    if frame_count % int(fps) == 0:
        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.1f}s | Frame: {frame_count} | "
              f"Raw: {raw_detection_count} | Valid: {len(dets_for_tracker)} | "
              f"Active: {len(active_track_ids)} | Total IDs: {len(all_track_ids)}/{MAX_TOTAL_IDS} | "
              f"Displayed: {displayed_tracks}")

# Final stats
end_time = time.time()
actual_duration = end_time - start_time
expected_duration = frame_count / fps

cap.release()
cv2.destroyAllWindows()

print(f"\n=== FINAL STATISTICS ===")
print(f"Total frames processed: {frame_count}")
print(f"Total unique IDs created: {len(all_track_ids)}")
print(f"Expected duration: {expected_duration:.2f}s")
print(f"Actual duration: {actual_duration:.2f}s")
print(f"Timing accuracy: {(expected_duration / actual_duration) * 100:.1f}%")
print(f"Average IDs per frame: {len(all_track_ids) / frame_count * fps:.1f}")
print(f"ID limit respected: {'YES' if len(all_track_ids) <= MAX_TOTAL_IDS else 'NO'}")
print(f"Final active tracks: {len(active_track_ids)}")
print(f"Re-identification features stored: {len(reid_manager.id_features)}")
print("Enhanced with: Strict ID limits, Re-identification system, Persistent tracking")
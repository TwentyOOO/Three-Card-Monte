#!/usr/bin/env python3
"""
Three-Card Monte Tracker - FIXED & IMPROVED VERSION
Addresses all common issues and improves tracking accuracy
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import json
import os
import sys
import traceback

# Try to import scipy for Hungarian Algorithm, fallback to greedy matching
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  Warning: scipy not installed. Using greedy matching instead of Hungarian Algorithm.")
    print("   Install scipy: pip install scipy")


class KalmanFilter:
    """Simple 2D Kalman Filter for smooth position tracking"""

    def __init__(self, dt=0.1, process_variance=0.01, measurement_variance=4):
        """
        Initialize Kalman Filter for 2D position tracking

        Args:
            dt: Time step
            process_variance: Process noise covariance (system uncertainty)
            measurement_variance: Measurement noise covariance (sensor uncertainty)
        """
        self.dt = dt
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # State: [x, y, vx, vy] (position and velocity)
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process covariance
        self.Q = np.eye(4) * process_variance

        # Measurement covariance
        self.R = np.eye(2) * measurement_variance

        # Estimate error covariance
        self.P = np.eye(4)

        self.initialized = False

    def predict(self):
        """Predict next state without measurement"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
        """Update state with new measurement"""
        if not self.initialized:
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.initialized = True
            return

        # Innovation
        z = np.array(measurement, dtype=np.float32)
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_position(self):
        """Get current estimated position"""
        return tuple(self.state[:2].astype(int))


class ImprovedCardTracker:
    def __init__(self):
        self.tracking_history = defaultdict(lambda: deque(maxlen=200))
        self.kalman_filters = {}  # Store Kalman filters for each card
        self.winner_id = None
        self.winner_initial_pos = None
        self.winner_final_pos = None
        self.card_colors = {}  # Store average color of each card

        # Statistics
        self.stats = {
            'total_frames': 0,
            'cards_detected': 0,
            'winner_detected_frame': None,
            'tracking_losses': 0,
            'detection_confidence': []
        }

    def preprocess_frame(self, frame):
        """Improve frame quality for detection"""
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        # Enhance contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_cards_improved(self, frame):
        """Improved card detection with multiple methods"""
        height, width = frame.shape[:2]

        # Method 1: Color-based detection (for white/light cards)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect light colored regions (cards)
        lower_light = np.array([0, 0, 150])
        upper_light = np.array([180, 100, 255])
        mask_light = cv2.inRange(hsv, lower_light, upper_light)

        # Method 2: Edge-based detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Combine masks
        combined_mask = cv2.bitwise_or(mask_light, edges)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cards = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area (cards should be reasonably large but not the whole frame)
            if 3000 < area < (width * height * 0.3):
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio (cards are rectangular)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Cards can be vertical or horizontal
                if (0.5 < aspect_ratio < 2.0) and w > 50 and h > 50:
                    # Calculate perimeter ratio (rectangularity)
                    perimeter = cv2.arcLength(contour, True)
                    rectangularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

                    # Get center
                    cx = x + w // 2
                    cy = y + h // 2

                    # Calculate confidence based on multiple factors
                    # Aspect ratio score: higher when closer to ideal 1.4
                    aspect_score = 1.0 - min(1.0, abs(aspect_ratio - 1.4) / 0.5)

                    confidence = min(1.0, (
                        0.4 * min(1.0, area / 20000) +  # Size factor (0.4 weight)
                        0.3 * aspect_score +             # Aspect ratio factor (0.3 weight) - rewards cards closer to 1.4
                        0.3 * rectangularity             # Shape factor (0.3 weight)
                    ))

                    cards.append({
                        'bbox': (x, y, w, h),
                        'center': (cx, cy),
                        'area': area,
                        'contour': contour,
                        'confidence': confidence,
                        'aspect_ratio': aspect_ratio
                    })

        # Sort by x-position (left to right)
        cards.sort(key=lambda c: c['center'][0])

        # Filter to top 3 most confident detections if we have too many
        if len(cards) > 3:
            cards.sort(key=lambda c: c['confidence'], reverse=True)
            cards = cards[:3]
            cards.sort(key=lambda c: c['center'][0])

        return cards

    def detect_winner_card(self, frame, bbox):
        """Improved winner card detection using multiple color spaces and strategies"""
        x, y, w, h = bbox

        # Ensure bbox is within frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return False, 0.0

        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            return False, 0.0

        # Method 1: HSV red detection (primary method)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red ranges in HSV (expanded for better coverage)
        lower_red1 = np.array([0, 40, 40])      # Reduced saturation threshold
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 40])    # Reduced saturation threshold
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Method 2: Check for pink/magenta (hearts/diamonds)
        lower_pink = np.array([135, 30, 30])    # Expanded pink range
        upper_pink = np.array([175, 255, 255])
        pink_mask = cv2.inRange(hsv_roi, lower_pink, upper_pink)

        # Combine masks
        combined_mask = cv2.bitwise_or(red_mask, pink_mask)

        # Apply morphological closing to connect nearby red regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Calculate red ratio
        red_pixels = cv2.countNonZero(combined_mask)
        total_pixels = w * h
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0

        # Method 3: BGR color space analysis
        mean_color = cv2.mean(roi)[:3]  # B, G, R
        # Check if red channel dominates
        red_value = mean_color[2]
        green_value = mean_color[1]
        blue_value = mean_color[0]

        # Red should be significantly higher than blue and green
        red_dominance = (red_value > (blue_value + green_value) * 0.5) and (red_value > 60)

        # Multi-factor decision logic (more lenient)
        is_winner = (
            red_ratio > 0.04 or  # Reduced from 0.05
            (red_ratio > 0.015 and red_dominance) or  # Reduced from 0.02
            (red_ratio > 0.02 and red_value > 100)  # Additional check for bright red
        )

        return is_winner, red_ratio

    def match_cards_advanced(self, prev_cards, curr_cards):
        """
        Advanced card matching using Hungarian Algorithm (optimal assignment)
        Metrics: position, size, aspect ratio, and confidence
        """
        if not prev_cards or not curr_cards:
            return {}

        # Build cost matrix
        n_prev = len(prev_cards)
        n_curr = len(curr_cards)
        cost_matrix = np.zeros((n_prev, n_curr))

        for i, prev_card in enumerate(prev_cards):
            prev_center = prev_card['center']
            prev_area = prev_card['area']
            prev_aspect = prev_card['aspect_ratio']
            prev_confidence = prev_card['confidence']

            for j, curr_card in enumerate(curr_cards):
                curr_center = curr_card['center']
                curr_area = curr_card['area']
                curr_aspect = curr_card['aspect_ratio']
                curr_confidence = curr_card['confidence']

                # Distance cost (lower weight - position changes with motion)
                distance = np.sqrt(
                    (prev_center[0] - curr_center[0])**2 +
                    (prev_center[1] - curr_center[1])**2
                )

                # Area similarity cost (important - size should be stable)
                area_diff = abs(prev_area - curr_area) / max(prev_area, curr_area)

                # Aspect ratio similarity (important - shape should be stable)
                aspect_diff = abs(prev_aspect - curr_aspect) / max(prev_aspect, curr_aspect, 0.5)

                # Confidence similarity bonus (cards with consistent detection)
                confidence_sim = 1.0 - abs(prev_confidence - curr_confidence)

                # Weighted combined cost (lower is better)
                cost = (
                    0.3 * distance +           # Position (30% - flexible)
                    0.35 * area_diff * 100 +   # Area (35% - important)
                    0.25 * aspect_diff * 100 + # Shape (25% - important)
                    -0.1 * confidence_sim * 10 # Confidence bonus (negative = reward)
                )

                cost_matrix[i, j] = cost

        # Use Hungarian Algorithm if available, else fallback to greedy matching
        if HAS_SCIPY:
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # Build matches with cost threshold
                matches = {}
                for i, j in zip(row_indices, col_indices):
                    # Only accept matches with reasonable cost
                    if cost_matrix[i, j] < 150:
                        matches[i] = j

                return matches

            except Exception:
                # If Hungarian fails, fallback to greedy
                pass

        # Greedy matching (either scipy not available or Hungarian failed)
        matches = {}
        used_curr = set()

        for i in range(n_prev):
            valid_matches = [(cost_matrix[i, j], j) for j in range(n_curr)
                           if j not in used_curr and cost_matrix[i, j] < 150]

            if valid_matches:
                best_cost, best_j = min(valid_matches)
                matches[i] = best_j
                used_curr.add(best_j)

        return matches

    def smooth_trajectory(self, card_id):
        """Apply smoothing to trajectory to reduce jitter"""
        if card_id not in self.tracking_history:
            return

        history = list(self.tracking_history[card_id])
        if len(history) < 3:
            return

        # Simple moving average smoothing
        window = 3
        smoothed = []
        for i in range(len(history)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(history), i + window // 2 + 1)
            window_points = history[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            smoothed.append((int(avg_x), int(avg_y)))

        self.tracking_history[card_id] = deque(smoothed, maxlen=200)

    def update_kalman(self, card_id, position):
        """Update Kalman filter with new card position"""
        if card_id not in self.kalman_filters:
            self.kalman_filters[card_id] = KalmanFilter()

        self.kalman_filters[card_id].update(position)

    def predict_position(self, card_id):
        """Predict next position using Kalman filter"""
        if card_id in self.kalman_filters:
            return self.kalman_filters[card_id].predict()
        return None

    def get_kalman_position(self, card_id):
        """Get smoothed position from Kalman filter"""
        if card_id in self.kalman_filters:
            return self.kalman_filters[card_id].get_position()
        return None

def process_video_fixed(video_path, output_dir="output"):
    """Fixed video processing with proper error handling"""

    print(f"\n{'='*60}")
    print(f"THREE-CARD MONTE TRACKER - FIXED VERSION")
    print(f"{'='*60}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("⚠️  Warning: FPS reported as 0, defaulting to 30")
        fps = 30
    fps = int(fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video Information:")
    print(f"   Path: {video_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")

    if fps > 0:
        duration = total_frames / fps
        print(f"   Duration: {duration:.2f} seconds")
    else:
        print(f"   Duration: Unknown (FPS = 0)")

    print(f"\n{'='*60}\n")

    # Setup video writer with proper codec
    output_video_path = os.path.join(output_dir, "tracked_output_fixed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    try:
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print("⚠️  Warning: Could not open video writer, will save frames only")
            writer = None
    except Exception as e:
        print(f"⚠️  Warning: Video writer error: {e}")
        print("   Will save frames only")
        writer = None

    # Initialize tracker
    tracker = ImprovedCardTracker()
    tracker.stats['total_frames'] = total_frames

    prev_cards = []
    card_ids = {}
    cards = []  # Initialize to prevent UnboundLocalError if video has no frames
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
    frame_count = 0
    success_count = 0

    print("🔍 Processing video...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Preprocess frame for better detection (enhancement + denoising)
            enhanced_frame = tracker.preprocess_frame(frame)

            # Detect cards on enhanced frame
            cards = tracker.detect_cards_improved(enhanced_frame)
            tracker.stats['cards_detected'] += len(cards)

            if cards:
                tracker.stats['detection_confidence'].extend([c['confidence'] for c in cards])

            # Match with previous frame
            if prev_cards and cards:
                matches = tracker.match_cards_advanced(prev_cards, cards)

                # Update card IDs
                new_card_ids = {}
                for prev_idx, curr_idx in matches.items():
                    if prev_idx in card_ids:
                        new_card_ids[curr_idx] = card_ids[prev_idx]

                # Assign new IDs to unmatched cards
                next_id = max(new_card_ids.values(), default=-1) + 1
                for j in range(len(cards)):
                    if j not in new_card_ids:
                        new_card_ids[j] = next_id
                        next_id += 1
                        tracker.stats['tracking_losses'] += 1

                card_ids = new_card_ids
            else:
                card_ids = {i: i for i in range(len(cards))}

            # Create annotated frame
            annotated_frame = frame.copy()

            # Store winner bbox for visual effects
            winner_bbox = None

            # Process each card
            for i, card in enumerate(cards):
                x, y, w, h = card['bbox']
                cx, cy = card['center']
                card_id = card_ids.get(i, -1)

                # Detect winner (only in early frames)
                if tracker.winner_id is None and frame_count < 90:
                    is_winner, red_ratio = tracker.detect_winner_card(frame, card['bbox'])

                    # Trust the advanced detection logic in detect_winner_card
                    if is_winner:
                        tracker.winner_id = card_id
                        tracker.winner_initial_pos = (cx, cy)
                        tracker.stats['winner_detected_frame'] = frame_count
                        print(f"✅ Winner detected at frame {frame_count}")
                        print(f"   Card ID: {card_id}")
                        print(f"   Position: ({cx}, {cy})")
                        print(f"   Red ratio: {red_ratio:.2%}\n")

                # Update tracking history
                if card_id >= 0:
                    # Update Kalman filter with new position (smooth tracking)
                    tracker.update_kalman(card_id, (cx, cy))
                    # Get smoothed position from Kalman for drawing
                    kalman_pos = tracker.get_kalman_position(card_id)
                    if kalman_pos is not None:
                        cx, cy = kalman_pos  # Use Kalman-smoothed position
                    # Store history with smoothed position
                    tracker.tracking_history[card_id].append((cx, cy))
                    # Apply trajectory smoothing for winner card (reduce jitter)
                    if card_id == tracker.winner_id and len(tracker.tracking_history[card_id]) > 3:
                        tracker.smooth_trajectory(card_id)

                # Choose visualization
                color = colors[card_id % 3] if card_id >= 0 else (128, 128, 128)
                thickness = 5 if card_id == tracker.winner_id else 2

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, thickness)

                # Draw label
                label = f"Card {card_id}"
                if card_id == tracker.winner_id:
                    label += " ★ WINNER"
                    # Store winner bbox for later visual effects
                    winner_bbox = (x, y, w, h)

                    # 🌈 Boost colors for winner card (higher saturation & brightness)
                    try:
                        roi = annotated_frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            h_c, s_c, v_c = cv2.split(hsv_roi)
                            # Increase saturation by 40%
                            s_c = np.clip(s_c.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
                            # Increase brightness by 10%
                            v_c = np.clip(v_c.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
                            hsv_boosted = cv2.merge([h_c, s_c, v_c])
                            roi_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
                            annotated_frame[y:y+h, x:x+w] = roi_boosted

                    except Exception as e:
                        pass  # Silently skip color boost if error

                # Add confidence score
                label += f" ({card['confidence']:.2f})"

                cv2.putText(annotated_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw center
                cv2.circle(annotated_frame, (cx, cy), 6, color, -1)

                # Draw trajectory for winner with motion filter
                if card_id == tracker.winner_id and len(tracker.tracking_history[card_id]) > 1:
                    points = list(tracker.tracking_history[card_id])
                    for j in range(1, len(points)):
                        x1, y1 = points[j-1]
                        x2, y2 = points[j]
                        # Calculate motion distance
                        motion_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        # Only draw line if motion is significant (> 1.5 pixels)
                        if motion_dist >= 1.5:
                            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            # 🔲 Blur background with winner card highlighted (if detected)
            if tracker.winner_id is not None and winner_bbox is not None:
                try:
                    x, y, w, h = winner_bbox
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, annotated_frame.shape[1] - x)
                    h = min(h, annotated_frame.shape[0] - y)

                    if w > 0 and h > 0:
                        # Create blurred version of frame
                        blurred = cv2.GaussianBlur(annotated_frame, (51, 51), 0)

                        # Create mask for winner card region (with padding)
                        mask = np.zeros(annotated_frame.shape[:2], dtype=np.uint8)
                        cv2.rectangle(mask, (max(0, x-15), max(0, y-15)),
                                     (min(annotated_frame.shape[1], x+w+15),
                                      min(annotated_frame.shape[0], y+h+15)), 255, -1)

                        # Inverse mask for background
                        mask_inv = cv2.bitwise_not(mask)

                        # Blend: blurred background + original winner region
                        bg_blurred = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
                        fg_original = cv2.bitwise_and(annotated_frame, annotated_frame, mask=mask)

                        annotated_frame = cv2.add(bg_blurred, fg_original)

                        # ✨ Add subtle glow around winner (transparent overlay)
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, (max(0, x-10), max(0, y-10)),
                                     (min(annotated_frame.shape[1], x+w+10),
                                      min(annotated_frame.shape[0], y+h+10)), (0, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)

                except Exception as e:
                    pass  # Silently skip blur effect if error

            # Add info overlay
            info_y = 30
            cv2.rectangle(annotated_frame, (0, 0), (400, 120), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            cv2.putText(annotated_frame, f"Cards: {len(cards)}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30

            if tracker.winner_id is not None:
                cv2.putText(annotated_frame, f"Tracking: Card {tracker.winner_id}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Write frame
            if writer is not None:
                try:
                    writer.write(annotated_frame)
                    success_count += 1
                except Exception as e:
                    print(f"⚠️  Frame {frame_count} write error: {e}")

            # Save keyframes
            if frame_count % 30 == 0 or (tracker.winner_id is not None and frame_count < 120):
                keyframe_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(keyframe_path, annotated_frame)

            # Progress update
            if frame_count % 60 == 0:
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                else:
                    print(f"⏳ Progress: {frame_count} frames processed (total_frames = 0)")

            prev_cards = cards

        print(f"\n✅ Processing complete!")
        print(f"   Frames processed: {frame_count}")
        print(f"   Frames written: {success_count}")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        traceback.print_exc()
        return False

    finally:
        # CRITICAL: Properly close everything
        cap.release()
        if writer is not None:
            writer.release()

        print(f"\n📹 Video capture released")
        if writer is not None:
            print(f"✅ Video writer released")

    # Find final position
    if tracker.winner_id is not None and cards:
        for i, card in enumerate(cards):
            if card_ids.get(i) == tracker.winner_id:
                tracker.winner_final_pos = card['center']
                break

    # Analysis
    print(f"\n{'='*60}")
    print("TRACKING ANALYSIS")
    print(f"{'='*60}\n")

    if tracker.winner_id is not None:
        print(f"✅ Winner Card Tracked Successfully")
        print(f"   Card ID: {tracker.winner_id}")
        print(f"   First detected: Frame {tracker.stats['winner_detected_frame']}")

        # Calculate movement
        history = list(tracker.tracking_history[tracker.winner_id])
        if len(history) > 1:
            total_distance = sum(
                np.sqrt((history[i][0] - history[i-1][0])**2 +
                       (history[i][1] - history[i-1][1])**2)
                for i in range(1, len(history))
            )

            print(f"\n   Movement Analysis:")
            print(f"   - Total distance: {total_distance:.1f} pixels")
            print(f"   - Tracked for: {len(history)} frames")

            if tracker.winner_initial_pos and tracker.winner_final_pos:
                print(f"   - Start: {tracker.winner_initial_pos}")
                print(f"   - End: {tracker.winner_final_pos}")

                # Determine final position
                final_x = tracker.winner_final_pos[0]
                if final_x < width / 3:
                    position = "LEFT"
                elif final_x > 2 * width / 3:
                    position = "RIGHT"
                else:
                    position = "CENTER"

                print(f"\n   ⭐ WINNER IS IN {position} POSITION")
    else:
        print("❌ Winner card was not detected")

    avg_confidence = np.mean(tracker.stats['detection_confidence']) if tracker.stats['detection_confidence'] else 0

    # Calculate success rate safely (avoid division by zero)
    if frame_count > 0:
        success_rate = (1 - tracker.stats['tracking_losses'] / frame_count) * 100
    else:
        success_rate = 0.0

    print(f"\n📊 Statistics:")
    print(f"   - Frames processed: {frame_count}")
    print(f"   - Cards detected: {tracker.stats['cards_detected']}")
    print(f"   - Avg detection confidence: {avg_confidence:.2f}")
    print(f"   - Tracking losses: {tracker.stats['tracking_losses']}")
    print(f"   - Success rate: {success_rate:.1f}%")

    # Save report
    duration = total_frames / fps if fps > 0 else 0.0

    report = {
        'video_info': {
            'path': video_path,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        },
        'tracking_results': {
            'winner_id': tracker.winner_id,
            'winner_detected_frame': tracker.stats['winner_detected_frame'],
            'winner_initial_pos': list(tracker.winner_initial_pos) if tracker.winner_initial_pos else None,
            'winner_final_pos': list(tracker.winner_final_pos) if tracker.winner_final_pos else None,
        },
        'statistics': {
            'frames_processed': frame_count,
            'frames_written': success_count,
            'cards_detected': tracker.stats['cards_detected'],
            'tracking_losses': tracker.stats['tracking_losses'],
            'avg_confidence': float(avg_confidence)
        }
    }

    report_path = os.path.join(output_dir, "tracking_report_fixed.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print(f"✅ Video: {output_video_path}")
    print(f"✅ Report: {report_path}")
    print(f"✅ Keyframes: {output_dir}/frame_*.jpg")
    print(f"{'='*60}\n")

    # Verify video file
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path)
        print(f"✅ Output video created: {file_size / 1024 / 1024:.1f} MB")

        # Try to verify video is readable
        test_cap = cv2.VideoCapture(output_video_path)
        try:
            if test_cap.isOpened():
                # Try reading first frame to verify integrity
                ret, _ = test_cap.read()
                if ret:
                    print("✅ Verified: Output video is readable")
                else:
                    print("⚠️  Warning: Output video opened but no frames could be read")
            else:
                print("⚠️  Warning: Could not open output video for verification")
        finally:
            test_cap.release()  # Always release, even if error occurs
    else:
        print("❌ Output video was not created")

    return True


def main():
    """Main entry point for the tracker"""
    if len(sys.argv) < 2:
        print("Usage: python track_cards_fixed.py <video_path> [output_dir]")
        print("\nExample:")
        print("  python track_cards_fixed.py input_video.mp4")
        print("  python track_cards_fixed.py input_video.mp4 results")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)

    success = process_video_fixed(video_path, output_dir)

    if success:
        print("\n🎉 Processing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

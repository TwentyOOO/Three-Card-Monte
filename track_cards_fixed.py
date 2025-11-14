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

class ImprovedCardTracker:
    def __init__(self):
        self.tracking_history = defaultdict(lambda: deque(maxlen=200))
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
                    confidence = min(1.0, (
                        0.4 * min(1.0, area / 20000) +  # Size factor
                        0.3 * min(1.0, abs(aspect_ratio - 1.4) / 0.5) +  # Aspect ratio factor
                        0.3 * rectangularity  # Shape factor
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
        """Improved winner card detection using multiple color spaces"""
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

        # Method 1: HSV red detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red ranges in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Method 2: Check for pink/magenta (hearts)
        lower_pink = np.array([140, 40, 40])
        upper_pink = np.array([170, 255, 255])
        pink_mask = cv2.inRange(hsv_roi, lower_pink, upper_pink)

        # Combine masks
        combined_mask = cv2.bitwise_or(red_mask, pink_mask)

        # Calculate red ratio
        red_pixels = cv2.countNonZero(combined_mask)
        total_pixels = w * h
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0

        # Also check BGR values for red dominance
        mean_color = cv2.mean(roi)[:3]  # B, G, R
        red_dominance = mean_color[2] > (mean_color[0] + mean_color[1]) * 0.6

        is_winner = (red_ratio > 0.05 or (red_ratio > 0.02 and red_dominance))

        return is_winner, red_ratio

    def match_cards_advanced(self, prev_cards, curr_cards):
        """Advanced card matching using position, size, and color"""
        if not prev_cards or not curr_cards:
            return {}

        # Build cost matrix
        n_prev = len(prev_cards)
        n_curr = len(curr_cards)
        cost_matrix = np.zeros((n_prev, n_curr))

        for i, prev_card in enumerate(prev_cards):
            prev_center = prev_card['center']
            prev_area = prev_card['area']

            for j, curr_card in enumerate(curr_cards):
                curr_center = curr_card['center']
                curr_area = curr_card['area']

                # Distance cost
                distance = np.sqrt(
                    (prev_center[0] - curr_center[0])**2 +
                    (prev_center[1] - curr_center[1])**2
                )

                # Area similarity cost
                area_diff = abs(prev_area - curr_area) / max(prev_area, curr_area)

                # Combined cost (lower is better)
                cost = distance + area_diff * 50
                cost_matrix[i, j] = cost

        # Find best matches using greedy assignment
        matches = {}
        used_curr = set()

        for i in range(n_prev):
            valid_matches = [(cost_matrix[i, j], j) for j in range(n_curr)
                           if j not in used_curr and cost_matrix[i, j] < 200]

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
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video Information:")
    print(f"   Path: {video_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")
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

            # Detect cards
            cards = tracker.detect_cards_improved(frame)
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

            # Process each card
            for i, card in enumerate(cards):
                x, y, w, h = card['bbox']
                cx, cy = card['center']
                card_id = card_ids.get(i, -1)

                # Detect winner (only in early frames)
                if tracker.winner_id is None and frame_count < 90:
                    is_winner, red_ratio = tracker.detect_winner_card(frame, card['bbox'])

                    if is_winner and red_ratio > 0.05:
                        tracker.winner_id = card_id
                        tracker.winner_initial_pos = (cx, cy)
                        tracker.stats['winner_detected_frame'] = frame_count
                        print(f"✅ Winner detected at frame {frame_count}")
                        print(f"   Card ID: {card_id}")
                        print(f"   Position: ({cx}, {cy})")
                        print(f"   Red ratio: {red_ratio:.2%}\n")

                # Update tracking history
                if card_id >= 0:
                    tracker.tracking_history[card_id].append((cx, cy))

                # Choose visualization
                color = colors[card_id % 3] if card_id >= 0 else (128, 128, 128)
                thickness = 5 if card_id == tracker.winner_id else 2

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, thickness)

                # Draw label
                label = f"Card {card_id}"
                if card_id == tracker.winner_id:
                    label += " ★ WINNER"

                # Add confidence score
                label += f" ({card['confidence']:.2f})"

                cv2.putText(annotated_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw center
                cv2.circle(annotated_frame, (cx, cy), 6, color, -1)

                # Draw trajectory for winner
                if card_id == tracker.winner_id and len(tracker.tracking_history[card_id]) > 1:
                    points = list(tracker.tracking_history[card_id])
                    for j in range(1, len(points)):
                        cv2.line(annotated_frame, points[j-1], points[j], (0, 255, 255), 3)

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
                progress = (frame_count / total_frames) * 100
                print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

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

    print(f"\n📊 Statistics:")
    print(f"   - Frames processed: {frame_count}")
    print(f"   - Cards detected: {tracker.stats['cards_detected']}")
    print(f"   - Avg detection confidence: {avg_confidence:.2f}")
    print(f"   - Tracking losses: {tracker.stats['tracking_losses']}")
    print(f"   - Success rate: {(1 - tracker.stats['tracking_losses']/frame_count)*100:.1f}%")

    # Save report
    report = {
        'video_info': {
            'path': video_path,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
            'duration': total_frames / fps
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
        if test_cap.isOpened():
            test_cap.release()
            print(f"✅ Output video verified: Readable")
        else:
            print(f"⚠️  Warning: Output video may be corrupted")
    else:
        print(f"❌ Output video was not created")

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python track_cards_fixed.py <video_path> [output_dir]")
        print("\nExample:")
        print("  python track_cards_fixed.py test.mp4")
        print("  python track_cards_fixed.py test.mp4 results")
        return

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return

    success = process_video_fixed(video_path, output_dir)

    if success:
        print("\n🎉 Processing completed successfully!")
    else:
        print("\n❌ Processing failed")

if __name__ == "__main__":
    main()

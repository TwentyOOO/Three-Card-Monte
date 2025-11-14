# Three-Card Monte Tracker

An advanced computer vision solution for tracking playing cards in Three-Card Monte video footage.

## Features

- **Multi-method Card Detection**: Combines color-based and edge-based detection for robust card identification
- **Advanced Tracking**: Uses position and size similarity for accurate card matching between frames
- **Winner Card Identification**: Automatically detects the red card (winner) using HSV color space analysis
- **Trajectory Analysis**: Tracks and visualizes the movement path of the winning card
- **Statistics & Reporting**: Generates detailed JSON reports with tracking metrics and confidence scores
- **Frame Enhancement**: Applies denoising and contrast enhancement for improved detection accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python track_cards_fixed.py <video_path>
```

With custom output directory:
```bash
python track_cards_fixed.py <video_path> <output_dir>
```

## Output

The tracker generates:
- **tracked_output_fixed.mp4**: Annotated video with card detection and tracking visualization
- **tracking_report_fixed.json**: Detailed statistics and tracking results
- **frame_*.jpg**: Keyframe snapshots at regular intervals

## How It Works

### Card Detection
1. Converts frame to HSV color space for light color detection
2. Applies edge detection to identify card boundaries
3. Uses morphological operations to clean up the detection mask
4. Filters contours by area, aspect ratio, and shape properties

### Card Tracking
1. Matches detected cards between consecutive frames
2. Uses position distance and area similarity as matching criteria
3. Maintains persistent card IDs across frames
4. Applies trajectory smoothing to reduce jitter

### Winner Detection
1. Identifies cards with red coloring (typical for winner in Three-Card Monte)
2. Uses HSV ranges for both pure red and pink/magenta colors
3. Validates with BGR color space analysis
4. Detects winner only in early frames (before shuffling starts)

## Configuration

Key parameters can be adjusted in the source code:

- **Card area thresholds**: `3000 < area < (width * height * 0.3)`
- **Aspect ratio range**: `0.5 < aspect_ratio < 2.0`
- **Confidence thresholds**: Adjustable in `detect_winner_card()`
- **Tracking match distance**: Max cost threshold of `200` pixels

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Supported video formats: MP4, AVI, MOV (depends on OpenCV build)

## Limitations & Future Improvements

- Currently optimized for standard Three-Card Monte setup
- Color detection may need adjustment for different lighting conditions
- Could benefit from deep learning-based detection for complex scenarios
- Video codec support depends on system OpenCV build

## License

[Add your license here]

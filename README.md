# People Detection and Tracking System

A computer vision project for real-time people detection, tracking, and counting using YOLO models. The system processes video streams (files and RTSP streams) to detect people, track them across frames, and count entries/exits within defined polygon regions.

## Features

- **Multiple YOLO Model Support**: YOLOv8n, YOLOv11n, YOLOv11s
- **Three Tracking Algorithms**: Custom CentroidTracker, SORT, and Ultralytics built-in tracking
- **Region-based Counting**: Define polygon zones for entry/exit counting
- **RTSP Stream Support**: Real-time processing of IP camera streams
- **Interactive Region Selection**: Visual tool for defining counting zones
- **Video Output**: Annotated video files with bounding boxes and track IDs

## Installation

### Dependencies

Make sure you have Python 3.8+ installed, then install the required packages:

```bash
pip install ultralytics opencv-python shapely numpy scipy
```

For RTSP stream processing with GStreamer:

```bash
# Ubuntu/Debian
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0

# Additional GStreamer plugins for RTSP
sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-rtsp
```

For SORT tracking (optional):
```bash
pip install sort-track
```

### YOLO Models

The system uses pre-trained YOLO models. Download them automatically on first run, or manually:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt  
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11s.pt
```

## Quick Start

### 1. Basic Video Tracking

```bash
# Simple YOLO tracking on video file
python3 track.py
```

### 2. People Counting with Region Analysis

```bash
# High-level people counting using Ultralytics ObjectCounter
python3 counter.py
```

### 3. Custom Centroid Tracker

```bash
# Advanced tracking with IoU-based matching
python3 centroid_tracker.py
```

### 4. RTSP Stream Processing

```bash
# Test RTSP connectivity and visualization
python3 rstp_test.py

# GStreamer-based RTSP with overlay graphics
python3 Gstreamer_rstp.py
```

## Usage Guide

### Defining Counting Regions

Use the interactive point selection tool to define polygon regions:

```bash
python3 points.py
```

Click on the video frame to select polygon vertices. The tool will output coordinates that you can use in other scripts.

### Configuration

#### Default Region Points
```python
region_points = [(3, 1530), (2043, 1533), (1813, 790), (0, 760)]
```

#### YOLO Models
- `yolo11n.pt` - Fastest, suitable for real-time processing
- `yolo11s.pt` - Balanced speed and accuracy  
- `yolov8n.pt` - Legacy model, good compatibility

#### Tracking Parameters
- **CentroidTracker**: `max_disappeared=100`, `max_distance=100`
- **SORT**: `max_age=50`, `min_hits=2`, `iou_threshold=0.3`
- **ByteTrack**: Uses `bytetrack.yaml` configuration

### Video Input Sources

#### Video Files
Place video files in the `data/` directory:
```python
video_path = "data/your_video.mp4"
```

#### RTSP Streams
Configure RTSP URL:
```python
rtsp_url = "rtsp://127.0.0.1:8554/mystream"
```

## Scripts Overview

| Script | Purpose | Algorithm |
|--------|---------|-----------|
| `track.py` | Basic YOLO tracking | Ultralytics built-in |
| `counter.py` | People counting with regions | Ultralytics ObjectCounter |
| `centroid_tracker.py` | Custom tracker with IoU | Custom CentroidTracker |
| `sort_tracker.py` | SORT algorithm integration | SORT |
| `temp.py` | Simple tracking with region check | ByteTrack |
| `points.py` | Interactive region selection | - |
| `rstp_test.py` | RTSP connectivity testing | - |
| `Gstreamer_rstp.py` | GStreamer RTSP processing | - |

## Architecture

### Detection Pipeline
1. **Frame Input**: Video file or RTSP stream
2. **YOLO Inference**: Detect people (class 0) with confidence > 0.25-0.4
3. **Tracking**: Associate detections across frames using chosen algorithm
4. **Region Analysis**: Check if centroids are within polygon zones
5. **Visualization**: Draw bounding boxes, IDs, and region overlays
6. **Output**: Save annotated video as `.avi` file

### Tracking Algorithms

#### CentroidTracker (`centroid_tracker.py`)
- **Best for**: Complex scenarios with occlusions
- **Method**: Combines centroid distance (20%) and IoU similarity (80%)
- **Features**: Handles object disappearance, robust to occlusions
- **Performance**: Higher computational cost but better accuracy

#### SORT (`sort_tracker.py`)  
- **Best for**: Simple tracking scenarios
- **Method**: Kalman filtering with Hungarian assignment
- **Features**: Fast, lightweight implementation
- **Performance**: Lower computational cost, good for real-time

#### Ultralytics Built-in (`counter.py`, `temp.py`)
- **Best for**: Quick prototyping and high-level counting
- **Method**: ByteTrack integration
- **Features**: Well-integrated with YOLO models, easy to use
- **Performance**: Balanced speed and accuracy

## Performance Tips

- **Real-time Processing**: Use `yolo11n.pt` model
- **Display Optimization**: Set `display_scale=0.5` to reduce visualization load
- **High Resolution**: Consider frame skipping for very large video streams
- **RTSP Reliability**: Use GStreamer pipeline for production RTSP streams

## Output

The system generates:
- **Annotated Videos**: `.avi` files with bounding boxes and track IDs
- **Console Logs**: Entry/exit events with timestamps
- **Region Visualization**: Polygon overlays showing counting zones

Example output:
```
[Fri Sep  6 13:45:23 2024] ID 15 ENTERED zone.
[Fri Sep  6 13:45:28 2024] ID 15 LEFT zone after 5.2s
```

## Testing Dependencies

Verify your installation:

```bash
# Test YOLO
python3 -c "from ultralytics import YOLO; print('✓ YOLO available')"

# Test OpenCV
python3 -c "import cv2; print('✓ OpenCV:', cv2.__version__)"

# Test GStreamer (for RTSP)
python3 -c "import gi; gi.require_version('Gst', '1.0'); print('✓ GStreamer OK')"

# Test Shapely
python3 -c "from shapely.geometry import Point, Polygon; print('✓ Shapely OK')"
```

## File Structure

```
├── data/                          # Video files (gitignored)
├── *.pt                          # YOLO model weights (gitignored)  
├── *.avi                         # Output videos (gitignored)
├── centroid_tracker.py           # Custom IoU-based tracker
├── counter.py                    # Ultralytics ObjectCounter
├── Gstreamer_rstp.py            # GStreamer RTSP processing
├── points.py                     # Interactive region selection
├── rstp_test.py                  # RTSP connectivity testing
├── sort_tracker.py               # SORT algorithm integration
├── temp.py                       # Simple tracking with regions
├── track.py                      # Basic YOLO tracking
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Troubleshooting

### Common Issues

**RTSP Connection Failed**
- Check camera IP and port
- Verify network connectivity
- Ensure RTSP credentials if required

**Low Detection Accuracy**
- Adjust confidence threshold (0.25-0.4)
- Try different YOLO models (`yolo11s.pt` for better accuracy)
- Check lighting conditions

**Tracking Issues**
- Increase `max_disappeared` parameter for CentroidTracker
- Adjust IoU threshold for SORT
- Consider switching tracking algorithms

**Performance Issues**
- Use smaller YOLO model (`yolo11n.pt`)
- Reduce display scale
- Skip frames if real-time processing isn't required

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with different video inputs
4. Submit a pull request

## License

This project is open source. Please ensure you comply with the licenses of dependencies (Ultralytics YOLO, OpenCV, etc.).

# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a computer vision project focused on people detection, tracking, and counting using YOLO models. The system processes video streams (both files and RTSP streams) to detect people, track them across frames, and count entries/exits within defined polygon regions.

## Core Architecture

### Detection & Tracking Pipeline
- **YOLO Models**: Uses YOLOv8/v11 models (`yolov8n.pt`, `yolo11n.pt`, `yolo11s.pt`) for person detection (class 0)
- **Tracking Algorithms**: Three different tracking implementations:
  - Custom CentroidTracker with IoU-based matching (`centroid_tracker.py`)
  - SORT tracker integration (`sort_tracker.py`) 
  - Ultralytics built-in tracking (`counter.py`, `temp.py`)
- **Region Analysis**: Uses Shapely polygons to define counting zones and detect entry/exit events

### Key Components
- **`centroid_tracker.py`**: Custom tracker combining centroid distance and IoU for robust person tracking
- **`counter.py`**: High-level people counting using Ultralytics ObjectCounter with region definitions
- **`Gstreamer_rstp.py`**: GStreamer-based RTSP stream processing with Cairo overlay graphics
- **`points.py`**: Interactive tool for polygon region point selection
- **`rstp_test.py`**: RTSP stream connectivity testing and region visualization

### Video Processing Flow
1. **Input**: Video files (`data/*.mp4`) or RTSP streams (`rtsp://127.0.0.1:8554/mystream`)
2. **Detection**: YOLO inference on each frame to detect people (confidence > 0.25-0.4)
3. **Tracking**: Associate detections across frames using chosen tracking algorithm
4. **Region Analysis**: Check if person centroids are within defined polygon zones
5. **Output**: Annotated video files (`.avi` format) with bounding boxes, IDs, and counts

## Common Development Commands

### Running People Detection and Tracking
```bash
# Basic YOLO tracking on video file
python3 track.py

# People counting with region analysis
python3 counter.py

# Custom centroid tracker with advanced IoU matching
python3 centroid_tracker.py

# SORT-based tracking implementation  
python3 sort_tracker.py

# Simple tracking with region checking
python3 temp.py
```

### RTSP Stream Processing
```bash
# Test RTSP stream connectivity
python3 rstp_test.py

# GStreamer-based RTSP with overlay graphics
python3 Gstreamer_rstp.py
```

### Region Definition
```bash
# Interactive polygon point selection tool
python3 points.py
```

### Testing Key Dependencies
```bash
# Verify YOLO installation
python3 -c "from ultralytics import YOLO; print('YOLO available')"

# Check OpenCV version
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"

# Test GStreamer (for RTSP)
python3 -c "import gi; gi.require_version('Gst', '1.0'); print('GStreamer OK')"
```

## Configuration Parameters

### Common Detection Settings
- **Person Class ID**: `0` (COCO dataset)
- **Confidence Threshold**: `0.25-0.4` (adjust based on video quality)
- **YOLO Models**: `yolo11n.pt` (fastest), `yolo11s.pt` (balanced), or `yolov8n.pt`

### Tracking Parameters
- **CentroidTracker**: `max_disappeared=100`, `max_distance=100`, IoU weight=0.8
- **ByteTrack**: Uses `bytetrack.yaml` configuration
- **SORT**: `max_age=50`, `min_hits=2`, `iou_threshold=0.3`

### Region Definitions
Default polygon points for counting zones are defined as:
```python
region_points = [(3, 1530), (2043, 1533), (1813, 790), (0, 760)]
```
Use `points.py` to interactively define new regions for different camera angles.

### RTSP Configuration
- **Default Stream**: `rtsp://127.0.0.1:8554/mystream`
- **GStreamer Pipeline**: Includes decodebin, videoconvert, and appsink
- **Reconnection**: Automatic stream reconnection on connection loss

## File Structure Context

- **Model Files**: `*.pt` files are pre-trained YOLO weights
- **Video Data**: `data/` contains test footage files (MP4 format)
- **Output Videos**: Generated `.avi` files with tracking annotations
- **Ignored Files**: `.gitignore` excludes `data/`, `*.pt`, and `*.avi` files

## Architecture Notes

### Tracking Algorithm Comparison
- **CentroidTracker**: Best for complex scenarios with occlusions, uses combined centroid+IoU distance
- **SORT**: Faster, suitable for simple tracking scenarios
- **Ultralytics**: Most convenient, well-integrated with YOLO models

### Performance Considerations
- Use `yolo11n.pt` for real-time performance
- Set `display_scale=0.5` to reduce visualization load
- Consider frame skipping for very high-resolution streams
- IoU-based matching in CentroidTracker provides better accuracy but higher computational cost

### Stream Processing
- GStreamer pipeline required for reliable RTSP handling
- Cairo overlay system for real-time graphics rendering
- Automatic reconnection logic essential for production RTSP streams

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Settings
# ----------------------------
video_path = "data/2025-04-22_08-34-09_footage.mp4"
model_path = "yolo11s.pt"
region_points = [(3, 1530), (2043, 1533), (1813, 790), (0, 760)]  # polygon
display_scale = 0.5  # shrink for display
tracker_cfg = "bytetrack.yaml"

# ----------------------------
# Init model and video
# ----------------------------
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter(
    "object_counting_output_1.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# ----------------------------
# Main loop
# ----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with tracking
    results = model.track(
        frame, tracker=tracker_cfg, persist=True, classes=[0], conf=0.25  # only persons
    )

    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:  # if tracking IDs exist
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if center is inside polygon
            inside = (
                cv2.pointPolygonTest(np.array(region_points, np.int32), (cx, cy), False)
                >= 0
            )

            # Draw bounding box
            color = (0, 255, 0) if inside else (0, 0, 255)  # green if inside
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw center
            cv2.circle(annotated_frame, (cx, cy), 4, color, -1)

            # Draw ID
            cv2.putText(
                annotated_frame,
                f"ID {int(track_id)}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    # Draw polygon region
    cv2.polylines(
        annotated_frame,
        [np.array(region_points, np.int32)],
        isClosed=True,
        color=(255, 255, 0),
        thickness=2,
    )

    # Save full-res output
    video_writer.write(annotated_frame)

    # Show scaled output
    display_frame = cv2.resize(
        annotated_frame, None, fx=display_scale, fy=display_scale
    )
    cv2.imshow("YOLO Tracking + Centers", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

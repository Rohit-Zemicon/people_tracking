import time
import numpy as np
import cv2
from shapely.geometry import Point, Polygon

# import the packaged SORT
from sort.tracker import SortTracker  # from sort-track package

# --- configure ---
# tune these for your CCTV entrance
SORT_KWARGS = dict(
    max_age=50, min_hits=2, iou_threshold=0.3
)  # example kwargs (package accepts common params)
tracker = SortTracker(**SORT_KWARGS)

# polygon region (image coordinates)
polygon_pts = np.array([[100, 100], [500, 100], [500, 400], [100, 400]])
zone = Polygon(polygon_pts)
inside_zone = {}  # track_id -> enter_time


# helper: convert your detector output to sort format
def detections_to_sort_array(detections):
    """
    detections: list of [x1,y1,x2,y2,score,class_id]  (your YOLO outputs)
    returns: np.ndarray shape (N,5) -> [[x1,y1,x2,y2,score], ...] only person class
    """
    dets = []
    for det in detections:
        x1, y1, x2, y2, score, cls = det
        if cls == 0 and score > 0.3:  # assume person class == 0; tune threshold
            dets.append([float(x1), float(y1), float(x2), float(y2), float(score)])
    return np.array(dets) if len(dets) > 0 else np.empty((0, 5))


# --- main frame loop (replace YOLO placeholder with your inference) ---
cap = cv2.VideoCapture(0)  # replace with RTSP or video file
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- RUN YOUR YOLO INFERENCE HERE ----
    # Example expected output format (replace this with your actual YOLO output)
    # each detection = [x1, y1, x2, y2, score, class_id]
    # detections = yolo_infer(frame)
    detections = []  # <-- replace with real detections
    # ---------------------------------------

    # convert detections to SORT input
    dets = detections_to_sort_array(detections)

    # update tracker (sort-track usually accepts numpy array of shape (N,5))
    # online_targets typical output: list of tracked objects; may be [[x1,y1,x2,y2, id], ...]
    online_targets = tracker.update(dets)

    # Draw polygon
    cv2.polylines(frame, [polygon_pts], True, (0, 255, 0), 2)

    # Iterate tracks and check polygon membership
    for tr in online_targets:
        # depending on package version the track format might be (x1,y1,x2,y2,tid) or an object
        # try to handle common formats safely:
        try:
            x1, y1, x2, y2, tid = tr  # list/tuple-like
        except Exception:
            # if tracks are dicts/objects, adapt here (fallback)
            # Example: tr.bbox, tr.track_id
            x1, y1, x2, y2 = tr.bbox
            tid = tr.track_id

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centroid = Point(cx, cy)
        inside = zone.contains(centroid)

        if inside and tid not in inside_zone:
            inside_zone[tid] = time.time()
            print(f"[{time.ctime()}] ID {tid} ENTERED zone.")
        if (not inside) and tid in inside_zone:
            entered = inside_zone.pop(tid)
            print(
                f"[{time.ctime()}] ID {tid} LEFT zone after {time.time()-entered:.1f}s"
            )

        col = (0, 200, 0) if inside else (0, 120, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            frame,
            f"ID:{tid}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.circle(frame, (cx, cy), 3, col, -1)

    cv2.imshow("SORT-track", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

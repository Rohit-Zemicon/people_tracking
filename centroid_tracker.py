# centroid_tracker.py
import numpy as np
import cv2
from collections import OrderedDict
from shapely.geometry import Point, Polygon
import time


class CentroidTracker:
    def __init__(self, max_disappeared=100, max_distance=100):
        self.next_object_id = 0
        self.objects = OrderedDict()  # object_id -> centroid (x,y)
        self.bboxes = OrderedDict()  # object_id -> last bbox (x1,y1,x2,y2)
        self.disappeared = OrderedDict()  # object_id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        # For tracking stability
        self.previous_rects = []  # Store previous frame detections for smoothing

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection coordinates
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)

        # Calculate intersection area
        if x2_int > x1_int and y2_int > y1_int:
            intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        else:
            intersection_area = 0

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area

        # Calculate IoU
        if union_area == 0:
            return 0
        iou = intersection_area / union_area
        # Ensure IoU is within valid range [0, 1]
        return np.clip(iou, 0.0, 1.0)

    def update(self, rects):
        """
        rects: list of bbox tuples (x1,y1,x2,y2)
        returns: dict object_id -> bbox
        """
        if len(rects) == 0:
            # mark all existing as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.bboxes.copy()

        # compute input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(tuple(input_centroids[i]), rects[i])
        else:
            # build arrays
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            existing_bboxes = list(self.bboxes.values())

            # distance matrix (object x input)
            D_centroid = np.linalg.norm(
                object_centroids[:, None] - input_centroids[None, :], axis=2
            )

            # IoU matrix (object x input)
            D_iou = np.zeros((len(existing_bboxes), len(rects)))
            for i, existing_bbox in enumerate(existing_bboxes):
                for j, new_bbox in enumerate(rects):
                    iou = self._calculate_iou(existing_bbox, new_bbox)
                    # Convert IoU to distance (1 - IoU)
                    D_iou[i, j] = 1 - iou

            # Combine centroid distance and IoU distance
            # Weight IoU more heavily (80% IoU, 20% centroid distance)
            D = 0.8 * D_iou + 0.2 * D_centroid

            # Ensure we don't have NaN or infinite values
            D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)

            # Use Hungarian algorithm for optimal assignment
            try:
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(D)
                used_rows, used_cols = set(rows), set(cols)

                # Update matched objects
                for r, c in zip(rows, cols):
                    # Check if the match is acceptable (combined distance threshold)
                    if (
                        D[r, c] <= 0.9
                    ):  # Even more lenient threshold for combined metric
                        oid = object_ids[r]
                        self.objects[oid] = tuple(input_centroids[c])
                        self.bboxes[oid] = rects[c]
                        self.disappeared[oid] = 0
                    else:
                        # Distance too large, consider as disappeared or new object
                        used_rows.remove(r)
                        used_cols.remove(c)
            except ImportError:
                # Fallback to original greedy approach if scipy is not available
                # find smallest distance pairs
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows, used_cols = set(), set()
                for r, c in zip(rows, cols):
                    if r in used_rows or c in used_cols:
                        continue
                    # Check if the match is acceptable (combined distance threshold)
                    if D[r, c] > 0.9:  # Even more lenient threshold for combined metric
                        continue
                    oid = object_ids[r]
                    self.objects[oid] = tuple(input_centroids[c])
                    self.bboxes[oid] = rects[c]
                    self.disappeared[oid] = 0
                    used_rows.add(r)
                    used_cols.add(c)

            # unmatched objects -> disappeared++
            unused_rows = set(range(0, D.shape[0])) - used_rows
            for r in unused_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            # unmatched input detections -> register new objects
            unused_cols = set(range(0, D.shape[1])) - used_cols
            for c in unused_cols:
                self.register(tuple(input_centroids[c]), rects[c])

        # Store current rects for next frame
        self.previous_rects = rects.copy()

        return self.bboxes.copy()


# Example integration with YOLO outputs and polygon checking
if __name__ == "__main__":
    # Suppose you already have a video capture / frame loop and YOLO detections per frame.
    # YOLO detections format assumed: list of [x1,y1,x2,y2,confidence,class_id]
    from ultralytics import YOLO

    ct = CentroidTracker(max_disappeared=100, max_distance=100)

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # or yolo11n.pt, yolo11s.pt

    # define polygon in image coordinates (example)
    polygon_pts = np.array([(3, 1530), (2043, 1533), (1813, 790), (0, 760)])
    zone = Polygon(polygon_pts)
    inside_zone = {}  # object_id -> entered_at_timestamp

    cap = cv2.VideoCapture(
        "data/2025-04-22_08-34-09_footage.mp4"
    )  # or path to CCTV RTSP with appropriate pipeline

    assert cap.isOpened(), "Error reading video file"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(
        "people_track_iid.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            H, W = frame.shape[:2]

            # YOLOv8 detection
            results = model(frame)
            detections = []

            # Convert YOLO results to detections format [x1,y1,x2,y2,conf,class_id]
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        detections.append(
                            [int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)]
                        )

            # Filter person class (e.g., class_id == 0) and build rects
            person_rects = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 0 and conf > 0.4:  # class 0 is person in COCO dataset
                    person_rects.append((int(x1), int(y1), int(x2), int(y2)))

            objects = ct.update(person_rects)  # object_id -> bbox

            # draw polygon
            cv2.polylines(frame, [polygon_pts], True, (0, 255, 0), 2)

            # check each tracked object against polygon
            for oid, bbox in objects.items():
                x1, y1, x2, y2 = bbox
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                centroid = Point(cX, cY)
                currently_inside = zone.contains(centroid)
                if currently_inside and oid not in inside_zone:
                    inside_zone[oid] = time.time()
                    print(f"[{time.ctime()}] ID {oid} ENTERED zone.")
                if (not currently_inside) and oid in inside_zone:
                    entered_at = inside_zone.pop(oid)
                    spent = time.time() - entered_at
                    print(f"[{time.ctime()}] ID {oid} LEFT zone after {spent:.1f}s")

                # visualization
                color = (0, 200, 0) if currently_inside else (0, 120, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{oid}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.circle(frame, (cX, cY), 3, color, -1)

            # Write frame to video file
            video_writer.write(frame)

            display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

            cv2.imshow("track", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except KeyboardInterrupt:
            print("Exiting...")
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

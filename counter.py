import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("data/2025-04-22_08-34-09_footage.mp4")
assert cap.isOpened(), "Error reading video file"

# region_points = [(20, 400), (1080, 400)]                                      # line counting
# region_points = [(258, 338), (587, 345), (607, 5), (236, 5)]  # rectangle region
# region_points = [(582, 756), (1284, 764), (1282, 754), (584, 742)]  # rectangle region
# region_points = [(562, 814), (1294, 826), (1294, 806), (574, 790)]
region_points = [(3, 1530), (2043, 1533), (1813, 790), (0, 760)]
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

# Video writer
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo11s.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    classes=[
        0
    ],  # count specific classes i.e. person and car with COCO pretrained model.
    tracker="bytetrack.yaml",  # choose trackers i.e "botsort.yaml"
)

display_scale = 0.5  # 50% of original size

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    print(results)  # access the output

    # for det in results.detections:
    #     x1, y1, x2, y2 = det["bbox"]  # bounding box coords
    #     track_id = det.get("id", None)  # tracker ID

    #     cx = int((x1 + x2) / 2)
    #     cy = int((y1 + y2) / 2)

    #     # Draw center
    #     cv2.circle(results.plot_im, (cx, cy), 4, (0, 255, 0), -1)

    #     # Draw ID if available
    #     if track_id is not None:
    #         cv2.putText(
    #             results.plot_im,
    #             str(int(track_id)),
    #             (cx + 5, cy - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0, 255, 0),
    #             1,
    #         )

    video_writer.write(results.plot_im)  # write the processed frame.

    # Show resized output
    display_frame = cv2.resize(
        results.plot_im, None, fx=display_scale, fy=display_scale
    )
    # cv2.imshow("YOLO Object Counting", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows

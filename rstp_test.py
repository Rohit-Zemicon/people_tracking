# import cv2
# import numpy as np

# # RTSP stream URL (replace with yours)
# rtsp_url = "rtsp://127.0.0.1:8554/mystream"

# region_points = [(3, 1530), (2043, 1533), (1813, 790), (0, 760)]  # polygon

# display_scale = 0.5

# cap = cv2.VideoCapture(rtsp_url)

# if not cap.isOpened():
#     print("Error: Cannot open RTSP stream")
#     exit()


# # print(cap.frame_width, cap.frame_height)

# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# print(f"Resolution: {int(width)} x {int(height)}")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     cv2.polylines(
#         frame,
#         [np.array(region_points, np.int32)],
#         isClosed=True,
#         color=(255, 255, 0),
#         thickness=2,
#     )

#     display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

#     # Display frame
#     cv2.imshow("RTSP Stream", display_frame)

#     # Quit with 'q'
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

region_points = [(3, 1530), (2043, 1533), (1813, 790), (0, 760)]  # polygon

pipeline = (
    "rtspsrc location=rtsp://127.0.0.1:8554/mystream ! "
    "decodebin ! videoconvert ! appsink"
)

# pipeline = (
#     f"appsrc ! video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
#     f"videoconvert ! x264enc tune=zerolatency bitrate=4000 speed-preset=superfast ! "
#     f"rtph264pay ! udpsink host={host_ip} port={port}"
# )

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Resolution: {int(width)} x {int(height)}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream lost. Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        continue

    # Draw polygon
    cv2.polylines(
        frame,
        [np.array(region_points, np.int32).reshape((-1, 1, 2))],
        isClosed=True,
        color=(255, 255, 0),
        thickness=2,
    )

    # Optional resize for speed
    frame = cv2.resize(frame, (960, 540))

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

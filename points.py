# import cv2

# points = []


# def click_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         print(f"Point selected: ({x}, {y})")


# # Load a sample frame
# frame = cv2.imread("frame.jpg")  # or use first frame from video
# cv2.imshow("Frame", frame)
# cv2.setMouseCallback("Frame", click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("Final points:", points)


import cv2

points = []
scale = 0.3  # shrink to 50% for display


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # scale back up to original coords
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        points.append((orig_x, orig_y))
        print(f"Point selected: ({orig_x}, {orig_y})")
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Frame", param)


# Load first frame
# cap = cv2.VideoCapture("data/2025-04-22_08-34-09_footage.mp4")
cap = cv2.VideoCapture("rtsp://192.168.1.10:8554/mystream")
ret, frame = cap.read()
cap.release()

# Resize for display
display_frame = cv2.resize(frame, None, fx=scale, fy=scale)

cv2.imshow("Frame", display_frame)
cv2.setMouseCallback("Frame", click_event, display_frame.copy())
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Final points:", points)

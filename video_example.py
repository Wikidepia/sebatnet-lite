import sys
import time

import cv2

from sebatnet import SebatNet

if len(sys.argv) != 2:
    print(f"[ERROR] Usage: python3 {sys.argv[0]} video.mp4")
    raise SystemExit

sebatnet = SebatNet()
vid_capture = cv2.VideoCapture(sys.argv[1])
prev_frame_time = 0

while True:
    ret, orig_image = vid_capture.read()
    if orig_image is None:
        break

    smokers = sebatnet.find_smokers(orig_image)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    for smoker in smokers:
        (startX, startY, endX, endY) = smoker["coords"]
        label = "Smoking" if smoker["is_smoking"] else "Not Smoking"

        cv2.putText(
            orig_image,
            label,
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        cv2.rectangle(orig_image, (startX, startY), (endX, endY), (255, 255, 0), 4)

    fps = str(round(fps))
    # puting the FPS count on the frame
    cv2.putText(
        orig_image,
        fps,
        (0, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (100, 255, 0),
        3,
        cv2.LINE_AA,
    )

    orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow(f"SMOKING_DETECTOR", orig_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
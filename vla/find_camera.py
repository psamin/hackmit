"""Save one frame from each camera index so you can see which one is the arm camera.

    python vla/find_camera.py        # writes camera_<index>.jpg to /tmp/cameras and opens the folder
The first run asks for camera permission for your terminal app; allow it and run again.
"""
import pathlib, subprocess, time

import cv2

out = pathlib.Path("/tmp/cameras")
out.mkdir(exist_ok=True)
for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    ok, frame = False, None
    for _ in range(20):  # cameras need a few frames to expose
        ok, frame = cap.read()
        if ok and frame.mean() > 5:
            break
        time.sleep(0.1)
    cap.release()
    if ok:
        cv2.imwrite(str(out / f"camera_{index}.jpg"), frame)
        print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]} -> {out}/camera_{index}.jpg")
    else:
        print(f"camera {index}: none")
subprocess.run(["open", str(out)])

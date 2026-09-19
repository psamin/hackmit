"""Blocker 4b: which detector finds hands holding objects in head-camera frames?
All 16 frames in data/hand_frames/ show at least one hand, checked by eye. Hit = any hand box at conf >= 0.10."""
import glob, time
import cv2
from ultralytics import YOLO, YOLOE, YOLOWorld

frames = sorted(glob.glob("data/hand_frames/*.jpg"))

def run(name, predict):
    predict(frames[0])
    hits, t = 0, 0.0
    for f in frames:
        t0 = time.perf_counter(); n = predict(f); t += time.perf_counter() - t0
        hits += n > 0
    print(f"{name:34s} {hits:2d}/{len(frames)}  {1000 * t / len(frames):4.0f} ms", flush=True)

def ultra(model, keep):
    def p(f):
        r = model.predict(f, device="mps", conf=0.10, verbose=False)[0]
        return sum(r.names[int(c)] in keep for c in r.boxes.cls)
    return p

for w in ("yoloe-26s-seg", "yoloe-26l-seg"):
    for prompt in ("hand", "human hand", "arm", "forearm"):
        m = YOLOE(f"weights/{w}.pt"); m.set_classes([prompt, "bottle"])
        run(f"{w} '{prompt}'", ultra(m, {prompt}))
m = YOLOWorld("weights/yolov8x-worldv2.pt"); m.set_classes(["hand", "bottle"])
run("yolov8x-worldv2 'hand'", ultra(m, {"hand"}))
run("yolo26n COCO 'person'", ultra(YOLO("weights/yolo26n.pt"), {"person"}))

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="weights/hand_landmarker.task", delegate=BaseOptions.Delegate.CPU), num_hands=2,
    min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
run("mediapipe hand_landmarker", lambda f: len(lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
    data=cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))).hand_landmarks))

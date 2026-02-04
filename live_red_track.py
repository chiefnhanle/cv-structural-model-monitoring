import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, Response

# -----------------------
# Camera config
# -----------------------
DEVICE = "/dev/video0"
WIDTH, HEIGHT, FPS = 640, 480, 30

# -----------------------
# Red detection config (tune these)
# -----------------------
# HSV ranges for red (red wraps around hue=0)
LOW1  = (0,   120, 70)
HIGH1 = (10,  255, 255)
LOW2  = (170, 120, 70)
HIGH2 = (180, 255, 255)

MIN_AREA = 30          # reject tiny specks
MAX_AREA = 20000       # reject giant blobs
MORPH_K  = 5           # kernel size for morphology

# -----------------------
# Tracking config
# -----------------------
MAX_MATCH_DIST = 35.0  # pixels: max distance to associate detection to an existing track
TTL_FRAMES = 15        # frames: how long to keep a track without seeing it
TRAIL_LEN = 40         # max trail points stored

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
cap = None

@dataclass
class Track:
    tid: int
    pos: Tuple[float, float]
    last_seen: int
    trail: List[Tuple[int, int]] = field(default_factory=list)

tracks: Dict[int, Track] = {}
next_id = 1
frame_idx = 0


def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {DEVICE}. Make sure usbipd attach worked and nothing else uses the camera.")

    # Force MJPG for stability over USB/IP
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    time.sleep(1.0)
    return cap


def detect_red_centroids(bgr: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Returns list of (cx, cy, area) for detected red blobs.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array(LOW1), np.array(HIGH1))
    mask2 = cv2.inRange(hsv, np.array(LOW2), np.array(HIGH2))
    mask = cv2.bitwise_or(mask1, mask2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets: List[Tuple[int, int, int]] = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < MIN_AREA or area > MAX_AREA:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        dets.append((cx, cy, area))

    # optional: stable ordering (left->right)
    dets.sort(key=lambda x: x[0])
    return dets


def assign_tracks(dets: List[Tuple[int, int, int]]):
    """
    Nearest-neighbor association between detections and existing tracks.
    Creates new tracks for unmatched detections.
    """
    global tracks, next_id, frame_idx

    # Mark all tracks unmatched initially
    track_ids = list(tracks.keys())
    unmatched_tracks = set(track_ids)
    unmatched_dets = set(range(len(dets)))

    # Build distance matrix (track x det)
    pairs = []
    for tid in track_ids:
        tx, ty = tracks[tid].pos
        for di, (cx, cy, _) in enumerate(dets):
            dist = math.hypot(cx - tx, cy - ty)
            pairs.append((dist, tid, di))

    # Greedy match by smallest distance
    pairs.sort(key=lambda p: p[0])
    for dist, tid, di in pairs:
        if dist > MAX_MATCH_DIST:
            break
        if tid in unmatched_tracks and di in unmatched_dets:
            # match
            tracks[tid].pos = (float(dets[di][0]), float(dets[di][1]))
            tracks[tid].last_seen = frame_idx
            px, py = int(tracks[tid].pos[0]), int(tracks[tid].pos[1])
            tracks[tid].trail.append((px, py))
            if len(tracks[tid].trail) > TRAIL_LEN:
                tracks[tid].trail = tracks[tid].trail[-TRAIL_LEN:]
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

    # Create tracks for unmatched detections
    for di in unmatched_dets:
        cx, cy, _ = dets[di]
        tracks[next_id] = Track(
            tid=next_id,
            pos=(float(cx), float(cy)),
            last_seen=frame_idx,
            trail=[(cx, cy)]
        )
        next_id += 1

    # Prune stale tracks
    stale = [tid for tid, t in tracks.items() if (frame_idx - t.last_seen) > TTL_FRAMES]
    for tid in stale:
        del tracks[tid]


def draw_overlay(frame: np.ndarray, dets: List[Tuple[int, int, int]]) -> np.ndarray:
    # Draw trails + IDs
    for tid, t in tracks.items():
        # trail
        if len(t.trail) >= 2:
            for i in range(1, len(t.trail)):
                cv2.line(frame, t.trail[i-1], t.trail[i], (255, 255, 255), 1)

        x, y = int(t.pos[0]), int(t.pos[1])
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(frame, f"ID {tid}", (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw detection circles (thin)
    for (cx, cy, area) in dets:
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)
        cv2.putText(frame, f"A={area}", (cx + 8, cy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Red dots detected: {len(dets)} | Tracks: {len(tracks)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def frame_generator():
    global cap, frame_idx

    if cap is None:
        cap = open_camera()

    while True:
        frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        dets = detect_red_centroids(frame)
        assign_tracks(dets)
        frame = draw_overlay(frame, dets)

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


@app.route("/")
def index():
    return (
        "<h2>Live Red Dot Tracking</h2>"
        "<p>Open <a href='/video'>/video</a></p>"
        "<img src='/video' style='max-width: 100%; height: auto;' />"
    )


@app.route("/video")
def video():
    return Response(frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

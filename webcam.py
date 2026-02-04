import time
import cv2
from flask import Flask, Response

DEVICE = "/dev/video0"
WIDTH, HEIGHT, FPS = 640, 480, 30

app = Flask(__name__)

def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {DEVICE}. Is it attached and not used elsewhere?")

    # Force MJPG (common fix for USB/IP + Logitech)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Give the driver a moment to start streaming
    time.sleep(1.0)
    return cap

cap = None

def frame_generator():
    global cap
    if cap is None:
        cap = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            # If the camera stalls, retry after a short pause
            time.sleep(0.2)
            continue

        # Encode as JPEG
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/")
def index():
    return (
        "<h2>WSL Live Webcam Feed</h2>"
        "<p>Stream: <a href='/video'>/video</a></p>"
        "<img src='/video' style='max-width: 100%; height: auto;' />"
    )

@app.route("/video")
def video():
    return Response(frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # host=0.0.0.0 lets Windows access it via localhost too
    app.run(host="0.0.0.0", port=5000, threaded=True)

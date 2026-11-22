
from flask import Flask, request, jsonify, Response, render_template
from datetime import datetime
import json
import threading
import time

app = Flask(__name__)

# Du lieu moi nhat
latest_data = {}

# Luu anh JPEG, khong can giai nen roi nen lai
latest_frame = None

lock = threading.Lock()

# Gioi han kich thuoc anh nhan duoc: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# Trang chu
@app.route("/")
def home():
    return render_template("index.html")


# Nhan anh JPEG tu Windows
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global latest_frame

    image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({"error": "Khong co du lieu anh"}), 400

    with lock:
        latest_frame = image_bytes

    return jsonify({"status": "success"})


# Nhan du lieu JSON tu AI
@app.route("/receive", methods=["POST"])
def receive_data():
    global latest_data

    try:
        data = request.get_json(force=True)

        if not isinstance(data, dict):
            return jsonify({"error": "JSON phai la object"}), 400

        with lock:
            latest_data = {
                "time": datetime.now().strftime("%H:%M:%S"),
                **data
            }

        print(json.dumps(
            latest_data,
            indent=4,
            ensure_ascii=False
        ))

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Phat camera MJPEG
def generate_frames():
    while True:
        with lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() +
            b"\r\n\r\n" + frame + b"\r\n"
        )

        # Gioi han toc do phat toi da 15 FPS
        time.sleep(1 / 15)


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache"
        }
    )


# Tra ve du lieu AI
@app.route("/data")
def get_data():
    with lock:
        data = latest_data.copy()

    return jsonify(data)


if __name__ == "__main__":
    print("================================")
    print("JETSON ORIN AI CAMERA SERVER")
    print("http://0.0.0.0:5000")
    print("================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )

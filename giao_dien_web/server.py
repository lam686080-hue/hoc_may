"""
Flask server hiển thị luồng camera trực tiếp và kết quả nhận diện AI.
Chạy:
  python3 giao_dien_web/server.py          (mặc định: nhận khung hình từ camera_sender.py và tự nhận diện trên Orin)
  python3 giao_dien_web/server.py --relay  (chỉ phát lại khung hình + JSON do buoc6_nhan_dien.py --server gửi sang)
"""
from flask import Flask, request, jsonify, Response, render_template
from datetime import datetime
import argparse
from pathlib import Path
import sys
import json
import threading
import time

app = Flask(__name__)

# Du lieu moi nhat
latest_data = {}

# Luu anh JPEG, khong can giai nen roi nen lai
latest_frame = None

lock = threading.Lock()

# Chỉ giữ ảnh mới nhất; worker nhận quyền xử lý mỗi ảnh đúng một lần.
camera = threading.Condition()
pending = None
relay = False
TIMEOUT_CAMERA = 2.0


def ket_qua(jpg, data):
    global latest_frame, latest_data
    with lock:
        latest_frame = jpg
        latest_data = {"time": datetime.now().strftime("%H:%M:%S"), **data} if data else {}


def nhan_dien(nd, lm, stop):
    global pending
    import buoc6_nhan_dien as B
    import cv2
    import numpy as np
    xoay = B.T.XOAY_DU_PHONG if isinstance(B.T.XOAY, str) else B.T.XOAY
    last = None
    ts = 0
    fps = 0
    while not stop.is_set():
        with camera:
            camera.wait_for(lambda: pending is not None or stop.is_set(), timeout=0.25)
            item, pending = pending, None
        if stop.is_set():
            break
        if item is None:
            if last is not None and time.monotonic() - last >= TIMEOUT_CAMERA:
                nd.theo_doi.clear()
                last = None
                notice = np.zeros((240, 480, 3), np.uint8)
                cv2.putText(notice, "Mat tin hieu camera", (35, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ket_qua(cv2.imencode(".jpg", notice)[1].tobytes(), {})
            continue
        received, jpg = item
        try:
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("JPEG không đọc được")
            if last is None or received - last > B.GIU_MAT:
                nd.theo_doi.clear()
                fps = 0
            else:
                fps = 0.9 * fps + 0.1 / max(received - last, 0.001)
            last = received
            ts = max(ts + 1, int(received * 1000))
            image, faces = B.xu_ly_khung(frame, xoay, nd, lm, ts, fps)
            visible = [m for m in faces if m["tu_luc"] is not None]
            candidates = [m for m in visible if m["xs"] is not None]
            best = max(candidates, key=lambda m: m["xs"].max(), default=None)
            data = {"name": best["ten"] if best else ("?" if visible else "Khong co khuon mat"),
                    "confidence": float(best["xs"].max()) if best else None, "count": len(visible)}
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                ket_qua(buf.tobytes(), data)
        except Exception:
            nd.theo_doi.clear()
            app.logger.exception("Lỗi xử lý ảnh từ laptop")


# Gioi han kich thuoc anh nhan duoc: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# Trang chu
@app.route("/")
def home():
    return render_template("index.html")


# Nhan anh JPEG tu Windows
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global latest_frame, pending

    image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({"error": "Khong co du lieu anh"}), 400

    if relay:
        with lock:
            latest_frame = image_bytes
    else:
        if not image_bytes.startswith(bytes.fromhex("ffd8ff")):
            return jsonify({"error": "Cần ảnh JPEG"}), 400
        with camera:
            pending = (time.monotonic(), image_bytes)
            camera.notify()

    return jsonify({"status": "success"})


# Nhan du lieu JSON tu AI
@app.route("/receive", methods=["POST"])
def receive_data():
    global latest_data
    if not relay:
        return jsonify({"error": "Orin tự tạo kết quả nhận diện"}), 409

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay", action="store_true", help="Chỉ hiển thị ảnh và JSON từ AI bên ngoài")
    relay = parser.parse_args().relay
    if relay:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import buoc6_nhan_dien as B
        model = B.NhanDien()
        stop = threading.Event()
        with B.tao_bo_tim_mat() as detector:
            worker = threading.Thread(target=nhan_dien, args=(model, detector, stop))
            worker.start()
            try:
                print("Orin đang chờ hình từ laptop (cổng 5000).", flush=True)
                app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
            finally:
                stop.set()
                with camera:
                    camera.notify_all()
                worker.join()

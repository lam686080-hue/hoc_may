"""
Gửi khung hình webcam từ máy Windows sang Flask server trên Jetson Orin (/upload_frame).
Chạy: python camera_sender.py --orin-ip <IP_ORIN>
"""
import argparse
import os
import cv2
import gc
import numpy as np
import requests
import threading
import time

ORIN_IP = "100.69.60.112"


# Neu camera khong nha khi tat, thu doi sang cv2.CAP_DSHOW
BACKEND = cv2.CAP_MSMF

FPS_SEND = 10
JPEG_QUALITY = 50
WIN = "Camera Windows"

latest_frame = None
lock = threading.Lock()
stop_event = threading.Event()
cam_enabled = threading.Event()   # set = bat camera, clear = tat camera
cam_enabled.set()


# ---------- Camera ----------
def open_camera():
    cap = cv2.VideoCapture(0, BACKEND)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def close_camera(cap):
    try:
        cap.release()
    finally:
        del cap
        gc.collect()
        time.sleep(0.3)


def capture_loop():
    """Thread doc camera rieng -> vong lap chinh luon phan hoi duoc phim/thoat."""
    global latest_frame
    cap = None
    try:
        while not stop_event.is_set():
            if cam_enabled.is_set():
                if cap is None:
                    cap = open_camera()
                    if cap is None:
                        print("Khong mo duoc camera, thu lai sau 1s...")
                        time.sleep(1)
                        continue
                    print("Da BAT camera")
                ret, frame = cap.read()
                if not ret:
                    print("Khong doc duoc camera")
                    close_camera(cap)
                    cap = None
                    time.sleep(0.5)
                    continue
                with lock:
                    latest_frame = frame
            else:
                if cap is not None:
                    close_camera(cap)
                    cap = None
                    print("Da TAT camera")
                with lock:
                    latest_frame = None
                time.sleep(0.05)
    finally:
        if cap is not None:
            close_camera(cap)


# ---------- Gui frame ----------
def sender_loop(url):
    session = requests.Session()
    interval = 1.0 / FPS_SEND
    next_t = time.perf_counter()
    last_err_print = 0.0
    last_sent = None

    while not stop_event.is_set():
        now = time.perf_counter()
        if now < next_t:
            time.sleep(min(next_t - now, 0.005))
            continue
        next_t = now + interval

        with lock:
            frame = latest_frame
        if frame is None or frame is last_sent or not cam_enabled.is_set():
            continue

        last_sent = frame
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        try:
            r = session.post(
                url,
                data=buf.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=(1, 2),
            )
            if r.status_code != 200:
                print("Loi HTTP:", r.status_code)
        except requests.RequestException as e:
            if now - last_err_print > 2:
                print("Loi ket noi:", e)
                last_err_print = now
            time.sleep(0.5)

    session.close()


def make_off_screen():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "CAMERA OFF", (170, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(img, "c: bat lai | q / ESC: thoat", (150, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
    return img


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Gửi webcam Windows tới Orin.")
    parser.add_argument("--orin-ip", default=ORIN_IP, help="Địa chỉ IP của Orin")
    args = parser.parse_args()
    url = f"http://{args.orin_ip}:5000/upload_frame"
    print(f"Gửi hình tới {url}")
    off_screen = make_off_screen()

    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_send = threading.Thread(target=sender_loop, args=(url,), daemon=True)
    t_cap.start()
    t_send.start()

    print("Phim: 'c' = bat/tat camera | 'q' hoac ESC = thoat | Ctrl+C = thoat")
    print("Hoac bam nut X cua cua so de thoat.")

    try:
        while not stop_event.is_set():
            with lock:
                frame = latest_frame
            cv2.imshow(WIN, frame if frame is not None else off_screen)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):      # q hoac ESC
                break
            if key in (ord("c"), ord("C")):
                if cam_enabled.is_set():
                    cam_enabled.clear()
                else:
                    cam_enabled.set()

            # Bam nut X de dong cua so
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Dang thoat...")
        stop_event.set()
        cam_enabled.clear()
        t_cap.join(timeout=2)
        t_send.join(timeout=2)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
        # Ep thoat han, ke ca khi driver camera / request bi treo
        os._exit(0)


if __name__ == "__main__":
    main()

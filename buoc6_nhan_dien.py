"""
=====================================================================
 BƯỚC 6 – TRIỂN KHAI: NHẬN DIỆN THỜI GIAN THỰC + ĐIỂM DANH    (Thành viên 3)
=====================================================================
 Làm gì   : Camera điện thoại -> tìm các khuôn mặt -> căn chỉnh + tiền xử lý GIỐNG HỆT
            bước 1 -> mô hình tốt nhất (bước 5) cho xác suất -> hiện tên / "Nguoi la"
            -> nhận ổn định 1,5 giây thì ghi điểm danh. Chạy trên Jetson, xem qua web.
 Đầu vào  : mo_hinh/mo_hinh_tot_nhat.joblib (bước 5), camera IP Webcam (khai báo trong thu_anh.py)
 Đầu ra   : hình nhận diện (cửa sổ hoặc web), file diem_danh.csv
 Các bước : 6.1 Nạp mô hình
            6.2 Kết nối camera (luồng đọc riêng, tự kết nối lại khi mất tín hiệu)
            6.3 MediaPipe tìm tối đa 4 khuôn mặt trong mỗi khung hình
            6.4 Mỗi khuôn mặt: căn chỉnh 5 điểm -> tiền xử lý -> predict_proba
            6.5 Quyết định tên (ngưỡng người lạ, làm mượt), điểm danh
            6.6 Vẽ kết quả, hiển thị ra cửa sổ hoặc trang web
 Chạy     : python buoc6_nhan_dien.py          (laptop, hiện cửa sổ – bấm q để thoát)
            python3 buoc6_nhan_dien.py --web   (Jetson; mở http://<IP Jetson>:5004)
 Cần cùng thư mục: chung.py, thu_anh.py (hàm camera + căn chỉnh mặt), face_landmarker.task
=====================================================================
"""
import os
import sys
import csv
import time
import threading
from datetime import datetime

import chung as C            # nạp đầu tiên (scikit-learn trước OpenCV, tránh lỗi trên Jetson)
import cv2
import numpy as np
import joblib
import mediapipe as mp       # tìm khuôn mặt + 478 điểm mốc
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import thu_anh as T          # dùng lại: camera, xoay hình, góc đầu, căn chỉnh mặt 5 điểm

# ============================ CẤU HÌNH ============================
CAMERA = T.CAMERA                  # địa chỉ IP Webcam (sửa trong thu_anh.py); webcam laptop: 0
NGUONG_NGUOI_LA = None             # None = dùng ngưỡng bước 5 đã chọn; hoặc đặt tay, vd 0.5
SO_MAT_TOI_DA = 4                  # nhận diện tối đa 4 người cùng lúc
LAM_MUOT = 0.75                    # làm mượt xác suất qua các khung hình (0 = không làm mượt)
GIU_MAT = 0.7                      # mất dấu mặt < 0,7 giây vẫn giữ khung + tên (chống nhấp nháy)
GOC_TOI_DA = 50                    # quay mặt quá 50° -> không nhận diện (dữ liệu chỉ tới ~42°)
GIAY_DIEM_DANH = 1.5               # nhận đúng 1 người liên tục bao lâu thì điểm danh
PHUT_GIUA_2_LAN = 10               # 1 người chỉ điểm danh lại sau 10 phút
CONG_WEB = 5000                    # trong container w7_4, cổng 5000 được ánh xạ ra 5004 của Jetson
FILE_DIEM_DANH = os.path.join(C.THU_MUC, "diem_danh.csv")
# ==================================================================


class NhanDien:
    # ---------------------- 6.1 NẠP MÔ HÌNH ----------------------
    def __init__(self):
        if not os.path.exists(C.FILE_MO_HINH_TOT):
            raise SystemExit("Chưa có mo_hinh/mo_hinh_tot_nhat.joblib. Chạy bước 1 -> 5 trước.")
        goi = joblib.load(C.FILE_MO_HINH_TOT)           # mô hình tốt nhất do bước 5 lưu
        self.mo_hinh = goi["mo_hinh"]
        self.lop = list(self.mo_hinh.classes_)          # danh sách người, đúng thứ tự cột xác suất
        # ngưỡng người lạ: dùng ngưỡng bước 5 đã chọn (FAR = FRR), trừ khi đặt tay ở phần cấu hình
        self.nguong = goi.get("nguong_nguoi_la", 0.6)
        if NGUONG_NGUOI_LA is not None:
            self.nguong = NGUONG_NGUOI_LA
        # chỉ người TRONG LỚP (ảnh tự thu / chụp tay) mới hiện tên + điểm danh;
        # giống người LFW (không có trong phòng) -> "Nguoi la"
        ds = goi.get("nguoi_diem_danh") or goi.get("nguoi_tu_thu") or self.lop
        self.nguoi_diem_danh = set(ds)
        print(f"Mô hình: {goi.get('ten_mo_hinh')} | ngưỡng người lạ: {self.nguong}\n"
              f"Nhận diện + điểm danh {len(self.nguoi_diem_danh)} người: "
              f"{', '.join(sorted(self.nguoi_diem_danh))}")
        self.theo_doi = []          # các mặt ở khung trước (vị trí, xác suất đã làm mượt, tên)
        self.da_diem_danh = {}      # ten -> thời điểm điểm danh gần nhất
        self.su_kien = []           # các lần điểm danh gần đây (để hiện lên màn hình)

    # ---------------------- 6.4 DỰ ĐOÁN 1 KHUÔN MẶT ----------------------
    def du_doan(self, raw, info):
        """Căn chỉnh 5 điểm -> ảnh xám 100x100 -> C.tien_xu_ly (GIỐNG HỆT bước 1)
        -> vector 10.000 chiều -> predict_proba: xác suất thuộc từng người."""
        mat = T.can_chinh_mat(raw, info["d5"], C.KICH_THUOC)       # căn chỉnh như lúc thu ảnh
        xam = C.tien_xu_ly(cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY))  # lọc Gaussian + CLAHE
        return self.mo_hinh.predict_proba(C.vector_hoa(xam[None]))[0]

    # ---------------------- 6.5 QUYẾT ĐỊNH TÊN + ĐIỂM DANH ----------------------
    def ghep_theo_doi(self, cac_mat, bay_gio):
        """Theo dõi từng mặt qua các khung hình: làm mượt xác suất, quyết định tên,
        đếm thời gian nhận liên tục 1 người (để điểm danh)."""
        moi, con_lai = [], list(self.theo_doi)
        for m in cac_mat:
            # (a) tìm mặt ở khung trước gần mặt này nhất (cùng 1 người đang đứng đó)
            cu = min(con_lai, key=lambda t: np.hypot(*(t["tam"] - m["tam"])), default=None)
            if cu is not None and np.hypot(*(cu["tam"] - m["tam"])) > m["co"] * 0.6:
                cu = None                                          # quá xa -> coi là người mới
            if cu is not None:
                con_lai.remove(cu)
            # (b) làm mượt: xác suất = 0,75 x xác suất cũ + 0,25 x xác suất khung này
            xs = m["xs"]
            if cu is not None and cu["xs"] is not None:
                xs = cu["xs"] if xs is None else LAM_MUOT * cu["xs"] + (1 - LAM_MUOT) * xs
            # (c) quyết định tên
            if xs is None:
                ten = "?"                                          # quay mặt quá nhiều
            else:
                i = int(np.argmax(xs))                             # người có xác suất cao nhất
                du_tu_tin = xs[i] >= self.nguong
                trong_lop = self.lop[i] in self.nguoi_diem_danh    # người trong lớp, không phải LFW
                ten = self.lop[i] if (du_tu_tin and trong_lop) else "Nguoi la"
            # (d) nhận liên tục cùng 1 tên từ lúc nào (đủ 1,5 giây thì điểm danh)
            tu_luc = cu["tu_luc"] if (cu is not None and cu["ten"] == ten) else bay_gio
            moi.append(dict(m, xs=xs, ten=ten, tu_luc=tu_luc, lan_cuoi=bay_gio))
        # mặt vừa mất dấu chưa quá 0,7 giây thì vẫn giữ (chống nhấp nháy)
        giu = [t for t in con_lai if bay_gio - t["lan_cuoi"] < GIU_MAT]
        self.theo_doi = moi + giu
        return self.theo_doi

    def diem_danh(self, mat, bay_gio):
        ten = mat["ten"]
        if ten in ("?", "Nguoi la") or bay_gio - mat["tu_luc"] < GIAY_DIEM_DANH:
            return
        if bay_gio - self.da_diem_danh.get(ten, -1e9) < PHUT_GIUA_2_LAN * 60:
            return
        self.da_diem_danh[ten] = bay_gio
        luc = datetime.now()
        xs = float(mat["xs"].max())
        moi = not os.path.exists(FILE_DIEM_DANH)
        with open(FILE_DIEM_DANH, "a", newline="", encoding="utf-8-sig") as f:   # ghi thêm 1 dòng
            w = csv.writer(f)
            if moi:
                w.writerow(["thoi_gian", "ten", "xac_suat"])
            w.writerow([luc.strftime("%Y-%m-%d %H:%M:%S"), ten, round(xs, 3)])
        self.su_kien = ([f"{luc:%H:%M:%S}  {ten}"] + self.su_kien)[:6]
        print(f"  ✔ ĐIỂM DANH: {ten}  lúc {luc:%H:%M:%S}  (xác suất {xs:.2f})")
        T.bip(1500, 200)

    def xu_ly(self, raw, kq):
        """Xử lý 1 khung hình: mỗi mặt MediaPipe tìm được -> góc đầu -> dự đoán
        -> tên -> điểm danh."""
        H, W = raw.shape[:2]
        cac_mat = []
        for lms in kq.face_landmarks:
            info = T.trich_thong_tin(lms, None, W, H)    # khung mặt, góc đầu, 5 điểm mốc
            if info is None:
                continue
            x1, y1, x2, y2 = info["khung"]
            qua_nghieng = abs(info["yaw"]) > GOC_TOI_DA or abs(info["pitch"]) > GOC_TOI_DA
            xs = None if qua_nghieng else self.du_doan(raw, info)      # xác suất từng người
            cac_mat.append({"khung": info["khung"], "tam": np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                            "co": max(x2 - x1, y2 - y1), "xs": xs})
        bay_gio = time.time()
        cac_mat = self.ghep_theo_doi(cac_mat, bay_gio)
        for m in cac_mat:
            self.diem_danh(m, bay_gio)
        return cac_mat

    # ---------------------- 6.6 VẼ KẾT QUẢ ----------------------
    def ve(self, raw, cac_mat, fps):
        H, W = raw.shape[:2]
        s = T.HIEN_THI_CAO / H if H > T.HIEN_THI_CAO else 1.0
        anh = cv2.flip(cv2.resize(raw, (int(W * s), int(H * s))), 1)        # lật như soi gương
        Wd = anh.shape[1]
        dong = []
        # đánh số các mặt từ trái sang phải (theo hình đã lật gương)
        for so, m in enumerate(sorted(cac_mat, key=lambda m: -m["tam"][0]), 1):
            x1, y1, x2, y2 = (int(v * s) for v in m["khung"])
            x1, x2 = Wd - x2, Wd - x1                       # tọa độ sau khi lật gương
            ten = m["ten"]
            mau = T.XANH if ten not in ("Nguoi la", "?") else (T.CAM if ten == "?" else T.DO)
            cv2.rectangle(anh, (x1, y1), (x2, y2), mau, 2)
            phan_tram = "" if m["xs"] is None else f" {m['xs'].max() * 100:.0f}%"
            chu = f"{so}. {'Quay mat lai' if ten == '?' else ten}"
            (tw, th), _ = cv2.getTextSize(chu, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x = min(max(2, x1), max(2, Wd - tw - 2))        # chữ luôn nằm gọn trong hình
            y = y1 - 6 if y1 - 6 > th + 2 else min(anh.shape[0] - 4, y2 + th + 6)
            T.ve_chu(anh, chu, x, y, mau, 0.45)
            dong.append((chu + phan_tram, mau))
        bang = np.zeros((anh.shape[0], 300, 3), np.uint8)   # bảng thông tin bên phải
        T.ve_chu(bang, "NHAN DIEN KHUON MAT", 10, 26, T.VANG, 0.6, 2)
        T.ve_chu(bang, f"So nguoi trong hinh: {len(cac_mat)}   (nguong {self.nguong:.2f})",
                 10, 50, T.XAM, 0.42)
        for i, (chu, mau) in enumerate(dong[:4]):
            T.ve_chu(bang, chu, 10, 76 + 22 * i, mau, 0.55)
        y0 = 76 + 22 * 4 + 14
        T.ve_chu(bang, "DIEM DANH GAN DAY:", 10, y0, T.VANG, 0.5)
        for i, sk in enumerate(self.su_kien[:5]):
            T.ve_chu(bang, sk, 10, y0 + 24 + 22 * i, T.XANH, 0.5)
        T.ve_chu(bang, f"FPS {fps:4.1f}", 10, anh.shape[0] - 10, T.XAM, 0.45)
        return np.hstack([anh, bang])


# ---------------------- 6.6 HIỂN THỊ QUA WEB (MJPEG) ----------------------
KHUNG_WEB = {"jpg": None}          # ảnh JPEG mới nhất để gửi lên trình duyệt
KHOA_WEB = threading.Lock()        # khóa: 2 luồng không đọc/ghi ảnh cùng lúc
TRANG_WEB = ("<html><head><meta charset='utf-8'><title>Nhan dien khuon mat</title></head>"
             "<body style='background:#111;color:#eee;font-family:sans-serif;text-align:center'>"
             "<h2>Nhận diện khuôn mặt – Jetson</h2>"
             "<img src='/stream' style='max-height:85vh'></body></html>").encode("utf-8")


def chay_web():
    """Máy chủ web nhỏ: trang chủ chứa thẻ <img src='/stream'>, /stream gửi liên tục ảnh JPEG."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class XuLy(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=khung")
                self.end_headers()
                try:
                    while True:
                        with KHOA_WEB:
                            jpg = KHUNG_WEB["jpg"]
                        if jpg is not None:
                            self.wfile.write(b"--khung\r\nContent-Type: image/jpeg\r\n\r\n"
                                             + jpg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(TRANG_WEB)

    sv = ThreadingHTTPServer(("0.0.0.0", CONG_WEB), XuLy)
    threading.Thread(target=sv.serve_forever, daemon=True).start()
    print(f"Web: mở trình duyệt tới http://<IP máy này>:{CONG_WEB}"
          "  (container Jetson w7_4 -> http://<IP Jetson>:5004)")
    return sv


def main():
    che_do_web = "--web" in sys.argv
    print("=" * 70)
    print(" BƯỚC 6 – NHẬN DIỆN THỜI GIAN THỰC" + (" (chế độ web)" if che_do_web else ""))
    print("=" * 70)
    print("6.1 Nạp mô hình")
    nd = NhanDien()

    print(f"6.2 Kết nối camera {CAMERA} ...")
    cam = T.CameraLuong(CAMERA)             # luồng đọc camera chạy nền, giữ khung mới nhất
    if not cam.cho_khung_dau(15):
        cam.dung()
        raise SystemExit("!! Không nhận được hình từ camera (kiểm tra IP Webcam / Wi-Fi).")
    if isinstance(T.XOAY, str) and not che_do_web:  # tự nhận chiều camera (cần cửa sổ)
        cv2.namedWindow(T.TEN_CUA_SO, cv2.WINDOW_AUTOSIZE)
        print("   Tự nhận chiều camera: nhìn thẳng vào camera...")
        ok, xoay = T.tu_nhan_chieu(cam)
        if ok is None:
            cam.dung()
            return
        xoay = xoay if ok else T.XOAY_DU_PHONG
    else:
        xoay = T.XOAY_DU_PHONG if isinstance(T.XOAY, str) else T.XOAY
    print(f"   Chiều camera: {T.TEN_XOAY.get(xoay, xoay)}")

    sv = chay_web() if che_do_web else None
    print("6.3 Tạo bộ tìm khuôn mặt MediaPipe (chế độ video)")
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=T.tai_model()),
        running_mode=vision.RunningMode.VIDEO, num_faces=SO_MAT_TOI_DA,
        min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6, output_face_blendshapes=False))
    id_cu, t0, ts_truoc, t_truoc, fps = -1, time.perf_counter(), -1, None, 0.0
    hien, t_khung_moi, so_loi = None, time.time(), 0
    print("Đang nhận diện... " + ("Ctrl+C để dừng." if che_do_web else "Bấm q ở cửa sổ để dừng."))
    try:
        while True:
            fid, frame = cam.doc()
            if frame is None or fid == id_cu:                              # chưa có khung hình mới
                if hien is not None and time.time() - t_khung_moi > 2.0:      # > 2 giây: mất camera
                    canh_bao = hien.copy()
                    y = canh_bao.shape[0] // 2
                    w = canh_bao.shape[1] - 301                       # trừ bảng bên phải
                    cv2.rectangle(canh_bao, (0, y - 28), (w, y + 22), (0, 0, 0), -1)
                    T.ve_chu(canh_bao, "MAT KET NOI CAMERA", 8, y - 6, T.DO, 0.5, 2)
                    T.ve_chu(canh_bao, "dang thu lai...", 8, y + 14, T.DO, 0.45)
                    if che_do_web:
                        ok, buf = cv2.imencode(".jpg", canh_bao)
                        with KHOA_WEB:
                            KHUNG_WEB["jpg"] = buf.tobytes() if ok else KHUNG_WEB["jpg"]
                    else:
                        cv2.imshow(T.TEN_CUA_SO, canh_bao)
                if not che_do_web and (cv2.waitKey(5) & 0xFF) in (ord("q"), 27):
                    break
                if che_do_web:
                    time.sleep(0.02)
                continue
            id_cu = fid
            bay_gio = t_khung_moi = time.time()
            if t_truoc:
                fps_moi = 1 / max(bay_gio - t_truoc, 1e-3)
                fps = 0.9 * fps + 0.1 * fps_moi if fps else fps_moi           # làm mượt FPS
            t_truoc = bay_gio

            try:                                    # 1 khung hình lỗi không làm chương trình tắt
                raw = T.chuan_bi_khung(frame, xoay)
                H, W = raw.shape[:2]
                anh_xl = raw                        # thu nhỏ (cao 480 px) cho MediaPipe chạy nhanh
                if H > T.XU_LY_CAO:
                    anh_xl = cv2.resize(raw, (max(1, int(W * T.XU_LY_CAO / H)), T.XU_LY_CAO))
                ts = max(int((time.perf_counter() - t0) * 1000), ts_truoc + 1)
                ts_truoc = ts
                # 6.3 MediaPipe tìm các khuôn mặt (ảnh RGB + thời điểm ts tính bằng ms)
                rgb = cv2.cvtColor(anh_xl, cv2.COLOR_BGR2RGB)
                kq = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                hien = nd.ve(raw, nd.xu_ly(raw, kq), fps)                   # 6.4 -> 6.6
            except Exception as e:
                so_loi += 1
                if so_loi <= 5 or so_loi % 100 == 0:
                    print(f"  ! Lỗi xử lý khung hình (lần {so_loi}): {e} – bỏ qua, chạy tiếp")
                continue

            if che_do_web:
                ok, buf = cv2.imencode(".jpg", hien, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with KHOA_WEB:
                        KHUNG_WEB["jpg"] = buf.tobytes()
            else:
                cv2.imshow(T.TEN_CUA_SO, hien)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.dung()
        lm.close()
        if sv:
            sv.shutdown()
        cv2.destroyAllWindows()
        print(f"Đã dừng. Danh sách điểm danh: {FILE_DIEM_DANH}")


if __name__ == "__main__":
    main()

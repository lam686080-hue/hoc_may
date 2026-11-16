"""
=====================================================================
 BƯỚC 6 – TRIỂN KHAI: NHẬN DIỆN THỜI GIAN THỰC + ĐIỂM DANH
 Phụ trách: Thành viên 3 – Lê Thanh Cao
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


class NhanDien:
    # ---------------------- 6.1 NẠP MÔ HÌNH ----------------------
    def __init__(self):
        if not os.path.exists(C.FILE_MO_HINH_TOT):
            raise SystemExit("Chưa có mo_hinh/mo_hinh_tot_nhat.joblib. Chạy bước 1 -> 5 trước.")
        goi = joblib.load(C.FILE_MO_HINH_TOT)           # mô hình tốt nhất do bước 5 lưu
        self.mo_hinh = goi["mo_hinh"]
        self.lop = list(self.mo_hinh.classes_)          # danh sách người, đúng thứ tự cột xác suất
        # ngưỡng người lạ: dùng ngưỡng bước 5 (FAR = FRR), trừ khi đặt tay ở phần cấu hình
        self.nguong = goi["nguong_nguoi_la"] if NGUONG_NGUOI_LA is None else NGUONG_NGUOI_LA
        # chỉ người TRONG LỚP mới hiện tên + điểm danh; giống người LFW -> "Nguoi la"
        self.nguoi_diem_danh = set(goi["nguoi_diem_danh"])
        print(f"Mô hình: {goi['ten_mo_hinh']} | ngưỡng người lạ: {self.nguong}\n"
              f"Nhận diện + điểm danh: {', '.join(sorted(self.nguoi_diem_danh))}")
        self.theo_doi = []          # các mặt ở khung trước (vị trí, xác suất đã làm mượt, tên)
        self.da_diem_danh = {}      # tên -> thời điểm điểm danh gần nhất
        self.su_kien = []           # các lần điểm danh gần đây (hiện lên màn hình)

    # ---------------------- 6.4 DỰ ĐOÁN 1 KHUÔN MẶT ----------------------
    def du_doan(self, raw, d5):
        """Căn chỉnh 5 điểm -> ảnh xám 100x100 -> C.tien_xu_ly (GIỐNG HỆT bước 1)
        -> vector 10.000 chiều -> predict_proba: xác suất thuộc từng người."""
        mat = T.can_chinh_mat(raw, d5, C.KICH_THUOC)                # căn chỉnh như lúc thu ảnh
        xam = C.tien_xu_ly(cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY))  # lọc Gaussian + CLAHE
        return self.mo_hinh.predict_proba(C.vector_hoa(xam[None]))[0]

    # ---------------------- 6.5 QUYẾT ĐỊNH TÊN + ĐIỂM DANH ----------------------
    def ghep_theo_doi(self, cac_mat, bay_gio):
        """Theo dõi từng mặt qua các khung hình: làm mượt xác suất, quyết định tên,
        đếm thời gian nhận liên tục 1 người (để điểm danh)."""
        moi, con_lai = [], list(self.theo_doi)
        for m in cac_mat:
            # (a) mặt ở khung trước gần mặt này nhất (cùng 1 người đang đứng đó); quá xa -> mới
            cu = min(con_lai, key=lambda t: np.hypot(*(t["tam"] - m["tam"])), default=None)
            if cu is not None and np.hypot(*(cu["tam"] - m["tam"])) > m["co"] * 0.6:
                cu = None
            if cu is not None:
                con_lai.remove(cu)
            # (b) làm mượt: xác suất = 0,75 x xác suất cũ + 0,25 x xác suất khung này
            xs = m["xs"]
            if cu is not None and cu["xs"] is not None:
                xs = cu["xs"] if xs is None else LAM_MUOT * cu["xs"] + (1 - LAM_MUOT) * xs
            # (c) quyết định tên: đủ tự tin VÀ là người trong lớp thì hiện tên
            ten = "?"                                              # "?" = quay mặt quá nhiều
            if xs is not None:
                i = int(np.argmax(xs))                             # người có xác suất cao nhất
                du_tu_tin = xs[i] >= self.nguong
                trong_lop = self.lop[i] in self.nguoi_diem_danh    # người trong lớp, không phải LFW
                ten = self.lop[i] if (du_tu_tin and trong_lop) else "Nguoi la"
            # (d) nhận liên tục cùng 1 tên từ lúc nào (đủ 1,5 giây thì điểm danh)
            tu_luc = cu["tu_luc"] if (cu is not None and cu["ten"] == ten) else bay_gio
            moi.append(dict(m, xs=xs, ten=ten, tu_luc=tu_luc, lan_cuoi=bay_gio))
        # mặt vừa mất dấu chưa quá 0,7 giây thì vẫn giữ (chống nhấp nháy)
        self.theo_doi = moi + [t for t in con_lai if bay_gio - t["lan_cuoi"] < GIU_MAT]
        return self.theo_doi

    def diem_danh(self, mat, bay_gio):
        """Ghi 1 dòng vào diem_danh.csv khi nhận đủ 1,5 giây và chưa điểm danh trong 10 phút."""
        ten = mat["ten"]
        if ten in ("?", "Nguoi la") or bay_gio - mat["tu_luc"] < GIAY_DIEM_DANH \
                or bay_gio - self.da_diem_danh.get(ten, -1e9) < PHUT_GIUA_2_LAN * 60:
            return
        self.da_diem_danh[ten] = bay_gio
        luc, xs = datetime.now(), float(mat["xs"].max())
        moi = not os.path.exists(FILE_DIEM_DANH)
        with open(FILE_DIEM_DANH, "a", newline="", encoding="utf-8-sig") as f:   # ghi thêm 1 dòng
            w = csv.writer(f)
            if moi:
                w.writerow(["thoi_gian", "ten", "xac_suat"])
            w.writerow([luc.strftime("%Y-%m-%d %H:%M:%S"), ten, round(xs, 3)])
        self.su_kien = ([f"{luc:%H:%M:%S}  {ten}"] + self.su_kien)[:5]
        print(f"  ✔ ĐIỂM DANH: {ten}  lúc {luc:%H:%M:%S}  (xác suất {xs:.2f})")
        T.bip(1500, 200)

    def xu_ly(self, raw, kq):
        """1 khung hình: mỗi mặt MediaPipe tìm được -> góc đầu -> dự đoán -> tên -> điểm danh."""
        H, W = raw.shape[:2]
        cac_mat = []
        for lms in kq.face_landmarks:
            info = T.trich_thong_tin(lms, None, W, H)    # khung mặt, góc đầu, 5 điểm mốc
            if info is None:
                continue
            x1, y1, x2, y2 = info["khung"]
            qua_nghieng = abs(info["yaw"]) > GOC_TOI_DA or abs(info["pitch"]) > GOC_TOI_DA
            cac_mat.append({"khung": info["khung"], "tam": np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                            "co": max(x2 - x1, y2 - y1),
                            "xs": None if qua_nghieng else self.du_doan(raw, info["d5"])})
        bay_gio = time.time()
        cac_mat = self.ghep_theo_doi(cac_mat, bay_gio)
        for m in cac_mat:
            self.diem_danh(m, bay_gio)
        return cac_mat

    # ---------------------- 6.6 VẼ KẾT QUẢ ----------------------
    def ve(self, raw, cac_mat, fps):
        """Ảnh lật gương + khung, tên từng mặt; bảng bên phải: danh sách mặt, điểm danh, FPS."""
        H, W = raw.shape[:2]
        s = min(1.0, T.HIEN_THI_CAO / H)
        anh = cv2.flip(cv2.resize(raw, (int(W * s), int(H * s))), 1)        # lật như soi gương
        dong = [("NHAN DIEN KHUON MAT", T.VANG), (f"So nguoi: {len(cac_mat)}  (nguong "
                f"{self.nguong:.2f})", T.XAM)]
        for so, m in enumerate(sorted(cac_mat, key=lambda m: -m["tam"][0]), 1):   # trái -> phải
            x1, y1, x2, y2 = (int(v * s) for v in m["khung"])
            x1, x2 = anh.shape[1] - x2, anh.shape[1] - x1               # tọa độ sau khi lật gương
            mau = {"?": T.CAM, "Nguoi la": T.DO}.get(m["ten"], T.XANH)
            chu = f"{so}. {'Quay mat lai' if m['ten'] == '?' else m['ten']}"
            cv2.rectangle(anh, (x1, y1), (x2, y2), mau, 2)
            T.ve_chu(anh, chu, max(2, x1), max(14, y1 - 6), mau, 0.45)
            dong.append((chu + ("" if m["xs"] is None else f" {m['xs'].max():.0%}"), mau))
        dong += [("DIEM DANH GAN DAY:", T.VANG)] + [(sk, T.XANH) for sk in self.su_kien]
        bang = np.zeros((anh.shape[0], 300, 3), np.uint8)             # bảng thông tin bên phải
        for i, (chu, mau) in enumerate(dong):
            T.ve_chu(bang, chu, 10, 26 + 24 * i, mau, 0.5)
        T.ve_chu(bang, f"FPS {fps:4.1f}", 10, anh.shape[0] - 10, T.XAM, 0.45)
        return np.hstack([anh, bang])


# ---------------------- 6.6 HIỂN THỊ QUA WEB (MJPEG) ----------------------
KHUNG_WEB = {"jpg": None}          # ảnh JPEG mới nhất để gửi lên trình duyệt


class XuLyWeb(BaseHTTPRequestHandler):
    """Trang chủ chứa thẻ <img src='/stream'>; /stream gửi liên tục ảnh JPEG (MJPEG)."""
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        if not self.path.startswith("/stream"):
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<html><body style='background:#111;text-align:center'>"
                             "<img src='/stream' style='max-height:90vh'></body></html>".encode())
            return
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=khung")
        self.end_headers()
        try:
            while True:
                jpg = KHUNG_WEB["jpg"]
                if jpg is not None:
                    self.wfile.write(b"--khung\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass                                          # trình duyệt đã đóng trang


def hien_thi(anh, che_do_web):
    """Đưa 1 ảnh ra web (nén JPEG) hoặc ra cửa sổ. Trả về True nếu người dùng bấm q để thoát."""
    if che_do_web:
        ok, buf = cv2.imencode(".jpg", anh, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            KHUNG_WEB["jpg"] = buf.tobytes()
        return False
    cv2.imshow(T.TEN_CUA_SO, anh)
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)


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
    xoay = T.XOAY_DU_PHONG if isinstance(T.XOAY, str) else T.XOAY
    if isinstance(T.XOAY, str) and not che_do_web:    # tự nhận chiều camera (cần cửa sổ)
        cv2.namedWindow(T.TEN_CUA_SO, cv2.WINDOW_AUTOSIZE)
        print("   Tự nhận chiều camera: nhìn thẳng vào camera...")
        ok, x = T.tu_nhan_chieu(cam)
        if ok is None:
            cam.dung()
            return
        xoay = x if ok else xoay
    print(f"   Chiều camera: {T.TEN_XOAY.get(xoay, xoay)}")
    sv = None
    if che_do_web:
        sv = ThreadingHTTPServer(("0.0.0.0", CONG_WEB), XuLyWeb)     # máy chủ web chạy nền
        threading.Thread(target=sv.serve_forever, daemon=True).start()
        print(f"Web: http://<IP máy này>:{CONG_WEB}  (Jetson w7_4: http://<IP Jetson>:5004)")

    print("6.3 Tạo bộ tìm khuôn mặt MediaPipe (chế độ video)")
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=T.tai_model()),
        running_mode=vision.RunningMode.VIDEO, num_faces=SO_MAT_TOI_DA,
        min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6))
    id_cu, t0, ts, fps, hien, t_moi, so_loi = -1, time.perf_counter(), 0, 0.0, None, time.time(), 0
    print("Đang nhận diện... " + ("Ctrl+C để dừng." if che_do_web else "Bấm q ở cửa sổ để dừng."))
    try:
        while True:
            fid, frame = cam.doc()
            if frame is None or fid == id_cu:                       # chưa có khung hình mới
                if hien is not None and time.time() - t_moi > 2.0:   # > 2 giây: mất camera
                    canh_bao = hien.copy()
                    T.ve_chu(canh_bao, "MAT KET NOI CAMERA - dang thu lai...", 8,
                             canh_bao.shape[0] // 2, T.DO, 0.5, 2)
                    if hien_thi(canh_bao, che_do_web):
                        break
                elif not che_do_web and (cv2.waitKey(5) & 0xFF) in (ord("q"), 27):
                    break                                            # cửa sổ: bấm q để thoát
                time.sleep(0.02)
                continue
            id_cu = fid
            bay_gio = time.time()
            fps = 0.9 * fps + 0.1 / max(bay_gio - t_moi, 1e-3)       # số khung/giây (làm mượt)
            t_moi = bay_gio
            try:                                    # 1 khung hình lỗi không làm chương trình tắt
                raw = T.chuan_bi_khung(frame, xoay)
                H, W = raw.shape[:2]
                s = min(1.0, T.XU_LY_CAO / H)       # thu nhỏ (cao 480 px) cho MediaPipe chạy nhanh
                rgb = cv2.cvtColor(cv2.resize(raw, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB)
                ts = max(int((time.perf_counter() - t0) * 1000), ts + 1)   # thời điểm (ms) tăng
                kq = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                hien = nd.ve(raw, nd.xu_ly(raw, kq), fps)               # 6.4 -> 6.6
            except Exception as e:
                so_loi += 1
                if so_loi <= 5:                     # chỉ in vài lỗi đầu cho đỡ rối màn hình
                    print(f"  ! Lỗi xử lý khung hình: {e} – bỏ qua, chạy tiếp")
                continue
            if hien_thi(hien, che_do_web):
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

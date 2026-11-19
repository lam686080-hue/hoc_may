"""
=====================================================================
 THU THẬP DỮ LIỆU KHUÔN MẶT – BÀI TẬP LỚN NHẬN DIỆN KHUÔN MẶT
=====================================================================
Cần MediaPipe, NumPy và bản OpenCV có giao diện.
Chạy:              python thu_anh.py

KỊCH BẢN MỖI NGƯỜI (giống hệt nhau cho mọi người -> dữ liệu cân bằng)
  1 LƯỢT x 5 TƯ THẾ = 30 ảnh
    1. Nhìn thẳng      10 ảnh
    2. Quay TRÁI nhẹ    5 ảnh  (7–18°)
    3. Quay PHẢI nhẹ    5 ảnh  (7–18°)
    4. Ngẩng LÊN nhẹ    5 ảnh  (5–12°)
    5. Cúi XUỐNG nhẹ    5 ảnh  (5–12°)
  Trước khi thu: nhìn thẳng vào camera và giữ yên để lấy mốc.

CHỈ CHỤP KHI ĐẠT TẤT CẢ:
  - đúng 1 khuôn mặt, nằm trọn trong khung, đủ lớn
  - đúng tư thế (đo góc đầu bằng độ, so với mốc), giữ liên tục 3 khung hình
  - mắt mở, đủ sáng, đủ tương phản, ảnh nét
  - khác các ảnh vừa chụp (đứng yên -> không chụp)

DỮ LIỆU LƯU RA (thư mục "dataset" cạnh file này)
  dataset/anh_goc/<ten>/*.jpg      khung hình gốc đầy đủ (để sau này xử lý lại tùy ý)
  dataset/mat_mau_112/<ten>/*.png  mặt màu đã căn chỉnh 112x112 (chuẩn ArcFace)
  dataset/mat_xam_100/<ten>/*.png  mặt xám đã căn chỉnh 100x100 (dùng cho PCA và hồi quy logistic)
  dataset/metadata.csv             thông tin từng ảnh: người, lượt, tư thế, góc đầu,
                                   điểm cười, độ nét, độ sáng, 5 điểm mốc...
  dataset/xem_nhanh/<ten>.jpg      lưới toàn bộ ảnh mặt của 1 người để soát nhanh
  dataset/cau_hinh_thu_thap.json   cấu hình + thông tin các phiên thu (ghi vào báo cáo)
  Tên file: <ten>_<so>_L<luot>_<tu_the>  (vd: Nguyen_Van_Lam_0007_L1_trai)

Bị ngắt giữa chừng? Chạy lại, nhập ĐÚNG tên cũ, gõ c -> chụp tiếp phần còn thiếu.

SAU KHI THU: chạy  python kiem_tra.py dataset  để kiểm tra. Nếu có ảnh lỗi, script kiểm
tra sẽ chuyển ảnh lỗi sang dataset/loai_bo -> chạy lại thu_anh.py, nhập đúng tên, gõ c:
chương trình tự chụp lại ĐÚNG những ảnh còn thiếu (đúng lượt, tư thế, khoảng góc).

PHÍM TẮT (bấm khi đang chọn cửa sổ camera):  q thoát | p tạm dừng | m lấy lại mốc
=====================================================================
"""
import os
import csv
import json
import math
import time
import queue
from pathlib import Path
import threading
import unicodedata
import urllib.request
from datetime import datetime

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    raise SystemExit("Chưa cài MediaPipe. Chạy lệnh:  pip install mediapipe")

try:
    import winsound  # chỉ có trên Windows
except ImportError:
    winsound = None


# =====================================================================
# 1. CẤU HÌNH
# =====================================================================
# ---------- Camera ----------
CAMERA = 0   # webcam đầu tiên trên máy đang chạy (Orin hoặc laptop); với IP Webcam, dùng URL http://IP:8080/video
# Chiều xoay hình: "tu_dong" = lúc khởi động tự thử 4 chiều, chọn chiều có khuôn mặt ĐỨNG THẲNG.
# Muốn cố định thì đặt: cv2.ROTATE_90_CLOCKWISE / cv2.ROTATE_90_COUNTERCLOCKWISE / cv2.ROTATE_180 / None
XOAY = None   # giữ nguyên chiều webcam giữa các lần chạy
XOAY_DU_PHONG = cv2.ROTATE_90_COUNTERCLOCKWISE   # dùng nếu tự nhận không thành công
CAMERA_DA_LAT_GUONG = False   # True nếu app đã lật gương sẵn (xem phần kiểm tra khi chạy)
HIEN_THI_CAO = 360            # chiều cao cửa sổ camera (tối thiểu 360)
XU_LY_CAO = 480               # chiều cao ảnh đưa vào MediaPipe (nhỏ hơn cho nhanh)

# ---------- Kịch bản thu ----------
LUOT = [
    "Ngồi ngay ngắn, đặt camera ngang tầm mắt và giữ ánh sáng đủ để nhìn rõ mặt.",
]
# (mã, chữ trên cửa sổ, lời nhắc ở terminal, số ảnh, khoảng góc hoặc None)
TU_THE = [
    ("thang", "Nhin THANG vao camera", "Nhìn thẳng, dịch nhẹ đầu giữa các ảnh", 10, None),
    ("trai", "Quay nhe sang TRAI", "Quay nhẹ sang trái (7–18°)", 5, [(7, 18)]),
    ("phai", "Quay nhe sang PHAI", "Quay nhẹ sang phải (7–18°)", 5, [(7, 18)]),
    ("len", "Ngang nhe len", "Ngẩng nhẹ (5–12°)", 5, [(5, 12)]),
    ("xuong", "Cui nhe xuong", "Cúi nhẹ (5–12°)", 5, [(5, 12)]),
]

# ---------- Ngưỡng tư thế (đơn vị: độ, so với mốc nhìn thẳng) ----------
THANG_TOI_DA = 7              # nhìn thẳng: |yaw|, |pitch| tối đa
LECH_PHU_QUAY = 12            # quay trái/phải: |pitch| tối đa
LECH_PHU_NGANG_CUI = 10       # ngẩng/cúi: |yaw| tối đa
BIEU_CAM_LECH_TOI_DA = 12     # cười / nói: |yaw|, |pitch| tối đa
SAI_HUONG = 3                 # lệch ngược hướng quá 3° -> báo sai hướng

# ---------- Ngưỡng biểu cảm (điểm 0..1 của MediaPipe) ----------
NGUONG_CUOI = 0.50            # bước cười: tối thiểu
CUOI_TOI_DA = 0.40            # các bước khác: không cười quá mức này
NGUONG_HA_MIENG = 0.25        # bước nói: độ há miệng tối thiểu
NGUONG_NHAM_MAT = 0.50        # nhắm mắt quá mức này -> không chụp
NGUONG_NHAM_MAT_KHI_CUI = 0.85  # khi cúi, mí mắt tự hạ xuống nên nới ngưỡng

# ---------- Chất lượng ảnh ----------
MAT_TOI_THIEU_PX = 150        # chiều cao mặt tối thiểu (pixel, trên ảnh gốc)
MAT_TOI_DA_TI_LE = 0.85       # chiều cao mặt tối đa so với khung hình
LE_KHUNG_TI_LE = 0.02         # mặt phải cách mép khung tối thiểu 2%
DO_SANG_MIN, DO_SANG_MAX = 55, 200
TUONG_PHAN_MIN = 20
NGUONG_NET = 25               # độ nét (phương sai Laplacian) tối thiểu
NGUONG_KHAC = 3.0             # độ khác tối thiểu so với các ảnh vừa chụp
SO_ANH_SO_SANH = 5            # so với 5 ảnh gần nhất của bước
GIU_KHUNG = 3                 # đúng tư thế liên tục 3 khung hình mới chụp
KHOANG_CACH = 0.25            # giây tối thiểu giữa 2 lần chụp

# ---------- Lấy mốc ----------
SO_KHUNG_MOC = 15             # số khung hình liên tiếp để lấy mốc
MOC_DO_LECH_TOI_DA = 2.5      # độ dao động tối đa (độ) khi lấy mốc
MOC_GOC_TUYET_DOI = 30        # góc tuyệt đối tối đa khi lấy mốc (đặt điện thoại ngang tầm mắt)

CHUYEN_TU_THE = 2.0           # giây chuẩn bị trước mỗi bước
AM_THANH = True               # tiếng bíp khi xong bước / xong lượt (Windows)

# ---------- Lưu trữ ----------
THU_MUC_GOC = os.path.dirname(os.path.abspath(__file__))
THU_MUC_DATA = os.path.join(THU_MUC_GOC, "dataset")
LUU_ANH_GOC = True
CHAT_LUONG_JPG = 95
MODEL = os.path.join(THU_MUC_GOC, "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
# =====================================================================

THU_MUC_ANH_GOC = os.path.join(THU_MUC_DATA, "anh_goc")
THU_MUC_MAU = os.path.join(THU_MUC_DATA, "mat_mau_112")
THU_MUC_XAM = os.path.join(THU_MUC_DATA, "mat_xam_100")
THU_MUC_XEM = os.path.join(THU_MUC_DATA, "xem_nhanh")
FILE_CSV = os.path.join(THU_MUC_DATA, "metadata.csv")
FILE_CAU_HINH = os.path.join(THU_MUC_DATA, "cau_hinh_thu_thap.json")

KE_HOACH = [(l, b) for l in range(len(LUOT)) for b in range(len(TU_THE))]
TONG_MOI_NGUOI = len(LUOT) * sum(t[3] for t in TU_THE)
CHI_SO_MA = {t[0]: i for i, t in enumerate(TU_THE)}
TEN_CUA_SO = "Thu du lieu khuon mat  (q: thoat | p: tam dung | m: lay lai moc)"
RONG_BANG = 440

VANG, DO, XANH, CAM, XAM, TRANG = (0, 255, 255), (0, 0, 255), (0, 255, 0), (0, 165, 255), (190, 190, 190), (255, 255, 255)

COT_CSV = ["file", "so", "ten", "luot", "tu_the", "khoang_goc", "phien", "thoi_gian",
           "yaw", "pitch", "roll", "yaw_tuong_doi", "pitch_tuong_doi",
           "cuoi", "ha_mieng", "nham_mat", "do_net", "do_sang", "tuong_phan", "do_khac",
           "rong_anh", "cao_anh", "x1", "y1", "x2", "y2",
           "mat1_x", "mat1_y", "mat2_x", "mat2_y", "mui_x", "mui_y",
           "mieng1_x", "mieng1_y", "mieng2_x", "mieng2_y"]


# =====================================================================
# 2. HÀM XỬ LÝ KHUÔN MẶT (dùng lại được ở bước nhận diện/triển khai:
#    from thu_anh import tao_landmarker, trich_thong_tin, can_chinh_mat)
# =====================================================================
def tai_model():
    """Tải model Face Landmarker (lần đầu) và trả về nội dung dạng bytes."""
    if not os.path.exists(MODEL):
        print("Đang tải model face_landmarker.task (~4 MB, chỉ lần đầu)...")
        tam = MODEL + ".tmp"
        try:
            urllib.request.urlretrieve(MODEL_URL, tam)
            os.replace(tam, MODEL)
        except Exception as e:
            raise SystemExit(f"Không tải được model ({e}).\nTải thủ công tại:\n  {MODEL_URL}\n"
                             f"rồi đặt vào:\n  {MODEL}")
    with open(MODEL, "rb") as f:   # đọc bytes để tránh lỗi đường dẫn có dấu tiếng Việt
        return f.read()


def tao_landmarker(che_do="video"):
    """che_do='video' cho camera trực tiếp, 'image' cho xử lý ảnh tĩnh."""
    mode = vision.RunningMode.VIDEO if che_do == "video" else vision.RunningMode.IMAGE
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=tai_model()),
        running_mode=mode,
        num_faces=2,                       # để phát hiện người thứ 2 lọt vào khung
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_face_blendshapes=True,      # điểm cười, há miệng, nhắm mắt
    ))


# Mô hình 3D trung bình của khuôn mặt (hệ trục giống camera: x phải, y xuống, z ra xa camera)
MO_HINH_3D = np.array([
    (0.0, 0.0, 0.0),          # 1   đầu mũi
    (0.0, 330.0, 65.0),       # 152 cằm
    (-225.0, -170.0, 135.0),  # 33  khóe ngoài mắt (bên trái ảnh)
    (225.0, -170.0, 135.0),   # 263 khóe ngoài mắt (bên phải ảnh)
    (-150.0, 150.0, 125.0),   # 61  khóe miệng (bên trái ảnh)
    (150.0, 150.0, 125.0),    # 291 khóe miệng (bên phải ảnh)
], dtype=np.float64)
CHI_SO_PNP = [1, 152, 33, 263, 61, 291]

# 5 điểm mẫu chuẩn ArcFace cho ảnh 112x112: mắt trái ảnh, mắt phải ảnh, mũi, 2 khóe miệng
MAU_ARCFACE_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def goc_dau(diem, W, H):
    """Ước lượng góc đầu (độ) bằng solvePnP.
    yaw > 0: quay sang TRÁI của người | pitch > 0: ngẩng LÊN | roll: nghiêng đầu.
    (áp dụng cho ảnh KHÔNG lật gương – ảnh chụp bình thường của camera)"""
    f = float(max(W, H))
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float64)
    p2 = diem[CHI_SO_PNP].astype(np.float64)
    kc_mat = np.linalg.norm(p2[3] - p2[2])
    z0 = f * 450.0 / max(kc_mat, 1.0)
    t0 = np.array([[(p2[0, 0] - W / 2.0) * z0 / f], [(p2[0, 1] - H / 2.0) * z0 / f], [z0]])
    ok, rvec, _ = cv2.solvePnP(MO_HINH_3D, p2, K, np.zeros(4), np.zeros((3, 1)), t0,
                               True, cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    huong = -R[:, 2]                       # hướng mặt (từ mặt nhìn ra camera)
    if huong[2] > 0:                       # nghiệm vô lý (mặt quay ngược) -> bỏ
        return None
    yaw = math.degrees(math.atan2(huong[0], -huong[2]))
    pitch = math.degrees(math.atan2(-huong[1], math.hypot(huong[0], huong[2])))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def diem_5(diem):
    """5 điểm mốc theo thứ tự mẫu ArcFace (tọa độ pixel)."""
    return np.array([(diem[33] + diem[133]) / 2,     # tâm mắt bên trái ảnh
                     (diem[362] + diem[263]) / 2,    # tâm mắt bên phải ảnh
                     diem[1],                         # đầu mũi
                     diem[61], diem[291]],            # 2 khóe miệng
                    dtype=np.float64)


def bien_doi_tuong_dong(src, dst):
    """Phép biến đổi đồng dạng (xoay + co giãn + tịnh tiến) bình phương tối thiểu (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    var_s = (s_c ** 2).sum() / len(src)
    U, S, Vt = np.linalg.svd(d_c.T @ s_c / len(src))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / var_s
    t = mu_d - s * R @ mu_s
    return np.hstack([s * R, t.reshape(2, 1)])


def can_chinh_mat(anh, d5, kich_thuoc=112):
    """Cắt + căn chỉnh mặt theo mẫu ArcFace: mắt nằm ngang, cùng vị trí, cùng kích thước."""
    M = bien_doi_tuong_dong(np.asarray(d5, np.float64), MAU_ARCFACE_112 * (kich_thuoc / 112.0))
    return cv2.warpAffine(anh, M, (kich_thuoc, kich_thuoc),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def trich_thong_tin(lms, blendshapes, W, H):
    """Từ kết quả MediaPipe (1 khuôn mặt) -> dict thông tin dùng để kiểm tra & lưu."""
    diem = np.array([(p.x * W, p.y * H) for p in lms], dtype=np.float64)
    return thong_tin_tu_diem(diem, blendshapes, W, H)


def thong_tin_tu_diem(diem, blendshapes, W, H):
    """Tính khung mặt, góc đầu và các điểm căn chỉnh từ tọa độ pixel."""
    goc = goc_dau(diem, W, H)
    if goc is None:
        return None
    bs = {c.category_name: c.score for c in blendshapes} if blendshapes else {}
    x1, y1 = diem.min(0)
    x2, y2 = diem.max(0)
    return {
        "diem": diem, "khung": (x1, y1, x2, y2), "d5": diem_5(diem),
        "yaw": goc[0], "pitch": goc[1], "roll": goc[2],
        "cuoi": (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2,
        "ha_mieng": bs.get("jawOpen", 0.0),
        "nham": max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)),
    }


def chat_luong(xam):
    """(độ nét, độ sáng, độ tương phản) của ảnh mặt xám đã căn chỉnh."""
    return float(cv2.Laplacian(xam, cv2.CV_64F).var()), float(xam.mean()), float(xam.std())


# =====================================================================
# 3. TIỆN ÍCH
# =====================================================================
def chuan_hoa_ten(s):
    """'Nguyễn Văn Lâm' -> 'Nguyen_Van_Lam'."""
    s = s.strip().replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "_".join(s.split())
    return "".join(c for c in s if c.isascii() and (c.isalnum() or c in "_-"))


def luu_anh(duong_dan, anh, duoi=".png", tham_so=None):
    """Lưu ảnh, chạy được cả với đường dẫn có dấu tiếng Việt."""
    os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
    ok, buf = cv2.imencode(duoi, anh, tham_so or [])
    if ok:
        buf.tofile(duong_dan)
    return ok


def doc_anh(duong_dan, co_mau=True):
    data = np.fromfile(duong_dan, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR if co_mau else cv2.IMREAD_GRAYSCALE)


def bip(tan_so=1200, ms=150):
    if AM_THANH and winsound is not None:
        threading.Thread(target=winsound.Beep, args=(tan_so, ms), daemon=True).start()


def ve_chu(img, chu, x, y, mau=TRANG, co=0.5, day=1):
    cv2.putText(img, chu, (x, y), cv2.FONT_HERSHEY_SIMPLEX, co, (0, 0, 0), day + 3, cv2.LINE_AA)
    cv2.putText(img, chu, (x, y), cv2.FONT_HERSHEY_SIMPLEX, co, mau, day, cv2.LINE_AA)


def doc_csv():
    if not os.path.exists(FILE_CSV):
        return []
    with open(FILE_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ghi_dong_csv(dong):
    moi = not os.path.exists(FILE_CSV)
    if not moi:                     # dòng cuối bị cắt ngang (tắt máy đột ngột) -> xuống dòng trước khi ghi
        with open(FILE_CSV, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                thieu_xuong_dong = f.read(1) not in (b"\n", b"\r")
            else:
                thieu_xuong_dong = False
        if thieu_xuong_dong:
            with open(FILE_CSV, "ab") as f:
                f.write(b"\r\n")
    with open(FILE_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COT_CSV)
        if moi:
            w.writeheader()
        w.writerow(dong)


def chi_tieu_khoang(so_anh, so_khoang):
    """Chia đều số ảnh cho các khoảng góc, phần dư dồn cho các khoảng đầu."""
    return [so_anh // so_khoang + (1 if i < so_anh % so_khoang else 0) for i in range(so_khoang)]


def chuan_bi_khung(frame, xoay):
    """Xoay + bỏ lật gương -> ảnh chuẩn như camera chụp bình thường."""
    if xoay is not None:
        frame = cv2.rotate(frame, xoay)
    if CAMERA_DA_LAT_GUONG:
        frame = cv2.flip(frame, 1)
    return frame


TEN_XOAY = {None: "không xoay (None)", cv2.ROTATE_90_CLOCKWISE: "xoay phải 90° (cv2.ROTATE_90_CLOCKWISE)",
            cv2.ROTATE_90_COUNTERCLOCKWISE: "xoay trái 90° (cv2.ROTATE_90_COUNTERCLOCKWISE)",
            cv2.ROTATE_180: "xoay 180° (cv2.ROTATE_180)"}


def tu_nhan_chieu(cam, giay=30):
    """Thử 4 chiều xoay trên khung hình camera, chọn chiều mà khuôn mặt ĐỨNG THẲNG (|roll| < 35°).
    Cần 3 khung hình liên tiếp cùng kết quả.
    Trả về (True, chiều) nếu nhận được | (False, None) nếu hết giờ | (None, None) nếu bấm q."""
    lm = tao_landmarker("image")
    ung_vien, lien_tiep, id_cu, t0 = None, 0, -1, time.time()
    try:
        while time.time() - t0 < giay:
            fid, frame = cam.doc()
            if frame is None or fid == id_cu:
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    return None, None
                continue
            id_cu = fid
            tim_thay, xoay_dung = False, None
            for xoay in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
                anh = chuan_bi_khung(frame, xoay)
                H, W = anh.shape[:2]
                tl = min(1.0, XU_LY_CAO / max(H, W))
                nho = cv2.resize(anh, (max(1, int(W * tl)), max(1, int(H * tl))), interpolation=cv2.INTER_AREA)
                kq = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(nho, cv2.COLOR_BGR2RGB)))
                if len(kq.face_landmarks) == 1:
                    info = trich_thong_tin(kq.face_landmarks[0], None, W, H)
                    if info is not None and abs(info["roll"]) < 35:
                        tim_thay, xoay_dung = True, xoay
                        break
            if tim_thay and lien_tiep > 0 and xoay_dung == ung_vien:
                lien_tiep += 1
            else:
                ung_vien, lien_tiep = xoay_dung, (1 if tim_thay else 0)
            # hiển thị để người dùng biết đang làm gì
            hien = chuan_bi_khung(frame, ung_vien if lien_tiep else None)
            s = HIEN_THI_CAO / hien.shape[0]
            hien = cv2.flip(cv2.resize(hien, (max(1, int(hien.shape[1] * s)), HIEN_THI_CAO)), 1)
            ve_chu(hien, "Tu nhan chieu camera...", 8, 24, VANG, 0.55)
            ve_chu(hien, "Nhin THANG vao camera", 8, 48, VANG, 0.55)
            cv2.imshow(TEN_CUA_SO, hien)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return None, None
            if lien_tiep >= 3:
                return True, ung_vien
    finally:
        lm.close()
    return False, None


def cau_hinh_thu():
    return {
        "luot": LUOT, "tu_the": [list(t[:4]) + [t[4]] for t in TU_THE],
        "tong_anh_moi_nguoi": TONG_MOI_NGUOI,
        "nguong": {
            "THANG_TOI_DA": THANG_TOI_DA, "LECH_PHU_QUAY": LECH_PHU_QUAY,
            "LECH_PHU_NGANG_CUI": LECH_PHU_NGANG_CUI, "BIEU_CAM_LECH_TOI_DA": BIEU_CAM_LECH_TOI_DA,
            "SAI_HUONG": SAI_HUONG, "NGUONG_CUOI": NGUONG_CUOI, "CUOI_TOI_DA": CUOI_TOI_DA,
            "NGUONG_HA_MIENG": NGUONG_HA_MIENG, "NGUONG_NHAM_MAT": NGUONG_NHAM_MAT,
            "NGUONG_NHAM_MAT_KHI_CUI": NGUONG_NHAM_MAT_KHI_CUI, "MAT_TOI_THIEU_PX": MAT_TOI_THIEU_PX,
            "MAT_TOI_DA_TI_LE": MAT_TOI_DA_TI_LE, "LE_KHUNG_TI_LE": LE_KHUNG_TI_LE,
            "DO_SANG_MIN": DO_SANG_MIN, "DO_SANG_MAX": DO_SANG_MAX, "TUONG_PHAN_MIN": TUONG_PHAN_MIN,
            "NGUONG_NET": NGUONG_NET, "NGUONG_KHAC": NGUONG_KHAC, "GIU_KHUNG": GIU_KHUNG,
            "XU_LY_CAO": XU_LY_CAO, "LUU_ANH_GOC": LUU_ANH_GOC,
        },
        "can_chinh": "Umeyama 5 diem -> mau ArcFace 112x112; ban xam 100x100",
        "phat_hien": "MediaPipe Face Landmarker (478 diem + blendshapes)",
    }


def ghi_cau_hinh(phien, cao, rong, xoay):
    """Ghi cấu hình thu vào JSON. Script kiểm tra đọc file này để dùng ĐÚNG ngưỡng lúc thu."""
    du_lieu = {}
    if os.path.exists(FILE_CAU_HINH):
        try:
            with open(FILE_CAU_HINH, encoding="utf-8") as f:
                du_lieu = json.load(f)
        except Exception:
            du_lieu = {}
    du_lieu["cau_hinh"] = cau_hinh_thu()
    du_lieu.setdefault("phien", []).append({
        "phien": phien, "bat_dau": datetime.now().isoformat(timespec="seconds"),
        "camera": CAMERA, "do_phan_giai": f"{rong}x{cao}", "xoay": TEN_XOAY.get(xoay, str(xoay)),
        "camera_da_lat_guong": CAMERA_DA_LAT_GUONG})
    os.makedirs(THU_MUC_DATA, exist_ok=True)
    with open(FILE_CAU_HINH, "w", encoding="utf-8") as f:
        json.dump(du_lieu, f, ensure_ascii=False, indent=2)


def tao_anh_xem_nhanh(ten, so_cot=16):
    thu_muc = os.path.join(THU_MUC_MAU, ten)
    if not os.path.isdir(thu_muc):
        return None
    files = sorted(f for f in os.listdir(thu_muc) if f.endswith(".png"))
    if not files:
        return None
    o = 116
    so_hang = math.ceil(len(files) / so_cot)
    luoi = np.full((so_hang * o + 4, so_cot * o + 4, 3), 40, np.uint8)
    for i, fn in enumerate(files):
        anh = doc_anh(os.path.join(thu_muc, fn))
        if anh is None:
            continue
        r, c = divmod(i, so_cot)
        luoi[4 + r * o:4 + r * o + 112, 4 + c * o:4 + c * o + 112] = cv2.resize(anh, (112, 112))
    duong_dan = os.path.join(THU_MUC_XEM, f"{ten}.jpg")
    luu_anh(duong_dan, luoi, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 92])
    return duong_dan


def in_tong_ket():
    dong = [d for d in doc_csv() if d.get("ten") and d.get("tu_the")]   # bỏ dòng hỏng
    print("\n" + "=" * 64 + "\nTỔNG KẾT DỮ LIỆU (dataset/metadata.csv)\n" + "=" * 64)
    if not dong:
        print("  Chưa có ảnh nào.")
        return
    ten_ds = sorted({d["ten"] for d in dong})
    tieu_de = "Người".ljust(22) + "".join(m[0].rjust(7) for m in TU_THE) + "   Tổng"
    print(tieu_de)
    for ten in ten_ds:
        cua = [d for d in dong if d["ten"] == ten]
        dem = [sum(1 for d in cua if d["tu_the"] == t[0]) for t in TU_THE]
        trang_thai = "" if len(cua) >= TONG_MOI_NGUOI else "  <- CHƯA ĐỦ"
        print(ten[:21].ljust(22) + "".join(str(c).rjust(7) for c in dem)
              + f"{len(cua):7d}/{TONG_MOI_NGUOI}{trang_thai}")
    print(f"Tổng: {len(dong)} ảnh, {len(ten_ds)} người.")


class CameraLuong:
    """Đọc camera trong luồng riêng, luôn giữ khung hình MỚI NHẤT (tránh trễ hình
    do bộ đệm của luồng video qua mạng). Tự kết nối lại khi mất kết nối."""

    def __init__(self, nguon):
        self.nguon = nguon
        self.khoa = threading.Lock()
        self.khung, self.id, self.ket_noi, self.chay = None, 0, False, True
        self.luong = threading.Thread(target=self._chay, daemon=True)
        self.luong.start()

    def _mo(self):
        if isinstance(self.nguon, int):                     # webcam laptop: 0, 1, ...
            return cv2.VideoCapture(self.nguon)
        # camera qua mạng: đặt thời gian chờ để mất Wi-Fi thì tự kết nối lại, không bị treo
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC") and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            try:
                cap = cv2.VideoCapture(self.nguon, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception:
                pass
        return cv2.VideoCapture(self.nguon)

    def _chay(self):
        cap = None
        while self.chay:
            if cap is None:
                cap = self._mo()
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    self.ket_noi = False
                    time.sleep(1.0)
                    continue
            ok, f = cap.read()
            if not ok or f is None:
                self.ket_noi = False
                cap.release()
                cap = None
                time.sleep(0.5)
                continue
            with self.khoa:
                self.khung, self.id, self.ket_noi = f, self.id + 1, True
        if cap is not None:
            cap.release()

    def doc(self):
        with self.khoa:
            return self.id, self.khung

    def cho_khung_dau(self, giay=15):
        t = time.time()
        while time.time() - t < giay:
            if self.doc()[1] is not None:
                return True
            time.sleep(0.1)
        return False

    def dung(self):
        self.chay = False
        # Để luồng đọc tự release camera trước khi Python thoát (IP camera có timeout 5 giây).
        self.luong.join(timeout=6)


class HoiDap:
    """Hỏi ở terminal bằng luồng riêng -> cửa sổ camera không bị đứng hình."""

    def __init__(self):
        self.yeu_cau, self.tra_loi, self.dang_hoi = queue.Queue(), queue.Queue(), False
        threading.Thread(target=self._chay, daemon=True).start()

    def _chay(self):
        while True:
            loai, loi_nhac = self.yeu_cau.get()
            try:
                s = input(loi_nhac)
            except EOFError:
                s = ""
            self.tra_loi.put((loai, s))

    def hoi(self, loai, loi_nhac):
        if not self.dang_hoi:
            self.dang_hoi = True
            self.yeu_cau.put((loai, loi_nhac))

    def lay(self):
        try:
            kq = self.tra_loi.get_nowait()
        except queue.Empty:
            return None
        self.dang_hoi = False
        return kq


def thong_tin_ten_anh(ten, goc):
    """Đọc số thứ tự, lượt và tư thế từ tên PNG của bộ thu."""
    phan = goc.rsplit("_", 3)
    if (len(phan) != 4 or phan[0] != ten or not phan[1].isdigit()
            or not phan[2].startswith("L") or not phan[2][1:].isdigit()):
        return None
    return int(phan[1]), int(phan[2][1:]), phan[3]


class TienDo:
    """Đếm ảnh trong metadata và ảnh xám đã xuất để tiếp tục đúng phần còn thiếu."""

    def __init__(self, ten):
        self.ten = ten
        self.dem = {k: 0 for k in KE_HOACH}
        self.dem_khoang = {(l, b): [0] * len(TU_THE[b][4] or []) for (l, b) in KE_HOACH}
        self.so_max = 0
        da_ghi = set()
        for d in doc_csv():
            if d.get("ten") != ten:
                continue
            try:                                          # bỏ qua dòng hỏng, không làm sập chương trình
                k = (int(d["luot"]) - 1, CHI_SO_MA.get(d["tu_the"], -1))
                kg, so = int(d["khoang_goc"]), int(d["so"])
            except (ValueError, KeyError, TypeError):
                continue
            if k in self.dem:
                self.dem[k] += 1
                if 0 <= kg < len(self.dem_khoang[k]):
                    self.dem_khoang[k][kg] += 1
            self.so_max = max(self.so_max, so)
            da_ghi.add(d.get("file"))
        # Bản dữ liệu trên Git có ảnh xám và tên tư thế, có thể thiếu metadata.csv.
        for anh in (Path(THU_MUC_XAM) / ten).glob("*.png"):
            thong_tin = thong_tin_ten_anh(ten, anh.stem)
            if thong_tin is None:
                continue
            so, luot, ma = thong_tin
            self.so_max = max(self.so_max, so)
            k = (luot - 1, CHI_SO_MA.get(ma, -1))
            if anh.stem in da_ghi or k not in self.dem:
                continue
            if (Path(THU_MUC_MAU, ten, anh.name).exists()
                    or Path(THU_MUC_ANH_GOC, ten, anh.stem + ".jpg").exists()):
                continue  # bản thu gián đoạn: kiem_tra.py sẽ đối chiếu với metadata
            # Bộ thu hiện tại có tối đa một khoảng góc cho mỗi tư thế.
            khoang = self.dem_khoang[k]
            if len(khoang) <= 1:
                xam = doc_anh(anh, co_mau=False) if anh.stat().st_size else None
                if xam is not None and xam.shape == (100, 100):
                    self.dem[k] += 1
                    if khoang:
                        khoang[0] += 1

    def tong(self):
        return sum(self.dem.values())

    def muc_tiep(self):
        for i, (l, b) in enumerate(KE_HOACH):
            if self.dem[(l, b)] < TU_THE[b][3]:
                return i
        return None

    def con_thieu(self):
        """Danh sách phần còn thiếu, vd ['L1 trai 7-18°: 1', 'L1 thang: 2']."""
        ds = []
        for (l, b) in KE_HOACH:
            ma, _, _, so_anh, khoang = TU_THE[b]
            if khoang:
                ct = chi_tieu_khoang(so_anh, len(khoang))
                for (lo, hi), c, t in zip(khoang, self.dem_khoang[(l, b)], ct):
                    if c < t:
                        ds.append(f"L{l + 1} {ma} {lo}-{hi}°: {t - c}")
            elif self.dem[(l, b)] < so_anh:
                ds.append(f"L{l + 1} {ma}: {so_anh - self.dem[(l, b)]}")
        return ds


# =====================================================================
# 4. LUẬT KIỂM TRA
# =====================================================================
def kiem_tra_chung(info, W, H, sang, tuong_phan):
    """Kiểm tra vị trí, kích thước mặt và ánh sáng. Trả về (đạt?, gợi ý)."""
    x1, y1, x2, y2 = info["khung"]
    cao_mat = y2 - y1
    lx, ly = LE_KHUNG_TI_LE * W, LE_KHUNG_TI_LE * H
    if x1 < lx or y1 < ly or x2 > W - lx or y2 > H - ly:
        return False, "Dua mat vao GIUA khung hinh"
    if cao_mat < MAT_TOI_THIEU_PX:
        return False, "Lai GAN camera hon"
    if cao_mat > MAT_TOI_DA_TI_LE * H:
        return False, "Lui XA camera hon"
    if sang < DO_SANG_MIN:
        return False, "Thieu sang - bat den / quay mat ve phia den"
    if sang > DO_SANG_MAX:
        return False, "Qua sang / chay sang - giam den"
    if tuong_phan < TUONG_PHAN_MIN:
        return False, "Anh nhat - kiem tra anh sang"
    return True, ""


def vung_muc_tieu(buoc, dem_khoang):
    """Các vùng (yaw_min, yaw_max, pitch_min, pitch_max, da_du) của bước để vẽ biểu đồ."""
    ma, _, _, so_anh, khoang = buoc
    T, B = THANG_TOI_DA, BIEU_CAM_LECH_TOI_DA
    if ma == "thang":
        return [(-T, T, -T, T, False)]
    if ma in ("cuoi", "noi"):
        return [(-B, B, -B, B, False)]
    ct = chi_tieu_khoang(so_anh, len(khoang))
    vung = []
    for i, (lo, hi) in enumerate(khoang):
        du = dem_khoang[i] >= ct[i]
        if ma == "trai":
            vung.append((lo, hi, -LECH_PHU_QUAY, LECH_PHU_QUAY, du))
        elif ma == "phai":
            vung.append((-hi, -lo, -LECH_PHU_QUAY, LECH_PHU_QUAY, du))
        elif ma == "len":
            vung.append((-LECH_PHU_NGANG_CUI, LECH_PHU_NGANG_CUI, lo, hi, du))
        else:
            vung.append((-LECH_PHU_NGANG_CUI, LECH_PHU_NGANG_CUI, -hi, -lo, du))
    return vung


def kiem_tra_tu_the(buoc, dem_khoang, dyaw, dpitch, cuoi, ha_mieng, nham):
    """Trả về (đúng?, gợi ý, chỉ số khoảng góc, hướng cần chỉnh: +1 thêm / -1 bớt / 0)."""
    ma, _, _, so_anh, khoang = buoc
    if nham > (NGUONG_NHAM_MAT_KHI_CUI if ma == "xuong" else NGUONG_NHAM_MAT):
        return False, "Mo mat ra", -1, 0
    if ma != "cuoi" and cuoi > CUOI_TOI_DA:
        return False, "Dung cuoi o buoc nay", -1, 0

    if ma == "thang":
        if abs(dyaw) > THANG_TOI_DA:
            return False, "Quay mat ve CHINH GIUA", -1, 0
        if abs(dpitch) > THANG_TOI_DA:
            return False, "Dau thang lai (dung ngang/cui)", -1, 0
        return True, "", -1, 0

    if ma in ("cuoi", "noi"):
        if abs(dyaw) > BIEU_CAM_LECH_TOI_DA or abs(dpitch) > BIEU_CAM_LECH_TOI_DA:
            return False, "Nhin thang vao camera", -1, 0
        if ma == "cuoi" and cuoi < NGUONG_CUOI:
            return False, "Cuoi tuoi hon nua", -1, 0
        if ma == "noi" and ha_mieng < NGUONG_HA_MIENG:
            return False, "Ha mieng to hon / noi chuyen", -1, 0
        return True, "", -1, 0

    # ---- các bước có chia khoảng góc: trai / phai / len / xuong ----
    ngang = ma in ("trai", "phai")
    v = {"trai": dyaw, "phai": -dyaw, "len": dpitch, "xuong": -dpitch}[ma]
    phu = abs(dpitch) if ngang else abs(dyaw)
    dong_tu = {"trai": "Quay", "phai": "Quay", "len": "Ngang", "xuong": "Cui"}[ma]
    huong = {"trai": "sang TRAI", "phai": "sang PHAI", "len": "LEN", "xuong": "XUONG"}[ma]

    if phu > (LECH_PHU_QUAY if ngang else LECH_PHU_NGANG_CUI):
        return False, ("Giu dau ngang (dung ngang/cui)" if ngang else "Dung quay trai/phai"), -1, 0
    if v < -SAI_HUONG:
        return False, f"SAI HUONG! {dong_tu} {huong}", -1, 1

    ct = chi_tieu_khoang(so_anh, len(khoang))
    con_thieu = [i for i in range(len(khoang)) if dem_khoang[i] < ct[i]]
    kg = next((i for i, (lo, hi) in enumerate(khoang)
               if lo <= v < hi or (i == len(khoang) - 1 and v == hi)), -1)
    if not con_thieu:            # (phòng hờ) mọi khoảng đã đủ mà bước chưa đủ -> nhận mọi góc hợp lệ
        con_thieu = list(range(len(khoang)))
    if kg in con_thieu:
        return True, "", kg, 0
    # chưa vào khoảng nào / khoảng hiện tại đã đủ -> chỉ tới khoảng còn thiếu gần nhất
    gan = min(con_thieu, key=lambda i: abs((khoang[i][0] + khoang[i][1]) / 2 - v))
    lo, hi = khoang[gan]
    if v < lo:
        return False, f"{dong_tu} THEM nua (can {lo}-{hi} do)", -1, 1
    return False, f"{dong_tu} IT lai (can {lo}-{hi} do)", -1, -1


# =====================================================================
# 5. PHIÊN THU DỮ LIỆU
# =====================================================================
class PhienThu:
    def __init__(self, hoi, phien):
        self.hoi, self.phien = hoi, phien
        self.trang_thai, self.thoat, self.tam_dung = "cho_ten", False, False
        self.ten, self.td, self.td_cho, self.k = None, None, None, 0
        self.moc, self.mau_moc = None, []
        self.t_nghi = self.t_anh_truoc = self.t_flash = 0.0
        self.dung_lien, self.gan_day, self.anh_vua_chup = 0, [], None
        self.goi_y, self.dung, self.huong = "", False, 0
        self.do_net = self.do_sang = self.tuong_phan = self.do_khac = 0.0
        self.dyaw = self.dpitch = 0.0
        self.dang_in_dong = False

    # ---------- in ra terminal ----------
    def in_(self, chu):
        if self.dang_in_dong:
            print()
            self.dang_in_dong = False
        print(chu)

    def in_tien_do(self):
        l, b = KE_HOACH[self.k]
        ma, _, nhac, so_anh, khoang = TU_THE[b]
        chu = f"\r    {self.td.dem[(l, b)]:3d}/{so_anh} ảnh"
        if khoang:
            ct = chi_tieu_khoang(so_anh, len(khoang))
            chu += "  [" + "  ".join(f"{lo}-{hi}°: {c}/{t}" for (lo, hi), c, t
                                    in zip(khoang, self.td.dem_khoang[(l, b)], ct)) + "]"
        chu += f"   | tổng {self.td.tong()}/{TONG_MOI_NGUOI}"
        print(chu, end="", flush=True)
        self.dang_in_dong = True

    # ---------- chuyển trạng thái ----------
    def bat_dau_ten(self):
        self.trang_thai, self.ten, self.td, self.tam_dung = "cho_ten", None, None, False
        self.anh_vua_chup, self.moc, self.mau_moc = None, None, []
        self.hoi.hoi("ten", "\n>>> Nhập TÊN người cần thu (vd: Nguyen Van Lam). "
                            "Để trống + Enter = KẾT THÚC: ")

    def chuan_bi_luot(self):
        self.k = self.td.muc_tiep()
        l = KE_HOACH[self.k][0]
        self.in_(f"\n=== {self.ten} – LƯỢT {l + 1}/{len(LUOT)} ===")
        self.in_("  " + LUOT[l])
        self.trang_thai = "cho_luot"
        self.hoi.hoi("luot", "  Ngồi vào vị trí, sẵn sàng thì bấm Enter để bắt đầu... ")

    def bat_dau_moc(self):
        self.trang_thai, self.moc, self.mau_moc, self.tam_dung = "moc", None, [], False
        self.in_("  Lấy mốc: nhìn THẲNG vào camera, nét mặt bình thường, giữ yên...")

    def bat_dau_nghi(self):
        self.trang_thai, self.t_nghi, self.dung_lien, self.gan_day = "nghi", time.time(), 0, []
        l, b = KE_HOACH[self.k]
        self.in_(f"\n  [Lượt {l + 1} | Bước {b + 1}/{len(TU_THE)}] {TU_THE[b][2]}")
        self.in_tien_do()

    def sang_muc_tiep(self):
        l_cu = KE_HOACH[self.k][0]
        k_moi = self.td.muc_tiep()
        if k_moi is None:
            self.ket_thuc_nguoi()
            return
        self.k = k_moi
        if KE_HOACH[k_moi][0] != l_cu:
            self.in_(f"\n  -> XONG LƯỢT {l_cu + 1} của {self.ten}.")
            bip(900, 400)
            self.chuan_bi_luot()
        else:
            bip(1300, 150)
            self.bat_dau_nghi()

    def ket_thuc_nguoi(self):
        self.in_(f"\n=== HOÀN THÀNH {self.ten}: {self.td.tong()}/{TONG_MOI_NGUOI} ảnh ===")
        duong_dan = tao_anh_xem_nhanh(self.ten)
        if duong_dan:
            self.in_(f"  Ảnh xem nhanh để soát lại: {duong_dan}")
        bip(1500, 500)
        self.bat_dau_ten()

    # ---------- trả lời từ terminal ----------
    def nhan_tra_loi(self, loai, s):
        if loai == "ten":
            if not s.strip():
                self.thoat = True
                return
            ten = chuan_hoa_ten(s)
            if not ten or any(ky_tu in s for ky_tu in "\\/:>") or s.strip().lower().endswith(".py"):
                self.in_("  Chỉ nhập tên người. Hãy chạy lệnh Python ở dấu nhắc PowerShell.")
                self.bat_dau_ten()
                return
            td = TienDo(ten)
            if td.tong() == 0:
                self.in_(f"  -> Tên lưu trong dữ liệu: {ten}")
                self.td, self.ten = td, ten
                self.chuan_bi_luot()
            elif td.muc_tiep() is None:
                self.in_(f"  '{ten}' đã đủ {td.tong()} ảnh rồi. Nhập người khác.")
                self.bat_dau_ten()
            else:
                self.td_cho, self.trang_thai = td, "hoi_tiep"
                thieu = td.con_thieu()
                self.in_(f"  '{ten}' đã có {td.tong()}/{TONG_MOI_NGUOI} ảnh. Còn thiếu "
                         f"{TONG_MOI_NGUOI - td.tong()} ảnh: " + ", ".join(thieu[:12])
                         + (" ..." if len(thieu) > 12 else ""))
                self.hoi.hoi("tiep", "  Gõ c + Enter để CHỤP BÙ đúng phần còn thiếu, chỉ Enter để nhập tên khác: ")
        elif loai == "tiep":
            if s.strip().lower() == "c":
                self.td, self.ten = self.td_cho, self.td_cho.ten
                self.chuan_bi_luot()
            else:
                self.bat_dau_ten()
        elif loai == "luot":
            self.bat_dau_moc()

    # ---------- phím bấm ----------
    def xu_ly_phim(self, phim):
        if phim in (ord("q"), 27):
            self.thoat = True
        elif phim == ord("p") and self.trang_thai in ("moc", "nghi", "chup"):
            self.tam_dung = not self.tam_dung
            self.in_("  || TẠM DỪNG (bấm p để tiếp tục)" if self.tam_dung else "  > Tiếp tục")
            if not self.tam_dung:
                self.t_nghi, self.dung_lien = time.time(), 0
                if self.trang_thai == "chup":
                    self.trang_thai = "nghi"
        elif phim == ord("m") and self.trang_thai in ("nghi", "chup"):
            self.bat_dau_moc()

    # ---------- xử lý 1 khung hình ----------
    def xu_ly(self, raw, info, so_mat):
        H, W = raw.shape[:2]
        self.dung, self.huong, self.goi_y = False, 0, ""
        xam = None
        if info is None:
            self.goi_y = "Chi 1 nguoi trong hinh!" if so_mat > 1 else "Khong thay mat"
            ok_chung = False
        else:
            xam = cv2.cvtColor(can_chinh_mat(raw, info["d5"], 100), cv2.COLOR_BGR2GRAY)
            self.do_net, self.do_sang, self.tuong_phan = chat_luong(xam)
            ok_chung, self.goi_y = kiem_tra_chung(info, W, H, self.do_sang, self.tuong_phan)
            if self.moc is not None:
                self.dyaw, self.dpitch = info["yaw"] - self.moc[0], info["pitch"] - self.moc[1]

        if self.tam_dung:
            self.dung_lien = 0
            return

        if self.trang_thai == "moc":
            self._xu_ly_moc(info, ok_chung)
        elif self.trang_thai == "nghi":
            self.dung_lien = 0
            if time.time() - self.t_nghi >= CHUYEN_TU_THE:
                self.trang_thai = "chup"
        elif self.trang_thai == "chup":
            self._xu_ly_chup(raw, info, xam, ok_chung)

    def _xu_ly_moc(self, info, ok_chung):
        ok, goi_y = ok_chung, self.goi_y
        if ok:
            if abs(info["yaw"]) > MOC_GOC_TUYET_DOI or abs(info["pitch"]) > MOC_GOC_TUYET_DOI:
                ok, goi_y = False, f"Nhin THANG camera (yaw {info['yaw']:+.0f} pitch {info['pitch']:+.0f})"
            elif info["cuoi"] > 0.35:
                ok, goi_y = False, "Giu net mat binh thuong"
            elif info["nham"] > NGUONG_NHAM_MAT:
                ok, goi_y = False, "Mo mat ra"
        if not ok:
            self.mau_moc, self.goi_y = [], goi_y
            return
        self.mau_moc.append((info["yaw"], info["pitch"]))
        if len(self.mau_moc) > SO_KHUNG_MOC:
            self.mau_moc.pop(0)
        if len(self.mau_moc) < SO_KHUNG_MOC:
            self.goi_y = f"Giu yen... {len(self.mau_moc)}/{SO_KHUNG_MOC}"
            return
        mau = np.array(self.mau_moc)
        if mau.std(0).max() > MOC_DO_LECH_TOI_DA:
            self.goi_y = "Giu YEN dau"
            return
        self.moc = tuple(np.median(mau, 0))
        self.in_(f"  Đã lấy mốc (yaw {self.moc[0]:+.1f}°, pitch {self.moc[1]:+.1f}°).")
        self.bat_dau_nghi()

    def _xu_ly_chup(self, raw, info, xam, ok_chung):
        if info is None or not ok_chung:
            self.dung_lien = 0
            return
        l, b = KE_HOACH[self.k]
        buoc = TU_THE[b]
        dung, goi_y, kg, self.huong = kiem_tra_tu_the(
            buoc, self.td.dem_khoang[(l, b)], self.dyaw, self.dpitch,
            info["cuoi"], info["ha_mieng"], info["nham"])
        if dung and self.do_net < NGUONG_NET:
            dung, goi_y = False, "Anh mo - cu dong cham lai"
        self.dung, self.goi_y = dung, goi_y
        self.dung_lien = self.dung_lien + 1 if dung else 0
        if not dung or self.dung_lien < GIU_KHUNG or time.time() - self.t_anh_truoc < KHOANG_CACH:
            return
        self.do_khac = min((float(np.mean(cv2.absdiff(xam, a))) for a in self.gan_day), default=99.0)
        if self.do_khac < NGUONG_KHAC:
            self.dung, self.goi_y = False, "Dung yen - nhuc nhich dau / doi net mat"
            return
        self._luu(raw, info, xam, l, b, kg)

    def _luu(self, raw, info, xam, l, b, kg):
        ma = TU_THE[b][0]
        so = self.td.so_max + 1
        ten_file = f"{self.ten}_{so:04d}_L{l + 1}_{ma}"
        mau112 = can_chinh_mat(raw, info["d5"], 112)
        ok = luu_anh(os.path.join(THU_MUC_MAU, self.ten, ten_file + ".png"), mau112)
        ok = ok and luu_anh(os.path.join(THU_MUC_XAM, self.ten, ten_file + ".png"), xam)
        if LUU_ANH_GOC:
            ok = ok and luu_anh(os.path.join(THU_MUC_ANH_GOC, self.ten, ten_file + ".jpg"), raw,
                                ".jpg", [cv2.IMWRITE_JPEG_QUALITY, CHAT_LUONG_JPG])
        if not ok:
            self.in_("  !! Lỗi ghi file ảnh (ổ đĩa đầy / không có quyền ghi?)")
            return
        H, W = raw.shape[:2]
        x1, y1, x2, y2 = info["khung"]
        d5 = info["d5"]
        ghi_dong_csv({
            "file": ten_file, "so": so, "ten": self.ten, "luot": l + 1, "tu_the": ma,
            "khoang_goc": kg, "phien": self.phien,
            "thoi_gian": datetime.now().isoformat(timespec="milliseconds"),
            "yaw": round(info["yaw"], 2), "pitch": round(info["pitch"], 2), "roll": round(info["roll"], 2),
            "yaw_tuong_doi": round(self.dyaw, 2), "pitch_tuong_doi": round(self.dpitch, 2),
            "cuoi": round(info["cuoi"], 3), "ha_mieng": round(info["ha_mieng"], 3),
            "nham_mat": round(info["nham"], 3), "do_net": round(self.do_net, 1),
            "do_sang": round(self.do_sang, 1), "tuong_phan": round(self.tuong_phan, 1),
            "do_khac": round(self.do_khac, 2), "rong_anh": W, "cao_anh": H,
            "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
            "mat1_x": round(d5[0, 0], 2), "mat1_y": round(d5[0, 1], 2),
            "mat2_x": round(d5[1, 0], 2), "mat2_y": round(d5[1, 1], 2),
            "mui_x": round(d5[2, 0], 2), "mui_y": round(d5[2, 1], 2),
            "mieng1_x": round(d5[3, 0], 2), "mieng1_y": round(d5[3, 1], 2),
            "mieng2_x": round(d5[4, 0], 2), "mieng2_y": round(d5[4, 1], 2),
        })
        self.td.so_max = so
        self.td.dem[(l, b)] += 1
        if kg >= 0:
            self.td.dem_khoang[(l, b)][kg] += 1
        self.gan_day = (self.gan_day + [xam])[-SO_ANH_SO_SANH:]
        self.anh_vua_chup, self.t_flash = mau112, time.time()
        self.t_anh_truoc, self.dung_lien = time.time(), 0
        self.in_tien_do()
        if self.td.dem[(l, b)] >= TU_THE[b][3]:
            self.sang_muc_tiep()

    # ---------- vẽ màn hình ----------
    def ve(self, raw, info, fps):
        H, W = raw.shape[:2]
        s = HIEN_THI_CAO / H
        anh = cv2.resize(raw, (max(1, int(W * s)), HIEN_THI_CAO), interpolation=cv2.INTER_AREA)
        if info is not None:
            x1, y1, x2, y2 = (int(v * s) for v in info["khung"])
            cv2.rectangle(anh, (x1, y1), (x2, y2), XANH if self.dung else DO, 2)
            for px, py in info["d5"]:
                cv2.circle(anh, (int(px * s), int(py * s)), 2, (255, 255, 0), -1)
        anh = cv2.flip(anh, 1)                       # hiển thị như soi gương
        Wd = anh.shape[1]
        if self.trang_thai == "chup" and not self.dung and self.huong != 0:
            ma = TU_THE[KE_HOACH[self.k][1]][0]
            vec = {"trai": (-1, 0), "phai": (1, 0), "len": (0, -1), "xuong": (0, 1)}.get(ma)
            if vec:
                cx, cy = Wd // 2, 45
                dx, dy = vec[0] * self.huong * 35, vec[1] * self.huong * 25
                cv2.arrowedLine(anh, (cx - dx, cy - dy), (cx + dx, cy + dy), VANG, 4, tipLength=0.4)
        if time.time() - self.t_flash < 0.15:
            cv2.rectangle(anh, (0, 0), (Wd - 1, HIEN_THI_CAO - 1), XANH, 8)
        if self.tam_dung:
            ve_chu(anh, "TAM DUNG", 10, HIEN_THI_CAO // 2, VANG, 0.8, 2)
        return np.hstack([anh, self._ve_bang(info, fps)])

    def _ve_bang(self, info, fps):
        h = HIEN_THI_CAO
        bang = np.zeros((h, RONG_BANG, 3), np.uint8)
        tt = self.trang_thai
        buoc = None
        if tt == "cho_ten":
            ve_chu(bang, "NHAP TEN O TERMINAL", 10, 26, DO, 0.65, 2)
            ve_chu(bang, "(click vao terminal roi go ten)", 10, 50, XAM, 0.45)
        elif tt == "hoi_tiep":
            ve_chu(bang, "TRA LOI O TERMINAL (c / Enter)", 10, 26, DO, 0.6, 2)
        elif tt == "cho_luot":
            l = KE_HOACH[self.k][0]
            ve_chu(bang, f"LUOT {l + 1}/{len(LUOT)}: doc huong dan o terminal", 10, 26, VANG, 0.55)
            ve_chu(bang, "Ngoi vao vi tri roi bam Enter o terminal", 10, 50, VANG, 0.5)
        elif tt == "moc":
            ve_chu(bang, "LAY MOC: nhin THANG vao camera", 10, 26, VANG, 0.55)
            ve_chu(bang, "Giu yen, net mat binh thuong", 10, 50, VANG, 0.5)
        else:
            l, b = KE_HOACH[self.k]
            buoc = TU_THE[b]
            ve_chu(bang, f"LUOT {l + 1}/{len(LUOT)} - BUOC {b + 1}/{len(TU_THE)}", 10, 26, VANG, 0.55)
            if tt == "nghi" and not self.tam_dung:
                con = max(0.0, CHUYEN_TU_THE - (time.time() - self.t_nghi))
                ve_chu(bang, f"Chuan bi {con:.0f}s: {buoc[1]}", 10, 50, VANG, 0.55)
            else:
                ve_chu(bang, buoc[1], 10, 50, VANG, 0.6, 2)

        if self.td is not None and tt not in ("cho_ten", "hoi_tiep"):
            l, b = KE_HOACH[self.k]
            ve_chu(bang, f"{self.ten}: {self.td.tong()}/{TONG_MOI_NGUOI}   buoc: "
                         f"{self.td.dem[(l, b)]}/{TU_THE[b][3]}", 10, 76, DO, 0.5)
        if tt == "chup" and self.dung:
            ve_chu(bang, "DUNG TU THE - dang chup", 10, 100, XANH, 0.55)
        elif self.goi_y:
            ve_chu(bang, self.goi_y, 10, 100, CAM, 0.5)

        # biểu đồ góc đầu
        self._ve_bieu_do(bang, 10, 112, 150, buoc, info)
        # ảnh vừa chụp
        if self.anh_vua_chup is not None:
            bang[112:224, 172:284] = self.anh_vua_chup
            ve_chu(bang, "anh vua chup", 178, 238, XAM, 0.4)
        # trạng thái các khoảng góc
        if buoc is not None and buoc[4]:
            l, b = KE_HOACH[self.k]
            ct = chi_tieu_khoang(buoc[3], len(buoc[4]))
            ve_chu(bang, "Khoang goc:", 296, 126, XAM, 0.42)
            for i, ((lo, hi), c, t) in enumerate(zip(buoc[4], self.td.dem_khoang[(l, b)], ct)):
                ve_chu(bang, f"{lo}-{hi} do: {c}/{t}", 296, 148 + 20 * i, XANH if c >= t else TRANG, 0.42)

        # chỉ số trực tiếp
        if info is not None:
            if self.moc is not None:
                ve_chu(bang, f"Yaw {self.dyaw:+5.1f}  Pitch {self.dpitch:+5.1f}  (do, so voi moc)",
                       10, h - 64, XAM, 0.42)
            else:
                ve_chu(bang, f"Yaw {info['yaw']:+5.1f}  Pitch {info['pitch']:+5.1f}  (chua co moc)",
                       10, h - 64, XAM, 0.42)
            ve_chu(bang, f"Cuoi {info['cuoi']:.2f}  Ha mieng {info['ha_mieng']:.2f}  "
                         f"Nham mat {info['nham']:.2f}", 10, h - 45, XAM, 0.42)
            ve_chu(bang, f"Net {self.do_net:.0f}  Sang {self.do_sang:.0f}  Tuong phan {self.tuong_phan:.0f}"
                         f"  Khac {self.do_khac:.1f}", 10, h - 26, XAM, 0.42)
        ve_chu(bang, f"q thoat | p tam dung | m lay lai moc     FPS {fps:4.1f}", 10, h - 7, XAM, 0.4)
        return bang

    def _ve_bieu_do(self, bang, x0, y0, kich, buoc, info):
        """Biểu đồ yaw (ngang) – pitch (dọc) như soi gương: TRAI bên trái, LEN phía trên."""
        gioi_han = 25.0
        sc = (kich / 2) / gioi_han
        cx, cy = x0 + kich // 2, y0 + kich // 2
        cv2.rectangle(bang, (x0, y0), (x0 + kich, y0 + kich), (90, 90, 90), 1)
        cv2.line(bang, (cx, y0), (cx, y0 + kich), (60, 60, 60), 1)
        cv2.line(bang, (x0, cy), (x0 + kich, cy), (60, 60, 60), 1)

        def td(yw, pt):
            yw = max(-gioi_han, min(gioi_han, yw))
            pt = max(-gioi_han, min(gioi_han, pt))
            return int(round(cx - yw * sc)), int(round(cy - pt * sc))

        if buoc is not None:
            l, b = KE_HOACH[self.k]
            for (ya, yb, pa, pb, du) in vung_muc_tieu(buoc, self.td.dem_khoang[(l, b)]):
                (xa, ya_), (xb, yb_) = td(ya, pa), td(yb, pb)
                cv2.rectangle(bang, (min(xa, xb), min(ya_, yb_)), (max(xa, xb), max(ya_, yb_)),
                              (0, 120, 0) if du else VANG, 1 if du else 2)
        ve_chu(bang, "TRAI", x0 + 3, cy + 36, XAM, 0.35)
        ve_chu(bang, "PHAI", x0 + kich - 33, cy + 36, XAM, 0.35)
        ve_chu(bang, "LEN", cx + 4, y0 + 12, XAM, 0.35)
        ve_chu(bang, "XUONG", cx + 4, y0 + kich - 4, XAM, 0.35)
        if info is not None and self.moc is not None:
            cv2.circle(bang, td(self.dyaw, self.dpitch), 5, XANH if self.dung else DO, -1)


# =====================================================================
# 6. CHƯƠNG TRÌNH CHÍNH
# =====================================================================
def in_huong_dan():
    print("=" * 70)
    print(" THU THẬP DỮ LIỆU KHUÔN MẶT – CHUẨN BỊ (chỉ thu 1 lần, đọc kỹ)")
    print("=" * 70)
    print(f" 1. Nguồn camera: {CAMERA}. Đặt camera cố định, ngang tầm mắt.")
    print(" 2. Đèn/cửa sổ ở phía trước mặt. Chỉ một người trong khung hình.")
    if isinstance(XOAY, str):
        print(" 3. Nhìn thẳng lúc khởi động để chương trình tự nhận chiều camera.")
    else:
        print(f" 3. Chiều camera cố định: {TEN_XOAY[XOAY]}.")
    print(" 4. Nhập tên người trong terminal, sau đó nhìn thẳng để lấy mốc.")
    print(f" 5. Mỗi người {TONG_MOI_NGUOI} ảnh. Quay, ngẩng và cúi nhẹ theo gợi ý.")
    print("    Khung XANH = đúng tư thế, đang chụp | Khung ĐỎ = chưa đúng, đọc gợi ý màu cam.")
    print("    Biểu đồ: chấm phải nằm trong ô vàng. Mũi tên vàng chỉ hướng cần chỉnh.")
    print(" 6. Thu xong 1 người, mở dataset/xem_nhanh/<ten>.jpg để soát lại ảnh.")
    print("=" * 70)


def main():
    phien = datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in (THU_MUC_DATA, THU_MUC_ANH_GOC, THU_MUC_MAU, THU_MUC_XAM, THU_MUC_XEM):
        os.makedirs(d, exist_ok=True)
    in_huong_dan()

    landmarker = tao_landmarker("video")
    print(f"Đang kết nối camera {CAMERA} ...")
    cam = CameraLuong(CAMERA)
    if not cam.cho_khung_dau(15):
        cam.dung()
        landmarker.close()
        if isinstance(CAMERA, int):
            raise SystemExit("Không mở được webcam. Đóng ứng dụng đang dùng camera rồi thử lại.")
        raise SystemExit("Không nhận được IP Webcam. Kiểm tra Start server, URL và Wi-Fi.")
    print("Kết nối camera thành công.")
    cv2.namedWindow(TEN_CUA_SO, cv2.WINDOW_AUTOSIZE)

    # ---- chiều xoay hình ----
    if isinstance(XOAY, str):
        print("Đang TỰ NHẬN CHIỀU CAMERA: hãy nhìn thẳng vào camera vài giây...")
        ok, xoay = tu_nhan_chieu(cam)
        if ok is None:                                   # bấm q trong lúc tự nhận chiều
            cam.dung()
            landmarker.close()
            cv2.destroyAllWindows()
            print("Đã thoát.")
            return
        if ok:
            print(f"  -> Chiều camera: {TEN_XOAY[xoay]}")
        else:
            xoay = XOAY_DU_PHONG
            print(f"  ! Không tự nhận được (không thấy mặt). Dùng chiều dự phòng: {TEN_XOAY.get(xoay, xoay)}.")
            print("    Nếu hình bị ngang/ngược: thoát (q), đặt XOAY = cv2.ROTATE_... ở đầu file rồi chạy lại.")
    else:
        xoay = XOAY
    raw0 = chuan_bi_khung(cam.doc()[1], xoay)
    H0, W0 = raw0.shape[:2]
    print(f"Khung hình sau khi xoay: {W0}x{H0} ({'dọc' if H0 > W0 else 'ngang'})")
    if min(W0, H0) < 480:
        print("  ! Độ phân giải thấp. Dùng camera có độ phân giải cao hơn nếu ảnh bị mờ.")
    ghi_cau_hinh(phien, H0, W0, xoay)
    print(f"Dữ liệu lưu tại: {THU_MUC_DATA}")
    hoi = HoiDap()
    pt = PhienThu(hoi, phien)
    pt.bat_dau_ten()

    id_cu, t_khung_moi, t_truoc, fps = -1, time.time(), None, 0.0
    t0, ts_truoc, anh_cuoi = time.perf_counter(), -1, None
    try:
        while not pt.thoat:
            tl = hoi.lay()
            if tl:
                pt.nhan_tra_loi(*tl)
                if pt.thoat:
                    break

            fid, frame = cam.doc()
            if frame is None or fid == id_cu:            # chưa có khung hình mới
                if anh_cuoi is not None and time.time() - t_khung_moi > 2.0:
                    canh_bao = anh_cuoi.copy()
                    ve_chu(canh_bao, "MAT KET NOI CAMERA - dang thu lai...", 10, HIEN_THI_CAO // 2, DO, 0.6, 2)
                    cv2.imshow(TEN_CUA_SO, canh_bao)
                pt.xu_ly_phim(cv2.waitKey(5) & 0xFF)
                continue
            id_cu, bay_gio = fid, time.time()
            if t_truoc is not None:
                tuc_thoi = 1.0 / max(bay_gio - t_truoc, 1e-3)
                fps = tuc_thoi if fps == 0 else 0.9 * fps + 0.1 * tuc_thoi
            t_truoc = t_khung_moi = bay_gio

            raw = chuan_bi_khung(frame, xoay)
            H, W = raw.shape[:2]
            if H > XU_LY_CAO:
                tl_xl = XU_LY_CAO / H
                anh_xl = cv2.resize(raw, (max(1, int(W * tl_xl)), XU_LY_CAO), interpolation=cv2.INTER_AREA)
            else:
                anh_xl = raw
            ts = max(int((time.perf_counter() - t0) * 1000), ts_truoc + 1)
            ts_truoc = ts
            kq = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(anh_xl, cv2.COLOR_BGR2RGB)), ts)

            so_mat = len(kq.face_landmarks)
            info = None
            if so_mat == 1:
                bs = kq.face_blendshapes[0] if kq.face_blendshapes else None
                info = trich_thong_tin(kq.face_landmarks[0], bs, W, H)

            pt.xu_ly(raw, info, so_mat)
            anh_cuoi = pt.ve(raw, info, fps)
            cv2.imshow(TEN_CUA_SO, anh_cuoi)
            pt.xu_ly_phim(cv2.waitKey(1) & 0xFF)
            try:                    # bấm dấu X đóng cửa sổ = thoát (thu nhỏ cửa sổ thì KHÔNG thoát)
                if cv2.getWindowProperty(TEN_CUA_SO, cv2.WND_PROP_AUTOSIZE) < 0:
                    pt.thoat = True
            except cv2.error:
                pt.thoat = True
    finally:
        cam.dung()
        cv2.destroyAllWindows()
        landmarker.close()
        if pt.ten and pt.td and pt.td.muc_tiep() is not None:
            pt.in_(f"\n(!) {pt.ten} chưa thu xong ({pt.td.tong()}/{TONG_MOI_NGUOI}). "
                   f"Chạy lại, nhập đúng tên '{pt.ten}' rồi gõ c để chụp tiếp.")
        in_tong_ket()


if __name__ == "__main__":
    main()
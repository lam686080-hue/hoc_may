"""
=====================================================================
 KIỂM TRA DỮ LIỆU KHUÔN MẶT ĐÃ THU  (chạy riêng, KHÔNG cần thu_anh.py)
=====================================================================
Cài đặt (1 lần):  pip install mediapipe opencv-python numpy
Chạy:             python kiem_tra.py               (thư mục dataset cạnh file này)
            hoặc  python kiem_tra.py <đường_dẫn_dataset>
!! Hãy TẮT thu_anh.py trước khi chạy kiểm tra.

Ngưỡng kiểm tra được đọc từ dataset/cau_hinh_thu_thap.json (do thu_anh.py ghi lúc thu)
-> kiểm tra đúng theo cấu hình đã dùng khi chụp.

6 NHÓM KIỂM TRA
  [1] Đủ & cân bằng : mỗi người đủ số ảnh từng lượt / tư thế / khoảng góc
  [2] File          : đủ 3 bản (gốc, màu 112, xám 100), mở được, đúng kích thước, không có file lạ
  [3] Lúc chụp      : thông số ghi trong metadata.csv đạt mọi ngưỡng
  [4] Đo lại        : chạy lại MediaPipe trên ẢNH GỐC (độc lập với lúc chụp):
                      có mặt, đúng hướng, đúng tư thế, mặt cắt khớp ảnh đã lưu
  [5] Chất lượng    : ảnh xám đủ nét / sáng / tương phản, khớp thông số lúc chụp; ảnh gần trùng
  [6] Nhãn          : ảnh giống người KHÁC hơn hẳn chính mình (nghi gõ nhầm tên)

MỨC ĐỘ
  LOI      : ảnh hỏng thật -> NÊN THU LẠI
  CANH_BAO : nghi vấn -> mở lưới ảnh xem bằng mắt, đa số vẫn dùng được

THU LẠI ẢNH LỖI
  Cuối báo cáo, script hỏi có chuyển ảnh lỗi sang dataset/loai_bo/<thời_gian>/ không.
  Đồng ý -> ảnh lỗi được CHUYỂN (không xóa) + bỏ khỏi metadata.csv (có bản sao lưu).
  Sau đó: chạy thu_anh.py, nhập đúng tên, gõ c -> tự chụp lại đúng những ảnh còn thiếu.
  Thu xong chạy lại file này để kiểm tra lần nữa.
  Muốn thu lại 1 ảnh CẢNH BÁO cụ thể: xóa ảnh đó trong mat_mau_112 rồi chạy lại kiểm tra
  (ảnh sẽ thành LỖI "thiếu file" và được đề nghị thu lại).

KẾT QUẢ (dataset/kiem_tra)
  bao_cao.csv      trạng thái từng ảnh + lý do
  <ten>.jpg        lưới ảnh theo lượt & tư thế: viền ĐỎ = lỗi, viền CAM = cảnh báo
  phan_bo_goc.png  biểu đồ góc đầu toàn bộ ảnh (dùng cho báo cáo)
=====================================================================
"""
import os
import sys
import csv
import json
import math
import shutil
import collections
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

# ======================= CẤU HÌNH KIỂM TRA =======================
DO_LAI_MEDIAPIPE = True      # [4] chạy lại MediaPipe trên ảnh gốc
DUNG_SAI_GOC = 3.0           # [4] nới ngưỡng góc khi đo lại (độ)
DUNG_SAI_DIEM = 0.10         # [4] nới ngưỡng điểm biểu cảm khi đo lại
LECH_GOC_CANH_BAO = 8.0      # [4] góc đo lại lệch so với lúc chụp quá mức này -> cảnh báo
LECH_CAN_CHINH = 20.0        # [4] mặt căn chỉnh lại khác ảnh đã lưu quá mức này -> cảnh báo
GOC_LON_YAW, GOC_LON_PITCH = 25, 15   # [4] ảnh góc lớn: đo lại không thấy mặt chỉ là cảnh báo
NGUONG_TRUNG = 3.0           # [5] 2 ảnh khác nhau ít hơn mức này -> gần trùng
TI_LE_NHAN = 0.8             # [6] gần người khác < 80% khoảng cách tới chính mình -> nghi sai nhãn

THU_MUC_CODE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(THU_MUC_CODE, "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Ngưỡng MẶC ĐỊNH (giống thu_anh.py) – chỉ dùng khi không đọc được cau_hinh_thu_thap.json
MAC_DINH = {
    "THANG_TOI_DA": 7, "LECH_PHU_QUAY": 12, "LECH_PHU_NGANG_CUI": 10, "BIEU_CAM_LECH_TOI_DA": 12,
    "SAI_HUONG": 3, "NGUONG_CUOI": 0.50, "CUOI_TOI_DA": 0.40, "NGUONG_HA_MIENG": 0.25,
    "NGUONG_NHAM_MAT": 0.50, "NGUONG_NHAM_MAT_KHI_CUI": 0.85, "MAT_TOI_THIEU_PX": 150,
    "MAT_TOI_DA_TI_LE": 0.85, "LE_KHUNG_TI_LE": 0.02, "DO_SANG_MIN": 55, "DO_SANG_MAX": 200,
    "TUONG_PHAN_MIN": 20, "NGUONG_NET": 25, "NGUONG_KHAC": 4.0, "GIU_KHUNG": 3,
    "XU_LY_CAO": 480, "LUU_ANH_GOC": True,
}
TU_THE_MAC_DINH = [("thang", 15, None), ("trai", 15, [(10, 20), (20, 30), (30, 42)]),
                   ("phai", 15, [(10, 20), (20, 30), (30, 42)]), ("len", 10, [(8, 15), (15, 25)]),
                   ("xuong", 10, [(8, 15), (15, 25)])]
SO_LUOT_MAC_DINH = 2
# ==================================================================

LOI, CANH_BAO, DAT = "LOI", "CANH_BAO", "DAT"
VANG, DO, XANH, CAM, XAM, TRANG = (0, 255, 255), (0, 0, 255), (0, 255, 0), (0, 165, 255), (190, 190, 190), (255, 255, 255)
SO_COT = ["yaw", "pitch", "yaw_tuong_doi", "pitch_tuong_doi", "cuoi", "ha_mieng", "nham_mat",
          "do_net", "do_sang", "tuong_phan", "do_khac", "x1", "y1", "x2", "y2", "rong_anh", "cao_anh"]

# đường dẫn (gán trong cai_duong_dan)
THU_MUC_DATA = THU_MUC_ANH_GOC = THU_MUC_MAU = THU_MUC_XAM = FILE_CSV = FILE_CAU_HINH = THU_MUC_KQ = ""
# cấu hình (gán trong doc_cau_hinh)
C = dict(MAC_DINH)
TU_THE = list(TU_THE_MAC_DINH)
SO_LUOT = SO_LUOT_MAC_DINH


def cai_duong_dan(thu_muc_data):
    global THU_MUC_DATA, THU_MUC_ANH_GOC, THU_MUC_MAU, THU_MUC_XAM, FILE_CSV, FILE_CAU_HINH, THU_MUC_KQ
    THU_MUC_DATA = os.path.abspath(thu_muc_data)
    THU_MUC_ANH_GOC = os.path.join(THU_MUC_DATA, "anh_goc")
    THU_MUC_MAU = os.path.join(THU_MUC_DATA, "mat_mau_112")
    THU_MUC_XAM = os.path.join(THU_MUC_DATA, "mat_xam_100")
    FILE_CSV = os.path.join(THU_MUC_DATA, "metadata.csv")
    FILE_CAU_HINH = os.path.join(THU_MUC_DATA, "cau_hinh_thu_thap.json")
    THU_MUC_KQ = os.path.join(THU_MUC_DATA, "kiem_tra")


def doc_cau_hinh():
    """Đọc ngưỡng đã dùng lúc thu từ cau_hinh_thu_thap.json (nếu có)."""
    global C, TU_THE, SO_LUOT
    C, TU_THE, SO_LUOT = dict(MAC_DINH), list(TU_THE_MAC_DINH), SO_LUOT_MAC_DINH
    nguon = "mặc định trong script (không thấy cau_hinh_thu_thap.json)"
    if os.path.exists(FILE_CAU_HINH):
        try:
            with open(FILE_CAU_HINH, encoding="utf-8") as f:
                ch = json.load(f).get("cau_hinh", {})
            if ch.get("tu_the"):
                TU_THE = [(t[0], int(t[3]), [tuple(k) for k in t[4]] if t[4] else None) for t in ch["tu_the"]]
            if ch.get("luot"):
                SO_LUOT = len(ch["luot"])
            C.update(ch.get("nguong", {}))
            nguon = "cau_hinh_thu_thap.json" + ("" if ch.get("nguong") else " (bản cũ: ngưỡng dùng mặc định)")
        except Exception as e:
            nguon = f"mặc định (lỗi đọc cau_hinh_thu_thap.json: {e})"
    return nguon


def ke_hoach():
    return [(l, b) for l in range(SO_LUOT) for b in range(len(TU_THE))]


def tong_moi_nguoi():
    return SO_LUOT * sum(t[1] for t in TU_THE)


def chi_so_ma():
    return {t[0]: i for i, t in enumerate(TU_THE)}


# =====================================================================
# HÀM XỬ LÝ KHUÔN MẶT (giống hệt thu_anh.py)
# =====================================================================
MO_HINH_3D = np.array([(0.0, 0.0, 0.0), (0.0, 330.0, 65.0), (-225.0, -170.0, 135.0),
                       (225.0, -170.0, 135.0), (-150.0, 150.0, 125.0), (150.0, 150.0, 125.0)], dtype=np.float64)
CHI_SO_PNP = [1, 152, 33, 263, 61, 291]
MAU_ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                            [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def tai_model():
    if not os.path.exists(MODEL):
        print("Đang tải model face_landmarker.task (~4 MB, chỉ lần đầu)...")
        tam = MODEL + ".tmp"
        try:
            urllib.request.urlretrieve(MODEL_URL, tam)
            os.replace(tam, MODEL)
        except Exception as e:
            raise SystemExit(f"Không tải được model ({e}). Tải thủ công:\n  {MODEL_URL}\nrồi đặt vào:\n  {MODEL}")
    with open(MODEL, "rb") as f:
        return f.read()


def tao_landmarker(tin_cay=0.6, so_mat=2):
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=tai_model()),
        running_mode=vision.RunningMode.IMAGE, num_faces=so_mat,
        min_face_detection_confidence=tin_cay, min_face_presence_confidence=tin_cay,
        output_face_blendshapes=True))


def goc_dau(diem, W, H):
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
    huong = -R[:, 2]
    if huong[2] > 0:
        return None
    yaw = math.degrees(math.atan2(huong[0], -huong[2]))
    pitch = math.degrees(math.atan2(-huong[1], math.hypot(huong[0], huong[2])))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def diem_5(diem):
    return np.array([(diem[33] + diem[133]) / 2, (diem[362] + diem[263]) / 2,
                     diem[1], diem[61], diem[291]], dtype=np.float64)


def bien_doi_tuong_dong(src, dst):
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
    M = bien_doi_tuong_dong(np.asarray(d5, np.float64), MAU_ARCFACE_112 * (kich_thuoc / 112.0))
    return cv2.warpAffine(anh, M, (kich_thuoc, kich_thuoc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def thong_tin_tu_diem(diem, blendshapes, W, H):
    """diem: mảng (478, 2) tọa độ pixel trên ảnh gốc."""
    goc = goc_dau(diem, W, H)
    if goc is None:
        return None
    bs = {c.category_name: c.score for c in blendshapes} if blendshapes else {}
    x1, y1 = diem.min(0)
    x2, y2 = diem.max(0)
    return {"diem": diem, "khung": (x1, y1, x2, y2), "d5": diem_5(diem),
            "yaw": goc[0], "pitch": goc[1], "roll": goc[2],
            "cuoi": (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2,
            "ha_mieng": bs.get("jawOpen", 0.0),
            "nham": max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))}


def chat_luong(xam):
    return float(cv2.Laplacian(xam, cv2.CV_64F).var()), float(xam.mean()), float(xam.std())


def chi_tieu_khoang(so_anh, so_khoang):
    return [so_anh // so_khoang + (1 if i < so_anh % so_khoang else 0) for i in range(so_khoang)]


def danh_gia_tu_the(ma, dyaw, dpitch, cuoi, ha_mieng, nham, g=0.0, s=0.0):
    """Luật tư thế (giống lúc thu). g, s: dung sai góc / điểm.
    Trả về (lý do không đạt hoặc "", chỉ số khoảng góc hoặc -1, sai_hướng?)."""
    if nham > (C["NGUONG_NHAM_MAT_KHI_CUI"] if ma == "xuong" else C["NGUONG_NHAM_MAT"]) + s:
        return f"mắt nhắm ({nham:.2f})", -1, False
    if ma != "cuoi" and cuoi > C["CUOI_TOI_DA"] + s:
        return f"đang cười ({cuoi:.2f})", -1, False
    if ma == "thang":
        if abs(dyaw) > C["THANG_TOI_DA"] + g or abs(dpitch) > C["THANG_TOI_DA"] + g:
            return f"không nhìn thẳng (yaw {dyaw:+.1f}°, pitch {dpitch:+.1f}°)", -1, False
        return "", -1, False
    if ma in ("cuoi", "noi"):
        if abs(dyaw) > C["BIEU_CAM_LECH_TOI_DA"] + g or abs(dpitch) > C["BIEU_CAM_LECH_TOI_DA"] + g:
            return f"không nhìn thẳng (yaw {dyaw:+.1f}°, pitch {dpitch:+.1f}°)", -1, False
        if ma == "cuoi" and cuoi < C["NGUONG_CUOI"] - s:
            return f"cười chưa đủ ({cuoi:.2f})", -1, False
        if ma == "noi" and ha_mieng < C["NGUONG_HA_MIENG"] - s:
            return f"há miệng chưa đủ ({ha_mieng:.2f})", -1, False
        return "", -1, False
    khoang = TU_THE[chi_so_ma()[ma]][2]
    ngang = ma in ("trai", "phai")
    v = {"trai": dyaw, "phai": -dyaw, "len": dpitch, "xuong": -dpitch}[ma]
    phu = abs(dpitch) if ngang else abs(dyaw)
    ten = "quay" if ngang else ("ngẩng" if ma == "len" else "cúi")
    if phu > (C["LECH_PHU_QUAY"] if ngang else C["LECH_PHU_NGANG_CUI"]) + g:
        return f"lệch hướng phụ ({'pitch' if ngang else 'yaw'} {phu:.1f}°)", -1, False
    if v < -C["SAI_HUONG"] - g:
        return f"SAI HƯỚNG ({ten} ngược {abs(v):.1f}°)", -1, True
    lo, hi = khoang[0][0], khoang[-1][1]
    if v < lo - g:
        return f"{ten} chưa đủ ({v:.1f}°, cần ≥ {lo}°)", -1, False
    if v > hi + g:
        return f"{ten} quá mức ({v:.1f}°, tối đa {hi}°)", -1, False
    kg = next((i for i, (a, b) in enumerate(khoang) if a <= v < b), len(khoang) - 1 if v >= lo else 0)
    return "", kg, False


# =====================================================================
# TIỆN ÍCH
# =====================================================================
def doc_anh(p, co_mau=True):
    try:
        if not os.path.exists(p):
            return None
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR if co_mau else cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def luu_anh(p, anh, duoi=".png", tham_so=None):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ok, buf = cv2.imencode(duoi, anh, tham_so or [])
    if ok:
        buf.tofile(p)
    return ok


def ve_chu(img, chu, x, y, mau=TRANG, co=0.5, day=1):
    cv2.putText(img, chu, (x, y), cv2.FONT_HERSHEY_SIMPLEX, co, (0, 0, 0), day + 3, cv2.LINE_AA)
    cv2.putText(img, chu, (x, y), cv2.FONT_HERSHEY_SIMPLEX, co, mau, day, cv2.LINE_AA)


def doc_csv():
    if not os.path.exists(FILE_CSV):
        return [], []
    with open(FILE_CSV, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), list(r.fieldnames or [])


def duong_dan(loai, ten, file):
    if loai == "goc":
        return os.path.join(THU_MUC_ANH_GOC, ten, file + ".jpg")
    if loai == "mau":
        return os.path.join(THU_MUC_MAU, ten, file + ".png")
    return os.path.join(THU_MUC_XAM, ten, file + ".png")


def so_thu_tu(d):
    try:
        return int(d.get("so", 0))
    except ValueError:
        return 0


class KetQua:
    def __init__(self):
        self.muc = collections.defaultdict(list)          # file -> [(mức, lý do)]

    def them(self, file, muc, ly_do):
        self.muc[file].append((muc, ly_do))

    def trang_thai(self, file):
        m = {x for x, _ in self.muc.get(file, [])}
        return LOI if LOI in m else (CANH_BAO if CANH_BAO in m else DAT)

    def ly_do(self, file, chi_muc=None):
        return "; ".join(f"[{m}] {l}" for m, l in self.muc.get(file, []) if chi_muc in (None, m))


# =====================================================================
# [1] ĐỦ & CÂN BẰNG
# =====================================================================
def kiem_tra_du(nhom, kq):
    print("\n[1] ĐỦ & CÂN BẰNG")
    thieu_tat_ca = {}
    hop_le_luot = {str(i + 1) for i in range(SO_LUOT)}
    for ten, ds in sorted(nhom.items()):
        thieu, thua = [], []
        for d in ds:
            if d.get("tu_the") not in chi_so_ma() or d.get("luot") not in hop_le_luot:
                kq.them(d["file"], LOI, f"tư thế/lượt không hợp lệ ({d.get('tu_the')}, L{d.get('luot')})")
        for (l, b) in ke_hoach():
            ma, so_anh, khoang = TU_THE[b]
            cua = [d for d in ds if d.get("luot") == str(l + 1) and d.get("tu_the") == ma]
            if len(cua) > so_anh:
                thua.append(f"L{l + 1} {ma} {len(cua)}/{so_anh}")
            if len(cua) < so_anh:
                if khoang:          # chỉ rõ thiếu ở khoảng góc nào (thu_anh.py sẽ chụp bù đúng khoảng đó)
                    ct = chi_tieu_khoang(so_anh, len(khoang))
                    for i, (lo, hi) in enumerate(khoang):
                        n = sum(1 for d in cua if d.get("khoang_goc") == str(i))
                        if n < ct[i]:
                            thieu.append(f"L{l + 1} {ma} {lo}-{hi}°: thiếu {ct[i] - n}")
                else:
                    thieu.append(f"L{l + 1} {ma}: thiếu {so_anh - len(cua)}")
            elif khoang:            # đủ số ảnh nhưng phân bố góc khác cấu hình (chỉ khi đổi cấu hình giữa chừng)
                ct = chi_tieu_khoang(so_anh, len(khoang))
                thuc = [sum(1 for d in cua if d.get("khoang_goc") == str(i)) for i in range(len(khoang))]
                if thuc != ct:
                    print(f"    ! {ten} L{l + 1} {ma}: phân bố khoảng góc {thuc} khác cấu hình {ct}")
        if thieu:
            thieu_tat_ca[ten] = thieu
        tong = len(ds)
        if not thieu and not thua:
            print(f"  {ten:<24} {tong}/{tong_moi_nguoi()}  ĐỦ, cân bằng")
        else:
            print(f"  {ten:<24} {tong}/{tong_moi_nguoi()}  "
                  + (f"THIẾU: {', '.join(thieu)}  " if thieu else "")
                  + (f"THỪA: {', '.join(thua)}" if thua else ""))
    so_luong = {len(ds) for ds in nhom.values()}
    if len(so_luong) > 1:
        print(f"  ! Số ảnh giữa các người KHÔNG bằng nhau: {sorted(so_luong)}")
    return thieu_tat_ca


# =====================================================================
# [2] FILE
# =====================================================================
def kiem_tra_file(dong, kq):
    print("\n[2] FILE")
    anh_xam, anh_mau = {}, {}
    dem_thieu = dem_hong = 0
    for d in dong:
        f, ten = d["file"], d["ten"]
        for loai, kich in (("xam", (100, 100)), ("mau", (112, 112, 3))):
            p = duong_dan(loai, ten, f)
            anh = doc_anh(p, co_mau=(loai == "mau"))
            ten_loai = "xám 100" if loai == "xam" else "màu 112"
            if not os.path.exists(p):
                kq.them(f, LOI, f"thiếu file ảnh {ten_loai}")
                dem_thieu += 1
            elif anh is None or anh.shape != kich:
                kq.them(f, LOI, f"ảnh {ten_loai} hỏng / sai kích thước")
                dem_hong += 1
            elif loai == "xam":
                anh_xam[f] = anh
            else:
                anh_mau[f] = anh
        if C["LUU_ANH_GOC"]:
            p = duong_dan("goc", ten, f)
            if not os.path.exists(p):
                kq.them(f, LOI, "thiếu ảnh gốc")
                dem_thieu += 1
            elif os.path.getsize(p) < 1000:
                kq.them(f, LOI, "ảnh gốc hỏng")
                dem_hong += 1

    co_trong_csv = {(d["ten"], d["file"]) for d in dong}
    file_la = []
    for goc, duoi in ((THU_MUC_ANH_GOC, ".jpg"), (THU_MUC_MAU, ".png"), (THU_MUC_XAM, ".png")):
        if not os.path.isdir(goc):
            continue
        for ten in sorted(os.listdir(goc)):
            tm = os.path.join(goc, ten)
            if not os.path.isdir(tm):
                continue
            for fn in sorted(os.listdir(tm)):
                ten_file, ext = os.path.splitext(fn)
                if ext.lower() == duoi and (ten, ten_file) not in co_trong_csv:
                    file_la.append(os.path.join(tm, fn))
    print(f"  {len(dong)} ảnh trong metadata | thiếu file: {dem_thieu} | file hỏng: {dem_hong} "
          f"| file lạ (không có trong metadata): {len(file_la)}")
    for p in file_la[:10]:
        print(f"    file lạ: {os.path.relpath(p, THU_MUC_DATA)}")
    if len(file_la) > 10:
        print(f"    ... và {len(file_la) - 10} file lạ khác")
    return anh_xam, anh_mau, file_la


# =====================================================================
# [3] THÔNG SỐ LÚC CHỤP
# =====================================================================
def kiem_tra_luc_chup(dong, kq):
    print("\n[3] THÔNG SỐ LÚC CHỤP (metadata.csv)")
    dem_loi = 0
    for d in dong:
        f = d["file"]
        try:
            v = {k: float(d[k]) for k in SO_COT}
            kg_ghi = int(d["khoang_goc"])
        except (ValueError, KeyError, TypeError):
            kq.them(f, LOI, "metadata hỏng (giá trị không đọc được)")
            dem_loi += 1
            continue
        b = chi_so_ma().get(d.get("tu_the"))
        if b is None:
            continue
        truoc = len(kq.muc.get(f, []))
        # dung sai rất nhỏ vì số trong CSV đã làm tròn (góc 2 chữ số, điểm 3 chữ số)
        ly_do, kg, _ = danh_gia_tu_the(d["tu_the"], v["yaw_tuong_doi"], v["pitch_tuong_doi"],
                                       v["cuoi"], v["ha_mieng"], v["nham_mat"], g=0.01, s=0.001)
        khoang = TU_THE[b][2]
        if ly_do:
            kq.them(f, LOI, f"lúc chụp: {ly_do}")
        elif khoang:
            gt = {"trai": v["yaw_tuong_doi"], "phai": -v["yaw_tuong_doi"],
                  "len": v["pitch_tuong_doi"], "xuong": -v["pitch_tuong_doi"]}[d["tu_the"]]
            if not (0 <= kg_ghi < len(khoang)) or not (khoang[kg_ghi][0] - 0.01 <= gt <= khoang[kg_ghi][1] + 0.01):
                kq.them(f, LOI, f"lúc chụp: ghi sai khoảng góc ({gt:.1f}° nhưng ghi khoảng {kg_ghi})")
        if v["do_net"] < C["NGUONG_NET"] - 0.05:
            kq.them(f, LOI, f"lúc chụp: mờ (độ nét {v['do_net']:.0f})")
        if not C["DO_SANG_MIN"] - 0.05 <= v["do_sang"] <= C["DO_SANG_MAX"] + 0.05:
            kq.them(f, LOI, f"lúc chụp: độ sáng {v['do_sang']:.0f}")
        if v["tuong_phan"] < C["TUONG_PHAN_MIN"] - 0.05:
            kq.them(f, LOI, f"lúc chụp: tương phản thấp ({v['tuong_phan']:.0f})")
        if v["do_khac"] < C["NGUONG_KHAC"] - 0.005:
            kq.them(f, LOI, f"lúc chụp: quá giống ảnh trước ({v['do_khac']:.1f})")
        cao_mat = v["y2"] - v["y1"]
        if cao_mat < C["MAT_TOI_THIEU_PX"] - 0.1:
            kq.them(f, LOI, f"lúc chụp: mặt quá nhỏ ({cao_mat:.0f}px)")
        if cao_mat > C["MAT_TOI_DA_TI_LE"] * v["cao_anh"] + 0.1:
            kq.them(f, LOI, "lúc chụp: mặt quá sát camera")
        lx, ly = C["LE_KHUNG_TI_LE"] * v["rong_anh"], C["LE_KHUNG_TI_LE"] * v["cao_anh"]
        if (v["x1"] < lx - 0.1 or v["y1"] < ly - 0.1 or v["x2"] > v["rong_anh"] - lx + 0.1
                or v["y2"] > v["cao_anh"] - ly + 0.1):
            kq.them(f, LOI, "lúc chụp: mặt sát mép khung hình")
        if len(kq.muc.get(f, [])) > truoc:
            dem_loi += 1
    print(f"  {len(dong) - dem_loi} ảnh đạt, {dem_loi} ảnh có lỗi")


# =====================================================================
# [4] ĐO LẠI BẰNG MEDIAPIPE TRÊN ẢNH GỐC
# =====================================================================
def _mp_anh(anh_bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2RGB)))


def phat_hien(lm, lm_phu, raw, khung_ghi):
    """Tìm mặt trên ảnh gốc giống lúc thu (thu nhỏ về XU_LY_CAO). Nếu không thấy,
    thử lại trên vùng quanh mặt đã ghi (ảnh tĩnh đôi khi bỏ sót mặt nghiêng nhiều).
    Trả về danh sách (điểm pixel 478x2, blendshapes)."""
    H, W = raw.shape[:2]
    cao = int(C["XU_LY_CAO"])
    anh_xl = cv2.resize(raw, (max(1, int(W * cao / H)), cao), interpolation=cv2.INTER_AREA) if H > cao else raw
    r = lm.detect(_mp_anh(anh_xl))
    mat = [(np.array([(p.x * W, p.y * H) for p in lms]), r.face_blendshapes[i] if r.face_blendshapes else None)
           for i, lms in enumerate(r.face_landmarks)]
    if mat or khung_ghi is None:
        return mat
    x1, y1, x2, y2 = khung_ghi
    cx, cy, canh = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.8
    X1, Y1 = int(max(0, cx - canh / 2)), int(max(0, cy - canh / 2))
    X2, Y2 = int(min(W, cx + canh / 2)), int(min(H, cy + canh / 2))
    if X2 - X1 < 20 or Y2 - Y1 < 20:
        return []
    r = lm_phu.detect(_mp_anh(raw[Y1:Y2, X1:X2]))
    return [(np.array([(X1 + p.x * (X2 - X1), Y1 + p.y * (Y2 - Y1)) for p in lms]),
             r.face_blendshapes[i] if r.face_blendshapes else None)
            for i, lms in enumerate(r.face_landmarks)]


def do_lai_mot_anh(lm, lm_phu, d, xam_luu, kq):
    f = d["file"]
    raw = doc_anh(duong_dan("goc", d["ten"], f))
    if raw is None:
        return "bo_qua"
    try:
        v = {k: float(d[k]) for k in SO_COT}
    except (ValueError, KeyError, TypeError):
        return "bo_qua"
    H, W = raw.shape[:2]
    if (W, H) != (int(v["rong_anh"]), int(v["cao_anh"])):
        kq.them(f, LOI, f"ảnh gốc sai kích thước ({W}x{H}, metadata ghi {int(v['rong_anh'])}x{int(v['cao_anh'])})")
        return "loi"
    khung_ghi = (v["x1"], v["y1"], v["x2"], v["y2"])
    goc_lon = abs(v["yaw_tuong_doi"]) >= GOC_LON_YAW or abs(v["pitch_tuong_doi"]) >= GOC_LON_PITCH
    cac_mat = phat_hien(lm, lm_phu, raw, khung_ghi)
    infos = [i for i in (thong_tin_tu_diem(diem, bs, W, H) for diem, bs in cac_mat) if i is not None]
    if not infos:
        if goc_lon:
            kq.them(f, CANH_BAO, "đo lại: không tìm thấy mặt (ảnh góc lớn – xem bằng mắt)")
            return "canh_bao"
        kq.them(f, LOI, "đo lại: không tìm thấy mặt")
        return "loi"
    trang_thai = "khop"
    if len(infos) > 1:
        kq.them(f, CANH_BAO, f"đo lại: thấy {len(infos)} khuôn mặt (có người / ảnh khác phía sau?)")
        trang_thai = "canh_bao"
    tam = ((khung_ghi[0] + khung_ghi[2]) / 2, (khung_ghi[1] + khung_ghi[3]) / 2)
    info = min(infos, key=lambda i: math.hypot((i["khung"][0] + i["khung"][2]) / 2 - tam[0],
                                                (i["khung"][1] + i["khung"][3]) / 2 - tam[1]))
    moc_yaw, moc_pitch = v["yaw"] - v["yaw_tuong_doi"], v["pitch"] - v["pitch_tuong_doi"]
    dyaw, dpitch = info["yaw"] - moc_yaw, info["pitch"] - moc_pitch
    ly_do, _, sai_huong = danh_gia_tu_the(d["tu_the"], dyaw, dpitch, info["cuoi"], info["ha_mieng"],
                                          info["nham"], g=DUNG_SAI_GOC, s=DUNG_SAI_DIEM)
    if ly_do:
        kq.them(f, LOI if sai_huong else CANH_BAO, f"đo lại: {ly_do}")
        trang_thai = "loi" if sai_huong else ("canh_bao" if trang_thai != "loi" else trang_thai)
    lech = max(abs(dyaw - v["yaw_tuong_doi"]), abs(dpitch - v["pitch_tuong_doi"]))
    if lech > LECH_GOC_CANH_BAO:
        kq.them(f, CANH_BAO, f"đo lại: góc lệch {lech:.0f}° so với lúc chụp")
        if trang_thai == "khop":
            trang_thai = "canh_bao"
    if xam_luu is not None:
        xam_moi = cv2.cvtColor(can_chinh_mat(raw, info["d5"], 100), cv2.COLOR_BGR2GRAY)
        khac = float(np.mean(cv2.absdiff(xam_moi, xam_luu)))
        if khac > LECH_CAN_CHINH:
            kq.them(f, CANH_BAO, f"đo lại: mặt cắt/căn chỉnh không khớp ảnh đã lưu (khác {khac:.0f})")
            if trang_thai == "khop":
                trang_thai = "canh_bao"
    return trang_thai


def kiem_tra_do_lai(dong, kq, anh_xam):
    print("\n[4] ĐO LẠI BẰNG MEDIAPIPE TRÊN ẢNH GỐC")
    if not DO_LAI_MEDIAPIPE or not C["LUU_ANH_GOC"]:
        print("  (bỏ qua)")
        return
    lm, lm_phu = tao_landmarker(0.6, 2), tao_landmarker(0.3, 1)
    dem = collections.Counter()
    try:
        for i, d in enumerate(dong, 1):
            dem[do_lai_mot_anh(lm, lm_phu, d, anh_xam.get(d["file"]), kq)] += 1
            if i % 20 == 0 or i == len(dong):
                print(f"\r  Đã đo lại {i}/{len(dong)} ảnh...", end="", flush=True)
    finally:
        lm.close()
        lm_phu.close()
    print(f"\n  khớp: {dem['khop']} | cảnh báo: {dem['canh_bao']} | lỗi: {dem['loi']} "
          f"| bỏ qua (thiếu/hỏng ảnh gốc): {dem['bo_qua']}")


# =====================================================================
# [5] CHẤT LƯỢNG ẢNH XÁM & ẢNH GẦN TRÙNG
# =====================================================================
def kiem_tra_chat_luong(nhom, kq, anh_xam):
    print("\n[5] CHẤT LƯỢNG ẢNH XÁM & ẢNH GẦN TRÙNG")
    dem_loi = dem_trung = 0
    for ten, ds in sorted(nhom.items()):
        ds_sx = [d for d in sorted(ds, key=so_thu_tu) if d["file"] in anh_xam]
        for d in ds_sx:
            f = d["file"]
            net, sang, tp = chat_luong(anh_xam[f])
            loi = []
            if net < C["NGUONG_NET"]:
                loi.append(f"ảnh xám mờ ({net:.0f})")
            if not C["DO_SANG_MIN"] <= sang <= C["DO_SANG_MAX"]:
                loi.append(f"ảnh xám độ sáng {sang:.0f}")
            if tp < C["TUONG_PHAN_MIN"]:
                loi.append(f"ảnh xám tương phản thấp ({tp:.0f})")
            try:     # ảnh xám lưu bằng PNG (không nén mất) -> phải khớp số liệu lúc chụp
                if (abs(net - float(d["do_net"])) > max(0.5, 0.01 * net) or abs(sang - float(d["do_sang"])) > 0.5
                        or abs(tp - float(d["tuong_phan"])) > 0.5):
                    loi.append("ảnh xám không khớp thông số lúc chụp (file đã bị thay đổi?)")
            except (ValueError, KeyError, TypeError):
                pass
            for l in loi:
                kq.them(f, LOI, l)
            dem_loi += bool(loi)
        files = [d["file"] for d in ds_sx]
        if len(files) < 2:
            continue
        X = np.stack([anh_xam[f].astype(np.float32).ravel() for f in files])
        for i in range(len(files) - 1):
            khac = np.abs(X[i + 1:] - X[i]).mean(1)
            for j in np.where(khac < NGUONG_TRUNG)[0]:
                kq.them(files[i + 1 + j], CANH_BAO, f"gần trùng với {files[i]} (khác {khac[j]:.1f})")
                dem_trung += 1
    print(f"  ảnh xám lỗi: {dem_loi} | cặp ảnh gần trùng: {dem_trung}")


# =====================================================================
# [6] KIỂM TRA NHÃN
# =====================================================================
def kiem_tra_nhan(dong, kq, anh_xam):
    print("\n[6] KIỂM TRA NHÃN (PCA + láng giềng gần nhất, không gồm chính nó)")
    ds = [(d["file"], d["ten"]) for d in dong if d["file"] in anh_xam]
    if len({t for _, t in ds}) < 2:
        print("  (bỏ qua – cần ít nhất 2 người)")
        return None
    X = np.stack([cv2.equalizeHist(anh_xam[f]).astype(np.float32).ravel() for f, _ in ds])
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    Xc = X - X.mean(0)
    k = min(100, len(ds) - 1)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:k].T
    bp = (Z ** 2).sum(1)
    D = np.sqrt(np.maximum(bp[:, None] + bp[None, :] - 2 * Z @ Z.T, 0))
    np.fill_diagonal(D, np.inf)
    nhan = np.array([t for _, t in ds])
    cung = nhan[:, None] == nhan[None, :]
    do_chinh_xac = float((nhan[D.argmin(1)] == nhan).mean())
    d_cung = np.where(cung, D, np.inf).min(1)
    D_khac = np.where(~cung, D, np.inf)
    d_khac, j_khac = D_khac.min(1), D_khac.argmin(1)
    nghi = np.where(d_khac < TI_LE_NHAN * d_cung)[0]
    for i in nghi:
        kq.them(ds[i][0], CANH_BAO, f"giống ảnh {ds[j_khac[i]][0]} của {nhan[j_khac[i]]} hơn hẳn "
                                    f"ảnh của chính mình (xem lại nhãn)")
    print(f"  Độ phân biệt sơ bộ (chưa chia train/test): {do_chinh_xac * 100:.1f}% | ảnh nghi sai nhãn: {len(nghi)}")
    return do_chinh_xac


# =====================================================================
# BÁO CÁO
# =====================================================================
def ghi_bao_cao_csv(dong, kq, file_la):
    p = os.path.join(THU_MUC_KQ, "bao_cao.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "ten", "luot", "tu_the", "trang_thai", "ly_do"])
        for d in dong:
            w.writerow([d["file"], d["ten"], d.get("luot"), d.get("tu_the"), kq.trang_thai(d["file"]),
                        kq.ly_do(d["file"])])
        for fl in file_la:
            w.writerow([os.path.relpath(fl, THU_MUC_DATA), "", "", "", "FILE_LA",
                        "có file nhưng không có trong metadata.csv"])
    return p


def ve_luoi(ten, ds, kq, anh_mau):
    o, nhan_w = 100, 190
    kh = ke_hoach()
    so_cot = max(t[1] for t in TU_THE)
    luoi = np.full((70 + len(kh) * o + 10, nhan_w + so_cot * o + 10, 3), 28, np.uint8)
    dem = collections.Counter(kq.trang_thai(d["file"]) for d in ds)
    ve_chu(luoi, f"{ten}: {len(ds)}/{tong_moi_nguoi()} anh  |  DAT {dem[DAT]}  CANH BAO {dem[CANH_BAO]}  "
                 f"LOI {dem[LOI]}", 10, 28, TRANG, 0.7, 2)
    ve_chu(luoi, "Vien CAM = canh bao (xem bang mat) | Vien DO = loi (nen thu lai) | chi tiet: kiem_tra/bao_cao.csv",
           10, 56, XAM, 0.5)
    for r, (l, b) in enumerate(kh):
        y = 70 + r * o
        ma, so_anh, _ = TU_THE[b]
        cua = sorted((d for d in ds if d.get("luot") == str(l + 1) and d.get("tu_the") == ma), key=so_thu_tu)
        ve_chu(luoi, f"L{l + 1} {ma}", 10, y + 45, VANG, 0.6, 2)
        ve_chu(luoi, f"{len(cua)}/{so_anh}", 10, y + 72, XANH if len(cua) == so_anh else DO, 0.55)
        for c, d in enumerate(cua[:so_cot]):
            x = nhan_w + c * o
            anh = anh_mau.get(d["file"])
            if anh is None:
                anh = np.zeros((112, 112, 3), np.uint8)
                ve_chu(anh, "THIEU", 20, 60, DO, 0.6, 2)
            luoi[y + 5:y + o - 5, x + 5:x + o - 5] = cv2.resize(anh, (o - 10, o - 10))
            st = kq.trang_thai(d["file"])
            if st != DAT:
                cv2.rectangle(luoi, (x + 2, y + 2), (x + o - 3, y + o - 3), DO if st == LOI else CAM, 4)
            ve_chu(luoi, d.get("so", ""), x + 7, y + o - 9, TRANG, 0.38)
    p = os.path.join(THU_MUC_KQ, f"{ten}.jpg")
    luu_anh(p, luoi, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


def ve_phan_bo_goc(dong):
    kich, le = 520, 60
    anh = np.full((kich + 2 * le, kich + 2 * le + 170, 3), 255, np.uint8)
    sc = (kich / 2) / 50.0
    cx, cy = le + kich // 2, le + kich // 2
    mau = {"thang": (90, 90, 90), "trai": (200, 80, 0), "phai": (0, 140, 255), "len": (0, 160, 0),
           "xuong": (180, 0, 180)}
    for g in range(-50, 51, 10):
        c = (225, 225, 225) if g else (150, 150, 150)
        cv2.line(anh, (int(cx + g * sc), le), (int(cx + g * sc), le + kich), c, 1)
        cv2.line(anh, (le, int(cy - g * sc)), (le + kich, int(cy - g * sc)), c, 1)
        cv2.putText(anh, str(g), (int(cx + g * sc) - 10, le + kich + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(anh, str(g), (le - 32, int(cy - g * sc) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for d in dong:
        try:
            yw, pt = float(d["yaw_tuong_doi"]), float(d["pitch_tuong_doi"])
        except (ValueError, KeyError, TypeError):
            continue
        p = (int(cx + max(-50, min(50, yw)) * sc), int(cy - max(-50, min(50, pt)) * sc))
        cv2.circle(anh, p, 3, mau.get(d.get("tu_the"), (0, 0, 0)), -1)
    cv2.putText(anh, "Yaw (do, duong = quay TRAI)", (le + 120, le + kich + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(anh, "Pitch (do, duong = ngang LEN)", (5, le - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for i, (ma, c) in enumerate(mau.items()):
        cv2.circle(anh, (le + kich + 30, le + 20 + 26 * i), 6, c, -1)
        cv2.putText(anh, ma, (le + kich + 45, le + 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    p = os.path.join(THU_MUC_KQ, "phan_bo_goc.png")
    luu_anh(p, anh, ".png")
    return p


# =====================================================================
# CHUYỂN ẢNH LỖI SANG loai_bo ĐỂ THU LẠI
# =====================================================================
def _chuyen(nguon, dich):
    os.makedirs(os.path.dirname(dich), exist_ok=True)
    goc, ext = os.path.splitext(dich)
    i = 1
    while os.path.exists(dich):
        dich = f"{goc}_{i}{ext}"
        i += 1
    shutil.move(nguon, dich)


def chuyen_loai_bo(dong, cot, bo, file_la, kq):
    """bo: tập tên file cần loại. Chuyển ảnh + bỏ dòng khỏi metadata (có sao lưu)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tm = os.path.join(THU_MUC_DATA, "loai_bo", stamp)
    os.makedirs(tm, exist_ok=True)
    shutil.copy2(FILE_CSV, os.path.join(tm, "metadata_truoc_khi_loai.csv"))
    dong_bo = [d for d in dong if d["file"] in bo]
    for d in dong_bo:
        for loai, thu_muc in (("goc", "anh_goc"), ("mau", "mat_mau_112"), ("xam", "mat_xam_100")):
            p = duong_dan(loai, d["ten"], d["file"])
            if os.path.exists(p):
                _chuyen(p, os.path.join(tm, thu_muc, d["ten"], os.path.basename(p)))
    for p in file_la:
        if os.path.exists(p):
            _chuyen(p, os.path.join(tm, "file_la", os.path.relpath(p, THU_MUC_DATA)))
    # ghi lại metadata KHÔNG có các dòng bị loại (ghi file tạm rồi thay thế -> an toàn)
    con_lai = [d for d in dong if d["file"] not in bo]
    tam = FILE_CSV + ".tmp"
    with open(tam, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cot, extrasaction="ignore")
        w.writeheader()
        w.writerows(con_lai)
    os.replace(tam, FILE_CSV)
    with open(os.path.join(tm, "danh_sach_loai_bo.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "ten", "luot", "tu_the", "trang_thai", "ly_do"])
        for d in dong_bo:
            w.writerow([d["file"], d["ten"], d.get("luot"), d.get("tu_the"), kq.trang_thai(d["file"]),
                        kq.ly_do(d["file"])])
        for p in file_la:
            w.writerow([os.path.relpath(p, THU_MUC_DATA), "", "", "", "FILE_LA", "không có trong metadata"])
    return tm, len(dong_bo)


def tom_tat_theo_nguoi(dong, files):
    """{ten: ['L1 trai (2)', ...]} cho các file trong danh sách."""
    dem = collections.defaultdict(collections.Counter)
    for d in dong:
        if d["file"] in files:
            dem[d["ten"]][f"L{d.get('luot')} {d.get('tu_the')}"] += 1
    return {ten: [f"{k} ({n})" for k, n in sorted(c.items())] for ten, c in sorted(dem.items())}


# =====================================================================
def main():
    cai_duong_dan(sys.argv[1] if len(sys.argv) > 1 else os.path.join(THU_MUC_CODE, "dataset"))
    print("=" * 70)
    print(f" KIỂM TRA DỮ LIỆU: {THU_MUC_DATA}")
    print("=" * 70)
    if not os.path.isdir(THU_MUC_DATA):
        raise SystemExit("Không thấy thư mục dataset. Đặt file này cạnh thư mục dataset "
                         "hoặc chạy: python kiem_tra.py <đường_dẫn_dataset>")
    print(f"Ngưỡng kiểm tra lấy từ: {doc_cau_hinh()}")
    dong, cot = doc_csv()
    # dòng hỏng (bị cắt ngang khi tắt máy đột ngột): không có tên file / tên người -> bỏ ra
    dong_hong = [d for d in dong if not d.get("file") or not d.get("ten")]
    dong = [d for d in dong if d.get("file") and d.get("ten")]
    if dong_hong:
        print(f"! Có {len(dong_hong)} dòng metadata bị hỏng/cắt ngang (sẽ bị bỏ nếu chọn chuyển ảnh lỗi).")
    if not dong:
        raise SystemExit("Chưa có metadata.csv hoặc chưa có ảnh nào.")
    os.makedirs(THU_MUC_KQ, exist_ok=True)

    kq = KetQua()
    for f, n in collections.Counter(d.get("file") for d in dong).items():
        if n > 1:
            kq.them(f, LOI, f"tên file lặp {n} lần trong metadata")
    nhom = collections.defaultdict(list)
    for d in dong:
        nhom[d.get("ten", "")].append(d)

    thieu = kiem_tra_du(nhom, kq)
    anh_xam, anh_mau, file_la = kiem_tra_file(dong, kq)
    kiem_tra_luc_chup(dong, kq)
    kiem_tra_do_lai(dong, kq, anh_xam)
    kiem_tra_chat_luong(nhom, kq, anh_xam)
    kiem_tra_nhan(dong, kq, anh_xam)

    p_csv = ghi_bao_cao_csv(dong, kq, file_la)
    luoi = [ve_luoi(ten, ds, kq, anh_mau) for ten, ds in sorted(nhom.items())]
    p_goc = ve_phan_bo_goc(dong)

    loi = {d["file"] for d in dong if kq.trang_thai(d["file"]) == LOI}
    canh_bao = {d["file"] for d in dong if kq.trang_thai(d["file"]) == CANH_BAO}
    print("\n" + "=" * 70)
    print(f" KẾT QUẢ: {len(dong)} ảnh | ĐẠT {len(dong) - len(loi) - len(canh_bao)} | "
          f"CẢNH BÁO {len(canh_bao)} | LỖI {len(loi)} | file lạ {len(file_la)}")
    print("=" * 70)
    for d in [d for d in dong if d["file"] in loi][:20]:
        print(f"  LỖI      {d['file']}: {kq.ly_do(d['file'], LOI)}")
    if len(loi) > 20:
        print(f"  ... và {len(loi) - 20} ảnh lỗi khác (xem bao_cao.csv)")
    for d in [d for d in dong if d["file"] in canh_bao][:10]:
        print(f"  CẢNH BÁO {d['file']}: {kq.ly_do(d['file'], CANH_BAO)}")
    if len(canh_bao) > 10:
        print(f"  ... và {len(canh_bao) - 10} ảnh cảnh báo khác (xem bao_cao.csv)")
    print(f"\nBáo cáo từng ảnh : {p_csv}")
    for p in luoi:
        print(f"Lưới ảnh         : {p}")
    print(f"Biểu đồ góc đầu  : {p_goc}")

    # ------------------- kết luận + thu lại -------------------
    if dong_hong and not loi and not file_la:
        try:
            if input(f"\nXóa {len(dong_hong)} dòng metadata hỏng (có sao lưu)? (c = có / Enter = không): ").strip().lower() == "c":
                tm, _ = chuyen_loai_bo(dong, cot, set(), [], kq)
                print(f"Đã bỏ dòng hỏng. Bản metadata cũ sao lưu trong: {tm}")
        except EOFError:
            pass
    if not loi and not file_la and not thieu:
        print("\n=> DỮ LIỆU ĐẠT, không cần thu lại."
              + (f" Có {len(canh_bao)} ảnh CẢNH BÁO: mở lưới ảnh xem bằng mắt (viền cam)." if canh_bao else ""))
        return

    print("\n=> CẦN THU LẠI:")
    for ten, ds in tom_tat_theo_nguoi(dong, loi).items():
        print(f"  {ten}: {len(loi & {d['file'] for d in nhom[ten]})} ảnh lỗi – " + ", ".join(ds))
    for ten, ds in thieu.items():
        print(f"  {ten}: đang thiếu – " + ", ".join(ds))
    if file_la:
        print(f"  {len(file_la)} file lạ (không có trong metadata, thường do chương trình bị tắt ngang)")

    if loi or file_la:
        print("\nChọn ảnh chuyển sang dataset/loai_bo để THU LẠI (ảnh được CHUYỂN, không bị xóa):")
        print(f"  1 = chỉ ảnh LỖI ({len(loi)} ảnh) + file lạ   <- nên chọn")
        if canh_bao:
            print(f"  2 = ảnh LỖI + CẢNH BÁO ({len(loi) + len(canh_bao)} ảnh) + file lạ")
        print("  Enter = không chuyển gì")
        try:
            chon = input("Lựa chọn: ").strip()
        except EOFError:
            chon = ""
        if chon not in ("1", "2") or (chon == "2" and not canh_bao):
            print("Không chuyển gì. Có thể chạy lại kiểm tra bất cứ lúc nào.")
            return
        bo = loi | (canh_bao if chon == "2" else set())
        tm, n = chuyen_loai_bo(dong, cot, bo, file_la, kq)
        ten_can = sorted({d["ten"] for d in dong if d["file"] in bo} | set(thieu))
        print(f"\nĐã chuyển {n} ảnh" + (f" + {len(file_la)} file lạ" if file_la else "")
              + f" sang: {tm}")
        print("metadata.csv đã cập nhật (bản cũ sao lưu trong thư mục trên).")
    else:
        ten_can = sorted(thieu)

    print("\nBƯỚC TIẾP THEO:")
    print("  1. Chạy:  python thu_anh.py")
    print(f"  2. Nhập đúng tên: {', '.join(ten_can)}  rồi gõ c -> chương trình chụp lại ĐÚNG phần còn thiếu")
    print("  3. Thu xong chạy lại:  python kiem_tra.py")


if __name__ == "__main__":
    main()

"""
=====================================================================
 chung.py – CẤU HÌNH VÀ HÀM DÙNG CHUNG CHO CẢ 6 BƯỚC (cả nhóm dùng)
=====================================================================
 Vì sao cần file này:
   - Mọi bước đọc/ghi file ở CÙNG một chỗ (đường dẫn khai báo 1 lần ở đây).
   - Ảnh được tiền xử lý GIỐNG HỆT nhau ở bước 1 (huấn luyện) và bước 6
     (camera thật) – bài giảng: dữ liệu mới phải xử lý như tập train.
   - Hình vẽ cùng một kiểu, lưu vào thư mục ket_qua/.

 Nguồn ảnh (cột "nguon" trong danh_sach.csv):
   tu_thu   : ảnh nhóm tự thu bằng thu_anh.py      dataset/mat_xam_100/<Tên>/

 Dữ liệu đi qua các bước:
   Bước 1 -> dataset/du_lieu_anh.npz + dataset/danh_sach.csv
   Bước 2 -> dataset/danh_sach.csv (thêm cột "tap": train / test / nguoi_la)
   Bước 3 -> mo_hinh/hoi_quy_logistic.joblib + ket_qua/3_ket_qua_cv.csv
   Bước 4 -> ket_qua/du_doan.npz + ket_qua/4_du_doan_tap_test.csv
   Bước 5 -> mo_hinh/mo_hinh_tot_nhat.joblib + ket_qua/xem_ket_qua.html
   Bước 6 -> nhận diện qua camera, ghi diem_danh.csv
=====================================================================
"""
import os
import warnings

warnings.filterwarnings("ignore")      # ẩn cảnh báo phiên bản thư viện cho dễ đọc kết quả

import numpy as np                     # numpy: mảng và phép tính ma trận
import pandas as pd                    # pandas: bảng dữ liệu, đọc/ghi file CSV
import sklearn                         # nạp TRƯỚC OpenCV (tránh lỗi libgomp trên Jetson)
import cv2                             # OpenCV: đọc/ghi và xử lý ảnh
import matplotlib
matplotlib.use("Agg")                  # vẽ hình ra file, không cần màn hình (chạy được qua SSH)
import matplotlib.pyplot as plt

# ============================ ĐƯỜNG DẪN ============================
THU_MUC = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(THU_MUC, "dataset")
THU_MUC_TU_THU = os.path.join(DATASET, "mat_xam_100")         # ảnh mặt tự thu (thu_anh.py)
KET_QUA = os.path.join(THU_MUC, "ket_qua")                    # hình và bảng kết quả
MO_HINH = os.path.join(THU_MUC, "mo_hinh")                    # mô hình đã huấn luyện

FILE_ANH = os.path.join(DATASET, "du_lieu_anh.npz")                  # bước 1 -> 3, 4, 5
FILE_DANH_SACH = os.path.join(DATASET, "danh_sach.csv")              # bước 1, 2 -> 3, 4, 5
FILE_HOI_QUY_LOGISTIC = os.path.join(MO_HINH, "hoi_quy_logistic.joblib")       # bước 3 -> 4, 5
FILE_KET_QUA_CV = os.path.join(KET_QUA, "3_ket_qua_cv.csv")          # bước 3 -> 5
FILE_DU_DOAN = os.path.join(KET_QUA, "du_doan.npz")                  # bước 4 -> 5
FILE_MO_HINH_TOT = os.path.join(MO_HINH, "mo_hinh_tot_nhat.joblib")  # bước 5 -> 6

# ============================ THAM SỐ CHUNG ============================
KICH_THUOC = 100          # ảnh mặt xám 100x100 -> vector 10.000 chiều
LAM_MO = 1.0              # độ mờ (sigma) của bộ lọc Gaussian
CLAHE_CLIP = 2.0          # giới hạn tương phản của CLAHE
CLAHE_LUOI = 8            # CLAHE chia ảnh thành lưới 8 x 8 ô
RANDOM_STATE = 42         # cố định yếu tố ngẫu nhiên -> chạy lại ra cùng kết quả
TEN_TU_THE = {"thang": "Nhìn thẳng", "trai": "Quay trái", "phai": "Quay phải",
              "len": "Ngẩng lên", "xuong": "Cúi xuống"}
# Nguồn ảnh: tên hiển thị + màu dùng chung cho mọi biểu đồ
NGUON = {"tu_thu": ("Tự thu (thu_anh.py)", "darkorange")}
TRONG_LOP = ("tu_thu",)   # người trong lớp: được hiện tên + điểm danh ở bước 6

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_LUOI, CLAHE_LUOI))
plt.rcParams["font.family"] = "DejaVu Sans"       # font có đủ dấu tiếng Việt


# ============================ TIỀN XỬ LÝ ẢNH ============================
def tien_xu_ly(anh_xam):
    """Tiền xử lý 1 ảnh mặt xám 100x100 đã căn chỉnh. Dùng ở bước 1 VÀ bước 6.
    1) cv2.GaussianBlur : làm mờ nhẹ (sigma = 1) để khử nhiễu hạt của camera
    2) CLAHE            : cân bằng sáng theo từng ô 8x8 -> nửa mặt tối được làm rõ"""
    anh = cv2.GaussianBlur(anh_xam, (0, 0), LAM_MO)
    return _clahe.apply(anh)


def mat_xam_chuan(anh, d5):
    from thu_anh import can_chinh_mat
    mat = can_chinh_mat(anh, d5, KICH_THUOC)
    return tien_xu_ly(cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY))


def vector_hoa(cac_anh):
    """N ảnh 100x100 -> ma trận N x 10000: mỗi hàng là 1 ảnh
    (giống hàm vectorize_img trong ví dụ AR Face của bài giảng)."""
    return cac_anh.reshape(len(cac_anh), -1).astype(np.float32)


# ============================ ĐỌC / GHI ============================
def doc_du_lieu():
    """Đọc kết quả bước 1 (+ bước 2): X là ma trận N x 10000, df là bảng thông tin từng ảnh."""
    if not (os.path.exists(FILE_ANH) and os.path.exists(FILE_DANH_SACH)):
        raise SystemExit("Chưa có dữ liệu. Chạy buoc1_chuan_bi_du_lieu.py trước.")
    anh = np.load(FILE_ANH)["anh"]
    df = pd.read_csv(FILE_DANH_SACH, dtype={"file": str, "ten": str, "tu_the": str}).fillna("")
    if len(df) != len(anh):
        raise SystemExit("dataset/danh_sach.csv không khớp du_lieu_anh.npz. Chạy lại từ bước 1.")
    if not df["nguon"].isin(NGUON).all():
        raise SystemExit("Dữ liệu có nguồn không còn được sử dụng. Chạy lại từ bước 1.")
    return vector_hoa(anh), df


def doc_anh_xam(duong_dan):
    """Đọc ảnh xám, chạy được cả với đường dẫn có dấu tiếng Việt (cv2.imread thì không)."""
    try:
        du_lieu = np.fromfile(duong_dan, dtype=np.uint8)
        return cv2.imdecode(du_lieu, cv2.IMREAD_GRAYSCALE) if du_lieu.size else None
    except (OSError, cv2.error):
        return None


def luu_png(duong_dan, anh):
    os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
    ok, buf = cv2.imencode(".png", anh)
    if ok:
        buf.tofile(duong_dan)


# ============================ VẼ HÌNH ============================
def luu_hinh(ten):
    """Lưu hình matplotlib đang vẽ vào ket_qua/<ten>."""
    os.makedirs(KET_QUA, exist_ok=True)
    p = os.path.join(KET_QUA, ten)
    plt.tight_layout()
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close("all")
    print("    -> " + os.path.relpath(p, THU_MUC))


def rut_gon(ten):
    """'Gerhard_Schroeder' -> 'G. Schroeder' (tên ngắn để ghi lên hình)."""
    ten = str(ten)
    if len(ten) <= 15:
        return ten.replace("_", " ")
    p = ten.split("_")
    return " ".join([x[0] + "." for x in p[:-1]] + [p[-1]])


def ve_luoi_anh(X, dong1, dong2, dung, ten_file, tieu_de, cot=8):
    """Vẽ lưới ảnh mặt. Mỗi ảnh có 2 dòng chữ; dung[i] = True -> khung xanh, False -> khung đỏ."""
    hang = int(np.ceil(len(X) / cot))
    fig, ax = plt.subplots(hang, cot, figsize=(1.75 * cot, 2.2 * hang), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, x, d1, d2, ok in zip(ax.ravel(), X, dong1, dong2, dung):
        a.imshow(np.asarray(x).reshape(KICH_THUOC, KICH_THUOC), cmap="gray", vmin=0, vmax=255)
        a.add_patch(plt.Rectangle((0, 0), KICH_THUOC - 1, KICH_THUOC - 1, fill=False, lw=3,
                                  ec="limegreen" if ok else "red"))
        a.set_title(f"{d1}\n{d2}", fontsize=7.5, color="green" if ok else "red")
    plt.suptitle(tieu_de)
    luu_hinh(ten_file)

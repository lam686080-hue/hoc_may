"""
=====================================================================
 BƯỚC 1 – CHUẨN BỊ DỮ LIỆU
 Phụ trách: Thành viên 1 – Bùi Trung Hiếu
=====================================================================
 Làm gì   : Gom ảnh mặt từ ảnh tự thu và LFW, đưa về CÙNG một dạng (căn chỉnh 5 điểm, xám 100x100,
            lọc Gaussian + CLAHE), thống kê + vẽ biểu đồ, lưu lại cho các bước sau.
 Đầu vào  : dataset/mat_xam_100/<Tên>/*.png    ảnh nhóm tự thu bằng thu_anh.py (đã căn chỉnh)
            Bộ LFW (tải nếu chưa có, ~230 MB)
 Đầu ra   : dataset/du_lieu_anh.npz   tất cả ảnh đã tiền xử lý (N x 100 x 100)
            dataset/danh_sach.csv     mỗi dòng 1 ảnh: file, tên, nguồn, lượt, tư thế, vai trò
            ket_qua/1_*.png           biểu đồ cho báo cáo
 Các bước : 1.1 Đọc ảnh tự thu và gán vai trò
            1.2 Chuẩn bị ảnh LFW (tải + căn chỉnh giống hệt ảnh tự thu)
            1.3 Tiền xử lý: lọc Gaussian + cân bằng sáng CLAHE
            1.4 Thống kê và trực quan hóa
            1.5 Lưu dữ liệu
 Chạy     : python buoc1_chuan_bi_du_lieu.py
            python buoc1_chuan_bi_du_lieu.py --chuan-bi   (chỉ chuẩn bị bộ LFW đã căn chỉnh)
=====================================================================
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import chung as C
from chung import plt, cv2

# ======================= CẤU HÌNH BƯỚC 1 =======================
LFW_SO_ANH_TOI_THIEU = 70     # lấy người có >= 70 ảnh trong LFW -> 7 người
LFW_TOI_DA_MOI_NGUOI = 150    # mỗi người LFW lấy tối đa 150 ảnh
LFW_SO_NGUOI_LA = 30          # 30 người LFW khác làm "người lạ" (chỉ để test)
LFW_ANH_MOI_NGUOI_LA = 5      # mỗi người lạ lấy 5 ảnh
SO_ANH_TOI_THIEU = 5          # người trong lớp có < 5 ảnh: chưa đủ để học -> làm người lạ (test)
CHI_LAM_NGUOI_LA = []         # tên người chỉ dùng làm người lạ, vd ["Tran_Van_A"]


# ======================= 1.1 ĐỌC ẢNH TỰ THU =======================
def doc_anh_tu_thu():
    """Tên ảnh do thu_anh.py đặt: <Tên>_<số>_L<lượt>_<tư thế>.png -> biết người, lượt, tư thế.
    Ảnh bị kiem_tra.py đánh dấu LOI thì bỏ qua."""
    thu_muc_anh = C.THU_MUC_TU_THU
    print(f"  Đọc ảnh tự thu tại: {thu_muc_anh}")
    p_loi = os.path.join(C.DATASET, "kiem_tra", "bao_cao.csv")
    loi = set()
    if os.path.exists(p_loi):
        bc = pd.read_csv(p_loi, encoding="utf-8-sig", dtype=str)
        loi = set(bc.loc[bc["trang_thai"] == "LOI", "file"])
    dong, sai_ten = [], 0
    for p in sorted(glob.glob(os.path.join(thu_muc_anh, "*", "*.png"))):
        goc = os.path.splitext(os.path.basename(p))[0]
        phan = goc.rsplit("_", 3)                       # tách 3 dấu "_" cuối: [tên, số, L1, tư thế]
        if len(phan) < 4 or phan[2][:1] != "L" or not phan[2][1:].isdigit():
            sai_ten += 1                                # không phải ảnh do thu_anh.py chụp
        elif goc not in loi:
            dong.append({"duong_dan": p, "file": goc, "ten": os.path.basename(os.path.dirname(p)),
                         "nguon": "tu_thu", "luot": int(phan[2][1:]), "tu_the": phan[3],
                         "vai_tro": "biet"})
    print(f"  Ảnh tự thu: {len(dong)} ảnh ({len(loi)} ảnh LOI bị bỏ qua)")
    if sai_ten:
        print(f"  ! Bỏ qua {sai_ten} file không đúng tên của thu_anh.py")
    return dong


def gan_vai_tro(dong):
    """Người trong lớp có < SO_ANH_TOI_THIEU ảnh (tự thu) chưa đủ để học
    -> chỉ dùng làm NGƯỜI LẠ (test). Người trong CHI_LAM_NGUOI_LA cũng vậy."""
    dem = pd.Series([d["ten"] for d in dong], dtype=str).value_counts()
    it_anh = sorted(t for t, n in dem.items() if n < SO_ANH_TOI_THIEU)
    for d in dong:
        if d["ten"] in it_anh or d["ten"] in CHI_LAM_NGUOI_LA:
            d["vai_tro"] = "nguoi_la"
    if it_anh:
        print(f"  ! Chưa đủ {SO_ANH_TOI_THIEU} ảnh -> tạm làm người lạ: "
              + ", ".join(f"{t} ({dem[t]})" for t in it_anh))
    return dong


# ======================= 1.2 CHUẨN BỊ ẢNH LFW =======================
def tao_bo_tim_mat(so_mat):
    """MediaPipe Face Landmarker chế độ IMAGE (ảnh tĩnh): tìm mặt + 478 điểm mốc."""
    import thu_anh as T
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=T.tai_model()),
        running_mode=vision.RunningMode.IMAGE, num_faces=so_mat,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5))


def tim_mat(lm, bgr):
    """Trả về danh sách các mặt; mỗi mặt là mảng 478 điểm (tọa độ pixel trên ảnh bgr)."""
    import mediapipe as mp
    H, W = bgr.shape[:2]
    s = min(1.0, 1600 / max(H, W))                      # ảnh rất lớn: thu nhỏ cho MediaPipe nhanh
    nho = bgr if s == 1 else cv2.resize(bgr, (round(W * s), round(H * s)),
                                        interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(nho, cv2.COLOR_BGR2RGB)          # MediaPipe cần ảnh RGB
    kq = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    # tọa độ MediaPipe là tỉ lệ 0..1 -> nhân với kích thước ảnh GỐC ra pixel
    return [np.array([(p.x * W, p.y * H) for p in m]) for m in kq.face_landmarks]


def thu_muc_lfw_goc():
    """Thư mục ảnh LFW gốc (mỗi người 1 thư mục); None nếu chưa tải."""
    for p in (os.path.join(C.THU_MUC_LFW_GOC, "lfw_home", "lfw_funneled"),      # tải tự động
              os.path.join(C.THU_MUC_LFW_GOC, "lfw_funneled"),                  # giải nén tay
              os.path.join(os.path.expanduser("~"), "scikit_learn_data", "lfw_home",
                           "lfw_funneled")):
        if os.path.isdir(os.path.join(p, "George_W_Bush")):
            return p
    return None


def chuan_bi_lfw():
    """Tải LFW -> chọn người -> MediaPipe tìm mặt -> căn chỉnh 5 điểm GIỐNG HỆT ảnh tự thu
    -> lưu dataset/lfw_xam_100. Chỉ làm 1 lần, các lần sau dùng lại."""
    p_ds = os.path.join(C.THU_MUC_LFW, "danhsachfwl.csv")
    if os.path.exists(p_ds):
        print("  Đã có ảnh LFW căn chỉnh sẵn")
        return
    if thu_muc_lfw_goc() is None:
        print("  Đang tải bộ LFW (~230 MB, chỉ 1 lần)...")
        from sklearn.datasets import fetch_lfw_people   # hàm tải bộ LFW của scikit-learn
        try:
            fetch_lfw_people(data_home=C.THU_MUC_LFW_GOC, min_faces_per_person=LFW_SO_ANH_TOI_THIEU,
                             resize=0.1)
        except Exception as e:
            print(f"  ! Không tải được: {e}")
    goc = thu_muc_lfw_goc()
    if goc is None:
        raise SystemExit("Chưa có LFW: tải tay lfw-funneled.tgz tại "
                         "https://ndownloader.figshare.com/files/5976015, giải nén vào dataset/du_lieu_lfw")
    import thu_anh as T
    rng = np.random.default_rng(C.RANDOM_STATE)
    thu_muc_nguoi = sorted(n for n in os.listdir(goc) if os.path.isdir(os.path.join(goc, n)))
    so_anh = {n: len(os.listdir(os.path.join(goc, n))) for n in thu_muc_nguoi}
    biet = [n for n, s in so_anh.items() if s >= LFW_SO_ANH_TOI_THIEU]
    con_lai = [n for n, s in so_anh.items() if 5 <= s < LFW_SO_ANH_TOI_THIEU]
    la = sorted(rng.choice(con_lai, size=min(LFW_SO_NGUOI_LA, len(con_lai)), replace=False))
    # kế hoạch: (người, vai trò, số ảnh tối đa)
    ke_hoach = ([(n, "biet", LFW_TOI_DA_MOI_NGUOI) for n in biet]
                + [(n, "nguoi_la", LFW_ANH_MOI_NGUOI_LA) for n in la])
    lm, dong = tao_bo_tim_mat(so_mat=3), []
    for n, vai_tro, toi_da in ke_hoach:
        files = sorted(os.listdir(os.path.join(goc, n)))
        for fn in sorted(rng.choice(files, size=min(toi_da, len(files)), replace=False)):
            bgr = T.doc_anh(os.path.join(goc, n, fn))
            cac_mat = tim_mat(lm, bgr) if bgr is not None else []
            if not cac_mat:                             # không đọc được / không thấy mặt -> bỏ
                continue
            H, W = bgr.shape[:2]
            # nhiều mặt -> lấy mặt gần tâm ảnh nhất (LFW luôn đặt người được gán nhãn ở giữa)
            diem = min(cac_mat, key=lambda d: np.hypot(*(d.mean(0) - (W / 2, H / 2))))
            mat = T.can_chinh_mat(bgr, T.diem_5(diem), C.KICH_THUOC)
            ten_file = os.path.splitext(fn)[0]
            C.luu_png(os.path.join(C.THU_MUC_LFW, n, ten_file + ".png"),
                      cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY))
            dong.append({"file": ten_file, "ten": n, "vai_tro": vai_tro})
    lm.close()
    pd.DataFrame(dong).to_csv(p_ds, index=False)
    print(f"  LFW: {len(dong)} ảnh ({len(biet)} người quen, {len(la)} người lạ) -> lfw_xam_100")


def doc_anh_lfw():
    p_ds = os.path.join(C.THU_MUC_LFW, "danhsachfwl.csv")
    lfw = pd.read_csv(p_ds, dtype=str)
    dong = [{"duong_dan": os.path.join(C.THU_MUC_LFW, r.ten, r.file + ".png"), "file": r.file,
             "ten": r.ten, "nguon": "lfw", "luot": 0, "tu_the": "", "vai_tro": r.vai_tro} for r in lfw.itertuples()]
    print(f"  Ảnh LFW: {len(dong)} ảnh")
    return dong


# ======================= 1.3 TIỀN XỬ LÝ =======================
def tien_xu_ly_tat_ca(dong):
    """Đọc từng ảnh -> C.tien_xu_ly (lọc Gaussian + CLAHE). Ảnh hỏng/sai kích thước thì bỏ."""
    anh, giu = [], []
    for d in dong:
        a = C.doc_anh_xam(d["duong_dan"])
        if a is not None and a.shape == (C.KICH_THUOC, C.KICH_THUOC):
            anh.append(C.tien_xu_ly(a))
            giu.append(d)
    return np.array(anh, dtype=np.uint8), pd.DataFrame(giu)


# ======================= 1.4 THỐNG KÊ, TRỰC QUAN HÓA =======================
def thong_ke_va_ve(df, anh):
    biet = df[df["vai_tro"] == "biet"]
    bang = pd.crosstab(biet["ten"], biet["nguon"])              # số ảnh mỗi người theo từng nguồn
    bang = bang[[n for n in C.NGUON if n in bang]]             # cột: tự thu và LFW
    bang = bang.sort_values(list(bang.columns), ascending=False)   # người tự thu lên đầu
    print(bang.assign(tong=bang.sum(axis=1)).to_string())
    print(f"  Người lạ (chỉ test): {(df['vai_tro'] == 'nguoi_la').sum()} ảnh")

    # (a) Số ảnh mỗi người: thanh ngang xếp chồng, mỗi nguồn một màu
    ax = bang.rename(columns={n: C.NGUON[n][0] for n in bang}).rename_axis(columns=None).plot.barh(
        stacked=True, figsize=(9, 0.4 * len(bang) + 1.5), edgecolor="black",
        color=[C.NGUON[n][1] for n in bang])
    ax.invert_yaxis()
    ax.set(xlabel="Số ảnh", ylabel="",
           title=f"Số ảnh mỗi người: {len(biet)} ảnh, {len(bang)} người")
    C.luu_hinh("1_so_anh_moi_nguoi.png")

    # (b) Ảnh mẫu: mỗi người 1 hàng 8 ảnh
    rng = np.random.default_rng(C.RANDOM_STATE)
    fig, ax = plt.subplots(len(bang), 8, figsize=(10.4, 1.45 * len(bang)), squeeze=False)
    for i, ten in enumerate(bang.index):
        idx = biet.index[biet["ten"] == ten]
        chon = rng.choice(idx, size=min(8, len(idx)), replace=False)
        for j in range(8):
            ax[i, j].axis("off")
            if j < len(chon):
                ax[i, j].imshow(anh[chon[j]], cmap="gray")
        ax[i, 0].text(-8, 50, C.rut_gon(ten), ha="right", va="center", fontsize=8)
    plt.suptitle("Ảnh mẫu sau tiền xử lý (căn chỉnh 5 điểm, xám 100x100, lọc Gaussian + CLAHE)")
    C.luu_hinh("1_anh_mau.png")

    # (c) Trước / sau tiền xử lý: 4 ảnh + histogram mức xám của 200 ảnh
    chon = rng.choice(len(df), size=200 if len(df) > 200 else len(df), replace=False)
    truoc = np.array([C.doc_anh_xam(p) for p in df.loc[chon, "duong_dan"]])
    fig = plt.figure(figsize=(12, 4.6))
    for j in range(min(4, len(chon))):
        for hang, a_, nhan in ((0, truoc[j], "Trước"), (1, anh[chon[j]], "Sau lọc + CLAHE")):
            a = fig.add_subplot(2, 6, hang * 6 + j + 1)
            a.imshow(a_, cmap="gray", vmin=0, vmax=255)
            a.set_title(nhan if j == 0 else "", fontsize=9, loc="left")
            a.axis("off")
    a = fig.add_subplot(1, 3, 3)
    a.hist(truoc.ravel(), bins=64, alpha=0.6, label="Trước", color="gray")
    a.hist(anh[chon].ravel(), bins=64, alpha=0.6, label="Sau lọc + CLAHE", color="steelblue")
    a.set(xlabel="Mức xám", ylabel="Tần số", title="Histogram mức xám")
    a.legend()
    C.luu_hinh("1_can_bang_sang.png")

    # (d) Độ sáng trung bình mỗi ảnh theo nguồn (count, mean, median, std – như bài giảng)
    do_sang = pd.Series(anh.reshape(len(anh), -1).mean(1), index=df.index)
    print("\n  Độ sáng trung bình mỗi ảnh (sau tiền xử lý):")
    print(do_sang.groupby(df["nguon"]).agg(["count", "mean", "median", "std"]).round(1).to_string())


# =====================================================================
def main():
    chi_chuan_bi = "--chuan-bi" in sys.argv
    print("=" * 70)
    print(" BƯỚC 1 – CHUẨN BỊ DỮ LIỆU")
    print("=" * 70)
    print("\n1.1 Đọc ảnh tự thu (thu_anh.py)")
    dong = gan_vai_tro(doc_anh_tu_thu())

    print("\n1.2 Chuẩn bị ảnh LFW")
    chuan_bi_lfw()
    if chi_chuan_bi:
        print("\nXong phần chuẩn bị (--chuan-bi).")
        return
    dong += doc_anh_lfw()

    print("\n1.3 Tiền xử lý (lọc Gaussian + CLAHE)")
    anh, df = tien_xu_ly_tat_ca(dong)
    print(f"  {len(anh)} ảnh -> mảng {anh.shape} (mỗi ảnh sẽ thành vector 10.000 chiều)")
    if df.empty:
        raise SystemExit("Không có ảnh hợp lệ. Hãy chạy thu_anh.py hoặc kiểm tra dataset/mat_xam_100.")
    if df.loc[df["vai_tro"] == "biet", "ten"].nunique() < 2:
        raise SystemExit("Cần ít nhất 2 người quen để phân loại. Thêm ảnh cho người quen.")

    print("\n1.4 Thống kê và trực quan hóa")
    thong_ke_va_ve(df, anh)

    print("\n1.5 Lưu dữ liệu")
    os.makedirs(C.DATASET, exist_ok=True)
    np.savez_compressed(C.FILE_ANH, anh=anh)                   # lưu mảng ảnh (nén)
    df.drop(columns="duong_dan").to_csv(C.FILE_DANH_SACH, index=False)
    print("  -> dataset/du_lieu_anh.npz\n  -> dataset/danh_sach.csv")
    print("\nXong bước 1. Tiếp theo: python buoc2_chia_du_lieu.py")


if __name__ == "__main__":
    main()

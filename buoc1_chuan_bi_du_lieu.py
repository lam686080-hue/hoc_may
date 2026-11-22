"""
=====================================================================
 BƯỚC 1 – CHUẨN BỊ DỮ LIỆU
 Phụ trách: Thành viên 1 – Bùi Trung Hiếu
=====================================================================
 Làm gì   : Đọc ảnh mặt tự thu, đưa về CÙNG một dạng (căn chỉnh 5 điểm, xám 100x100,
            lọc Gaussian + CLAHE), thống kê + vẽ biểu đồ, lưu lại cho các bước sau.
 Đầu vào  : dataset/mat_xam_100/<Tên>/*.png    ảnh nhóm tự thu bằng thu_anh.py (đã căn chỉnh)
 Đầu ra   : dataset/du_lieu_anh.npz   tất cả ảnh đã tiền xử lý (N x 100 x 100)
            dataset/danh_sach.csv     mỗi dòng 1 ảnh: file, tên, nguồn, lượt, tư thế, vai trò
            ket_qua/1_*.png           biểu đồ cho báo cáo
 Các bước : 1.1 Đọc ảnh tự thu của người quen
            1.2 Tiền xử lý: lọc Gaussian + cân bằng sáng CLAHE
            1.3 Thống kê và trực quan hóa
            1.4 Lưu dữ liệu
 Chạy     : python buoc1_chuan_bi_du_lieu.py
=====================================================================
"""
import os
import glob
from contextlib import ExitStack
from pathlib import Path
import cv2
import mediapipe as mp
import thu_anh as T

import numpy as np
import pandas as pd
import chung as C
from chung import plt

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


# ======================= 1.2 TIỀN XỬ LÝ =======================
def tien_xu_ly_tat_ca(dong):
    """Đọc từng ảnh -> C.tien_xu_ly (lọc Gaussian + CLAHE). Ảnh hỏng/sai kích thước thì bỏ."""
    anh, giu = [], []
    for d in dong:
        a = C.doc_anh_xam(d["duong_dan"])
        if a is not None and a.shape == (C.KICH_THUOC, C.KICH_THUOC):
            anh.append(C.tien_xu_ly(a))
            giu.append(d)
    return np.array(anh, dtype=np.uint8), pd.DataFrame(giu)


# ======================= 1.3 THỐNG KÊ, TRỰC QUAN HÓA =======================
def thong_ke_va_ve(df, anh):
    biet = df[df["vai_tro"] == "biet"]
    bang = pd.crosstab(biet["ten"], biet["nguon"])              # số ảnh mỗi người theo từng nguồn
    bang = bang[[n for n in C.NGUON if n in bang]]
    bang = bang.sort_values(list(bang.columns), ascending=False)   # người tự thu lên đầu
    print(bang.assign(tong=bang.sum(axis=1)).to_string())

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
def doc_mat(path, detector):
    try:
        anh = T.doc_anh(str(path))
    except (cv2.error, OSError):
        anh = None
    if anh is None:
        return "loi_doc", None
    h, w = anh.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        anh = cv2.resize(anh, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
    faces = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks
    if len(faces) != 1:
        return ("nhieu_mat" if faces else "khong_thay_mat"), None
    h, w = anh.shape[:2]
    info = T.trich_thong_tin(faces[0], None, w, h)
    if info is None:
        return "khong_can_chinh_duoc", None
    return "ok", C.mat_xam_chuan(anh, info["d5"])


def chuan_bi_nguoi_la():
    data = {}
    for nhom in ("nguoi_la_chon_nguong", "nguoi_la"):
        folder = Path(C.DATASET) / nhom
        folder.mkdir(exist_ok=True)
        files = sorted(p for p in folder.rglob("*") if p.is_file() and
                       p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        names, status, faces = [], [], []
        with ExitStack() as resources:
            detector = None
            for path in files:
                rel = path.relative_to(folder)
                if nhom == "nguoi_la_chon_nguong" and rel.parts[0] == "lfw_xam_100":
                    # Ảnh này đã căn 5 điểm và chuyển xám khi chuẩn bị LFW.
                    gray = C.doc_anh_xam(str(path))
                    face = None
                    state = "loi_anh_can_chinh"
                    if gray is not None and gray.shape == (C.KICH_THUOC, C.KICH_THUOC):
                        face = C.tien_xu_ly(gray)
                        state = "ok"
                else:
                    if detector is None:
                        detector = resources.enter_context(T.tao_landmarker("image"))
                    state, face = doc_mat(path, detector)
                names.append(rel.as_posix())
                status.append(state)
                if face is not None:
                    faces.append(face)
        print(f"  {nhom}: {len(faces)}/{len(files)} ảnh hợp lệ")
        data[nhom] = np.asarray(faces, dtype=np.uint8).reshape(-1, C.KICH_THUOC, C.KICH_THUOC)
        data[nhom + "_file"] = np.asarray(names, dtype=str)
        data[nhom + "_status"] = np.asarray(status, dtype=str)
    return data


def main():
    print("=" * 70)
    print(" BƯỚC 1 – CHUẨN BỊ DỮ LIỆU")
    print("=" * 70)
    print("\n1.1 Đọc ảnh tự thu (thu_anh.py)")
    dong = doc_anh_tu_thu()

    print("\n1.2 Tiền xử lý (lọc Gaussian + CLAHE)")
    anh, df = tien_xu_ly_tat_ca(dong)
    print(f"  {len(anh)} ảnh -> mảng {anh.shape} (mỗi ảnh sẽ thành vector 10.000 chiều)")
    if df.empty:
        raise SystemExit("Không có ảnh hợp lệ. Hãy chạy thu_anh.py hoặc kiểm tra dataset/mat_xam_100.")
    if df.loc[df["vai_tro"] == "biet", "ten"].nunique() < 2:
        raise SystemExit("Cần ít nhất 2 người quen để phân loại. Thêm ảnh cho người quen.")

    print("\n1.3 Thống kê và trực quan hóa")
    thong_ke_va_ve(df, anh)

    print("\n1.4 Lưu dữ liệu")
    os.makedirs(C.DATASET, exist_ok=True)
    np.savez_compressed(C.FILE_ANH, anh=anh, **chuan_bi_nguoi_la())                   # lưu mảng ảnh (nén)
    df.drop(columns="duong_dan").to_csv(C.FILE_DANH_SACH, index=False)
    print("  -> dataset/du_lieu_anh.npz\n  -> dataset/danh_sach.csv")
    print("\nXong bước 1. Tiếp theo: python buoc2_chia_du_lieu.py")


if __name__ == "__main__":
    main()

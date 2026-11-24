"""
=====================================================================
 BƯỚC 1 – CHUẨN BỊ DỮ LIỆU
 Phụ trách: Thành viên 1 – Bùi Trung Hiếu
=====================================================================
 Làm gì   : Đọc ảnh mặt tự thu và ảnh người lạ, đưa về CÙNG một dạng (căn chỉnh 5 điểm,
            xám 100x100, lọc Gaussian + CLAHE), thống kê + vẽ biểu đồ, lưu lại cho các bước sau.
 Đầu vào  : dataset/mat_xam_100/<Tên>/*.png        ảnh nhóm tự thu bằng thu_anh.py (đã căn chỉnh)
            dataset/nguoi_la_chon_nguong/*         ảnh người lạ dùng để chọn ngưỡng ở bước 5
            dataset/nguoi_la/*                     ảnh người lạ dùng để kiểm tra độc lập ở bước 5
 Đầu ra   : dataset/du_lieu_anh.npz   tất cả ảnh người quen + người lạ đã tiền xử lý (N x 100 x 100)
            dataset/danh_sach.csv     mỗi dòng 1 ảnh người quen: file, tên, lượt, tư thế
            ket_qua/1_*.png           biểu đồ cho báo cáo
 Các bước : 1.1 Đọc ảnh tự thu của người quen
            1.2 Tiền xử lý: lọc Gaussian + cân bằng sáng CLAHE
            1.3 Thống kê và trực quan hóa
            1.4 Chuẩn bị ảnh người lạ và lưu dữ liệu
 Chạy     : python buoc1_chuan_bi_du_lieu.py
=====================================================================
"""
import os
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

import chung as C
from chung import plt
import thu_anh as T


# ======================= 1.1 ĐỌC ẢNH TỰ THU =======================
def doc_anh_tu_thu():
    """Tên ảnh do thu_anh.py đặt: <Tên>_<số>_L<lượt>_<tư thế>.png -> biết người, lượt, tư thế.
    Ảnh bị kiem_tra.py đánh dấu LOI thì bỏ qua."""
    print(f"  Đọc ảnh tự thu tại: {C.THU_MUC_TU_THU}")
    p_loi = Path(C.DATASET) / "kiem_tra" / "bao_cao.csv"
    loi = set()
    if p_loi.exists():
        bc = pd.read_csv(p_loi, encoding="utf-8-sig", dtype=str)
        loi = set(bc.loc[bc["trang_thai"] == "LOI", "file"])
    dong, sai_ten = [], 0
    for p in sorted(Path(C.THU_MUC_TU_THU).glob("*/*.png")):
        ten, goc = p.parent.name, p.stem
        tt = T.thong_tin_ten_anh(ten, goc)
        if tt is None:
            sai_ten += 1
        elif goc not in loi:
            dong.append({"duong_dan": str(p), "file": goc, "ten": ten, "luot": tt[1], "tu_the": tt[2]})
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
    dem = df["ten"].value_counts()
    print(dem.to_frame("so_anh").to_string())

    # (a) Số ảnh mỗi người
    ax = dem.plot.barh(figsize=(9, 0.4 * len(dem) + 1.5), color="darkorange", edgecolor="black")
    ax.invert_yaxis()
    ax.set(xlabel="Số ảnh", ylabel="", title=f"Số ảnh mỗi người: {len(df)} ảnh, {len(dem)} người")
    C.luu_hinh("1_so_anh_moi_nguoi.png")

    # (b) Ảnh mẫu: mỗi người 1 hàng 8 ảnh
    rng = np.random.default_rng(C.RANDOM_STATE)
    fig, ax = plt.subplots(len(dem), 8, figsize=(10.4, 1.45 * len(dem)), squeeze=False)
    for i, ten in enumerate(dem.index):
        idx = df.index[df["ten"] == ten]
        chon = rng.choice(idx, size=min(8, len(idx)), replace=False)
        for j, a in enumerate(ax[i]):
            a.axis("off")
            if j < len(chon):
                a.imshow(anh[chon[j]], cmap="gray")
        ax[i, 0].text(-8, 50, C.rut_gon(ten), ha="right", va="center", fontsize=8)
    plt.suptitle("Ảnh mẫu sau tiền xử lý (căn chỉnh 5 điểm, xám 100x100, lọc Gaussian + CLAHE)")
    C.luu_hinh("1_anh_mau.png")

    # (c) Trước / sau tiền xử lý: 4 ảnh + histogram mức xám của 200 ảnh
    chon = rng.choice(len(df), size=min(200, len(df)), replace=False)
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

    # (d) Độ sáng trung bình mỗi ảnh sau tiền xử lý
    ds = pd.Series(anh.reshape(len(anh), -1).mean(1)).agg(["count", "mean", "median", "std"]).round(1)
    print(f"\n  Độ sáng trung bình sau tiền xử lý: {ds.to_dict()}")


# ======================= 1.4 CHUẨN BỊ ẢNH NGƯỜI LẠ & LƯU =======================
def doc_mat(path, detector):
    """Đọc 1 ảnh bất kỳ -> MediaPipe tìm đúng 1 mặt -> căn chỉnh 5 điểm + tiền xử lý."""
    try:
        anh = T.doc_anh(str(path))
    except (cv2.error, OSError):
        anh = None
    if anh is None:
        return "loi_doc", None
    h, w = anh.shape[:2]
    if max(h, w) > 1200:
        tl = 1200 / max(h, w)
        anh = cv2.resize(anh, (max(1, round(w * tl)), max(1, round(h * tl))), interpolation=cv2.INTER_AREA)
        h, w = anh.shape[:2]
    rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
    faces = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks
    if len(faces) != 1:
        return ("nhieu_mat" if faces else "khong_thay_mat"), None
    info = T.trich_thong_tin(faces[0], None, w, h)
    return ("ok", C.mat_xam_chuan(anh, info["d5"])) if info is not None else ("khong_can_chinh_duoc", None)


def chuan_bi_nguoi_la():
    """Đọc và tiền xử lý 2 bộ ảnh người lạ (nguoi_la_chon_nguong và nguoi_la) cho bước 4 và 5."""
    data, detector = {}, None
    try:
        for nhom in ("nguoi_la_chon_nguong", "nguoi_la"):
            folder = Path(C.DATASET) / nhom
            folder.mkdir(exist_ok=True)
            files = sorted(p for p in folder.rglob("*") if p.is_file() and
                           p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            names, status, faces = [], [], []
            for path in files:
                rel = path.relative_to(folder)
                if nhom == "nguoi_la_chon_nguong" and rel.parts[0] == "lfw_xam_100":
                    gray = C.doc_anh_xam(str(path))           # ảnh LFW đã căn 5 điểm và chuyển xám sẵn
                    ok = gray is not None and gray.shape == (C.KICH_THUOC, C.KICH_THUOC)
                    state, face = ("ok", C.tien_xu_ly(gray)) if ok else ("loi_anh_can_chinh", None)
                else:
                    detector = detector or T.tao_landmarker("image")
                    state, face = doc_mat(path, detector)
                names.append(rel.as_posix())
                status.append(state)
                if face is not None:
                    faces.append(face)
            print(f"  {nhom}: {len(faces)}/{len(files)} ảnh hợp lệ")
            data[nhom] = np.asarray(faces, dtype=np.uint8).reshape(-1, C.KICH_THUOC, C.KICH_THUOC)
            data[nhom + "_file"] = np.asarray(names, dtype=str)
            data[nhom + "_status"] = np.asarray(status, dtype=str)
    finally:
        if detector is not None:
            detector.close()
    return data


def main():
    print("=" * 70 + "\n BƯỚC 1 – CHUẨN BỊ DỮ LIỆU\n" + "=" * 70)
    print("\n1.1 Đọc ảnh tự thu (thu_anh.py)")
    dong = doc_anh_tu_thu()

    print("\n1.2 Tiền xử lý (lọc Gaussian + CLAHE)")
    anh, df = tien_xu_ly_tat_ca(dong)
    print(f"  {len(anh)} ảnh -> mảng {anh.shape} (mỗi ảnh sẽ thành vector 10.000 chiều)")
    if df.empty or df["ten"].nunique() < 2:
        raise SystemExit("Cần ít nhất 2 người quen hợp lệ trong dataset/mat_xam_100.")

    print("\n1.3 Thống kê và trực quan hóa")
    thong_ke_va_ve(df, anh)

    print("\n1.4 Lưu dữ liệu")
    os.makedirs(C.DATASET, exist_ok=True)
    np.savez_compressed(C.FILE_ANH, anh=anh, **chuan_bi_nguoi_la())
    df.drop(columns="duong_dan").to_csv(C.FILE_DANH_SACH, index=False)
    print("  -> dataset/du_lieu_anh.npz\n  -> dataset/danh_sach.csv")
    print("\nXong bước 1. Tiếp theo: python buoc2_chia_du_lieu.py")


if __name__ == "__main__":
    main()

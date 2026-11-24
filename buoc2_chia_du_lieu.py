"""
=====================================================================
 BƯỚC 2 – CHIA DỮ LIỆU THÀNH TẬP TRAIN VÀ TEST
 Phụ trách: Thành viên 2 – Nguyễn Văn Lâm
=====================================================================
 Làm gì   : Quyết định ảnh nào dùng để HỌC (train), ảnh nào để KIỂM TRA (test).
            Mô hình chỉ được nhìn tập train; tập test để đánh giá khách quan ở bước 5.
 Cách chia:
   - Chia ảnh người quen theo từng người, khoảng 75% train / 25% test (chia phân tầng
     stratify theo tư thế để cả tập train và test đều có đủ các góc mặt).
     Người có từ hai ảnh lấy ít nhất một ảnh test; người chỉ có một ảnh dùng để train.
   - Ảnh người lạ đã được bước 1 lưu riêng trong dataset/du_lieu_anh.npz
     (nguoi_la_chon_nguong và nguoi_la), không tham gia huấn luyện.
 Đầu vào  : dataset/danh_sach.csv (bước 1)
 Đầu ra   : dataset/danh_sach.csv thêm cột "tap" = train / test
            ket_qua/2_chia_train_test.png
 Các bước : 2.1 Đọc danh sách ảnh
            2.2 Chia theo quy tắc trên (hàm chia)
            2.3 Kiểm tra + bảng số ảnh train/test của mỗi người
            2.4 Vẽ biểu đồ cột xếp chồng
            2.5 Lưu
 Chạy     : python buoc2_chia_du_lieu.py
=====================================================================
"""
import pandas as pd
from sklearn.model_selection import train_test_split   # hàm chia ngẫu nhiên của scikit-learn
import chung as C

TI_LE_TEST = 0.25          # mặc định: 25% mỗi người làm test


# ======================= 2.2 CHIA DỮ LIỆU =======================
def chia(df):
    """Chia ảnh người quen theo từng người, phân tầng theo tư thế (tu_the)."""
    tap = pd.Series("", index=df.index)
    for _, nhom in df.groupby("ten"):
        if len(nhom) == 1:
            tap.loc[nhom.index] = "train"
            continue
        so_test = max(1, round(TI_LE_TEST * len(nhom)))
        dem = nhom["tu_the"].value_counts() if "tu_the" in nhom else pd.Series(dtype=int)
        strat = nhom["tu_the"] if len(dem) and dem.min() >= 2 and so_test >= len(dem) else None
        idx_tr, idx_te = train_test_split(nhom.index, test_size=so_test,
                                          random_state=C.RANDOM_STATE, stratify=strat)
        tap.loc[idx_tr], tap.loc[idx_te] = "train", "test"
    return tap


# =====================================================================
def main():
    print("=" * 70 + "\n BƯỚC 2 – CHIA DỮ LIỆU TRAIN / TEST\n" + "=" * 70)

    print("\n2.1 Đọc danh sách ảnh")
    _, df = C.doc_du_lieu()
    print(f"  {len(df)} ảnh người quen")

    print("\n2.2 Chia dữ liệu")
    df["tap"] = chia(df)

    print("\n2.3 Kiểm tra")
    assert (df["tap"] != "").all(), "Có ảnh chưa được gán tập"
    assert not df["file"].duplicated().any(), "Có ảnh bị trùng (sẽ nằm ở 2 tập)"
    assert set(df["ten"]) == set(df.loc[df["tap"] == "train", "ten"]), "Có người quen không có ảnh train"
    bang = pd.crosstab(df["ten"], df["tap"]).reindex(columns=["train", "test"], fill_value=0)
    print(bang.to_string())
    dem = df["tap"].value_counts()
    so_la_nguong, so_la_test = len(C.doc_nguoi_la("nguoi_la_chon_nguong")[0]), len(C.doc_nguoi_la("nguoi_la")[0])
    print(f"  Tổng người quen: {dem.to_dict()} | người lạ riêng: {so_la_nguong} chọn ngưỡng, {so_la_test} kiểm tra")

    print("\n2.4 Biểu đồ xếp chồng: số ảnh train / test của mỗi người")
    ax = bang.rename_axis(columns=None).plot.bar(
        stacked=True, figsize=(max(8, 0.8 * len(bang)), 4.5),
        color=["steelblue", "darkorange"], edgecolor="black")
    for cot in ax.containers:
        ax.bar_label(cot, labels=[f"{v:.0f}" if v else "" for v in cot.datavalues],
                     label_type="center", fontsize=7)
    ax.set_xticklabels([C.rut_gon(t) for t in bang.index], rotation=30, ha="right")
    ax.set(xlabel="", ylabel="Số ảnh",
           title=f"Chia dữ liệu: train {dem.get('train', 0)} | test {dem.get('test', 0)} | "
                 f"người lạ riêng: {so_la_nguong} chọn ngưỡng, {so_la_test} kiểm tra")
    C.luu_hinh("2_chia_train_test.png")

    print("\n2.5 Lưu")
    df.to_csv(C.FILE_DANH_SACH, index=False)                 # ghi đè, có thêm cột "tap"
    print("  -> dataset/danh_sach.csv (đã thêm cột tap)")
    print("\nXong bước 2. Tiếp theo: python buoc3_huan_luyen.py")


if __name__ == "__main__":
    main()

"""
=====================================================================
 BƯỚC 2 – CHIA DỮ LIỆU THÀNH TẬP TRAIN VÀ TEST
 Phụ trách: Thành viên 2 – Nguyễn Văn Lâm
=====================================================================
 Làm gì   : Quyết định ảnh nào dùng để HỌC (train), ảnh nào để KIỂM TRA (test).
            Mô hình chỉ được nhìn tập train; tập test để đánh giá khách quan ở bước 5.
 Cách chia:
   - Ảnh tự thu có một lượt. Với ảnh người quen, chia theo từng người và từng nguồn
     (ảnh tự thu), khoảng 75% train / 25% test. Nhóm có từ hai ảnh lấy ít nhất một
     ảnh test. Nhóm chỉ có một ảnh dùng ảnh đó để train.
   - Người lạ: chỉ để test (mô hình chưa từng thấy) -> bước 5 chọn ngưỡng người lạ.
 Đầu vào  : dataset/danh_sach.csv (bước 1)
 Đầu ra   : dataset/danh_sach.csv thêm cột "tap" = train / test / nguoi_la
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
    """Trả về cột "tap" (train / test / nguoi_la) cho từng ảnh. Chia riêng từng (nguồn, người)."""
    tap = pd.Series("", index=df.index)                      # cột kết quả, ban đầu để trống
    tap[df["vai_tro"] == "nguoi_la"] = "nguoi_la"            # người lạ: chỉ để test
    biet = df[df["vai_tro"] == "biet"]                       # người quen: mô hình phải nhận ra
    for _, nhom in biet.groupby(["nguon", "ten"]):
        if len(nhom) == 1:
            tap[nhom.index] = "train"
        else:
            # Chia ngẫu nhiên, giữ ít nhất một ảnh test trong mỗi nhóm có từ hai ảnh.
            so_test = max(1, round(TI_LE_TEST * len(nhom)))
           strat = None
           if nhom["nguon"].iloc[0] == "tu_thu" and "tu_the" in nhom.columns:
              dem = nhom["tu_the"].value_counts()
            if len(dem) > 0 and dem.min() >= 2 and so_test >= len(dem):
               strat = nhom["tu_the"]
     
            idx_train, idx_test = train_test_split(nhom.index, test_size=so_test,
                                                   stratify=strat, random_state=C.RANDOM_STATE)
            tap[idx_train] = "train"
            tap[idx_test] = "test"
    return tap


# =====================================================================
def main():
    print("=" * 70)
    print(" BƯỚC 2 – CHIA DỮ LIỆU TRAIN / TEST")
    print("=" * 70)

    print("\n2.1 Đọc danh sách ảnh")
    _, df = C.doc_du_lieu()                                  # chỉ cần bảng thông tin ảnh
    print(f"  {len(df)} ảnh")

    print("\n2.2 Chia dữ liệu")
    df["tap"] = chia(df)
    # người quen không có ảnh train nào -> mô hình không học được -> coi là người lạ
    khong_train = (set(df.loc[df["vai_tro"] == "biet", "ten"])
                   - set(df.loc[df["tap"] == "train", "ten"]))
    if khong_train:
        df.loc[df["ten"].isin(khong_train), ["vai_tro", "tap"]] = "nguoi_la"
        print("  ! Không có ảnh train -> chuyển thành người lạ: " + ", ".join(sorted(khong_train)))

    print("\n2.3 Kiểm tra")
    assert (df["tap"] != "").all(), "Có ảnh chưa được gán tập"
    assert not df["file"].duplicated().any(), "Có ảnh bị trùng (sẽ nằm ở 2 tập)"
    assert not ((df["vai_tro"] == "nguoi_la") & (df["tap"] == "train")).any(), \
        "Người lạ không được có ảnh train"
    biet = df[df["vai_tro"] == "biet"]
    bang = pd.crosstab(biet["ten"], biet["tap"]).reindex(columns=["train", "test"], fill_value=0)
    print(bang.to_string())
    dem = df["tap"].value_counts()
    print("  Tổng:", dem.to_dict())

    print("\n2.4 Biểu đồ xếp chồng: số ảnh train / test của mỗi người")
    bang = bang.rename_axis(columns=None)                    # bỏ chữ "tap" ở chú thích
    ax = bang.plot.bar(stacked=True, figsize=(max(8, 0.8 * len(bang)), 4.5),
                       color=["steelblue", "darkorange"], edgecolor="black")
    for cot in ax.containers:                                # ghi số ảnh lên giữa mỗi phần cột
        ax.bar_label(cot, labels=[f"{v:.0f}" if v else "" for v in cot.datavalues],
                     label_type="center", fontsize=7)        # phần 0 ảnh: không ghi số
    ax.set_xticklabels([C.rut_gon(t) for t in bang.index], rotation=30, ha="right")
    ax.set(xlabel="", ylabel="Số ảnh", title=f"Chia dữ liệu: train {dem.get('train', 0)} | "
           f"test {dem.get('test', 0)} | người lạ (chỉ test) {dem.get('nguoi_la', 0)}")
    C.luu_hinh("2_chia_train_test.png")

    print("\n2.5 Lưu")
    df.to_csv(C.FILE_DANH_SACH, index=False)                 # ghi đè, có thêm cột "tap"
    print("  -> dataset/danh_sach.csv (đã thêm cột tap)")
    print("\nXong bước 2. Tiếp theo: python buoc3_huan_luyen.py")


if __name__ == "__main__":
    main()

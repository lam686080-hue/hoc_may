"""
=====================================================================
 BƯỚC 2 – CHIA DỮ LIỆU THÀNH TẬP TRAIN VÀ TEST            (Thành viên 1)
=====================================================================
 Làm gì   : Quyết định ảnh nào dùng để HỌC (train), ảnh nào để KIỂM TRA (test).
            Mô hình chỉ được nhìn tập train; tập test để đánh giá khách quan ở bước 5.
 Cách chia:
   - Ảnh tự thu (thu_anh.py): lượt 1 -> train, lượt 2 (chụp lại, đổi hướng ánh sáng) -> test.
                  Ảnh chụp liên tiếp trong 1 lượt rất giống nhau; chia ngẫu nhiên sẽ có
                  ảnh gần trùng ở cả 2 tập -> kết quả cao ảo.
   - Ảnh chụp tay, ảnh LFW: mỗi người chia ngẫu nhiên 75% train / 25% test
                  (mỗi người ít nhất 1 ảnh test; người chỉ có 1 ảnh -> train).
                  Ảnh chụp tay để trong thư mục con train/ hoặc test/: theo đúng thư mục đó.
   - Người lạ   : chỉ ở tập test (mô hình chưa từng thấy) -> dùng để chọn ngưỡng ở bước 5.
 Đầu vào  : dataset/danh_sach.csv (bước 1)
 Đầu ra   : dataset/danh_sach.csv thêm cột "tap" = train / test / nguoi_la
            ket_qua/2_chia_train_test.png
 Các bước : 2.1 Đọc danh sách ảnh
            2.2 Chia theo quy tắc trên
            2.3 Kiểm tra (không ảnh nào vừa train vừa test, bảng số ảnh mỗi người)
            2.4 Vẽ biểu đồ xếp chồng
            2.5 Lưu
 Chạy     : python buoc2_chia_du_lieu.py
=====================================================================
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   # hàm chia ngẫu nhiên của scikit-learn
import chung as C
from chung import plt

TI_LE_TEST = 0.25          # ảnh chụp tay, ảnh LFW: 25% mỗi người làm test


def chia(df):
    """Trả về cột "tap" cho từng ảnh. Chia riêng theo từng (nguồn, người)."""
    tap = pd.Series("", index=df.index)
    tap[df["vai_tro"] == "nguoi_la"] = "nguoi_la"            # người lạ: chỉ để test
    biet = df[df["vai_tro"] == "biet"]
    for (nguon, ten), nhom in biet.groupby(["nguon", "ten"]):
        chi_dinh = nhom[nhom["tap_co_dinh"] != ""]           # ảnh để trong thư mục con train/test
        tap[chi_dinh.index] = chi_dinh["tap_co_dinh"]
        nhom = nhom[nhom["tap_co_dinh"] == ""]               # phần còn lại chia theo quy tắc
        if len(nhom) == 0:
            continue
        if nguon == "tu_thu" and {1, 2} <= set(nhom["luot"]):
            # ảnh thu_anh.py: chia theo lượt
            tap[nhom.index[nhom["luot"] == 1]] = "train"
            tap[nhom.index[nhom["luot"] == 2]] = "test"
        elif len(nhom) < 2:
            tap[nhom.index] = "train"                         # chỉ có 1 ảnh: để học
        else:
            # ảnh chụp tay, ảnh LFW: chia ngẫu nhiên 75/25, mỗi người ít nhất 1 ảnh test
            # train_test_split xáo trộn danh sách rồi tách ra 2 phần (so_test ảnh làm test)
            so_test = max(1, int(round(TI_LE_TEST * len(nhom))))
            idx_train, idx_test = train_test_split(nhom.index, test_size=so_test,
                                                   random_state=C.RANDOM_STATE)
            tap[idx_train] = "train"
            tap[idx_test] = "test"
    return tap


def main():
    print("=" * 70)
    print(" BƯỚC 2 – CHIA DỮ LIỆU TRAIN / TEST")
    print("=" * 70)

    print("\n2.1 Đọc danh sách ảnh")
    _, df = C.doc_du_lieu()
    print(f"  {len(df)} ảnh")

    print("\n2.2 Chia dữ liệu")
    df["tap"] = chia(df)
    # người quen không có ảnh train nào -> mô hình không học được -> coi là người lạ
    khong_train = sorted(set(df.loc[df["vai_tro"] == "biet", "ten"])
                         - set(df.loc[df["tap"] == "train", "ten"]))
    if khong_train:
        bi = df["ten"].isin(khong_train)
        df.loc[bi, "vai_tro"], df.loc[bi, "tap"] = "nguoi_la", "nguoi_la"
        print("  ! Không có ảnh train -> chuyển thành người lạ: " + ", ".join(khong_train))

    print("\n2.3 Kiểm tra")
    assert (df["tap"] != "").all(), "Có ảnh chưa được gán tập"
    assert df.groupby("file")["tap"].nunique().max() == 1, "Có ảnh nằm ở 2 tập"
    biet = df[df["vai_tro"] == "biet"]
    # pd.crosstab: bảng đếm – mỗi người có bao nhiêu ảnh ở mỗi tập
    print(pd.crosstab(biet["ten"], biet["tap"]).to_string())
    print("  Tổng:", df["tap"].value_counts().to_dict())

    print("\n2.4 Biểu đồ xếp chồng: số ảnh train / test của mỗi người")
    biet = df[df["vai_tro"] == "biet"]
    thu_tu_nguon = {n: i for i, n in enumerate(C.NGUON)}            # tự thu -> chụp tay -> LFW
    nhom = biet.groupby("ten")["nguon"].agg(lambda s: min(thu_tu_nguon[n] for n in s))
    nguoi = sorted(nhom.index, key=lambda t: (nhom[t], t))
    so_tr = [int(((biet["ten"] == n) & (biet["tap"] == "train")).sum()) for n in nguoi]
    so_te = [int(((biet["ten"] == n) & (biet["tap"] == "test")).sum()) for n in nguoi]
    x = np.arange(len(nguoi))
    plt.figure(figsize=(max(8, 0.9 * len(nguoi)), 4.5))
    plt.bar(x, so_tr, color="steelblue", edgecolor="black", label="Train")
    plt.bar(x, so_te, bottom=so_tr, color="darkorange", edgecolor="black", label="Test")
    cao_nhat = max(t + s for t, s in zip(so_tr, so_te))
    for i in range(len(nguoi)):
        if so_tr[i] + so_te[i] >= 0.12 * cao_nhat:               # cột đủ cao: ghi số bên trong
            plt.text(i, so_tr[i] / 2, so_tr[i], ha="center", color="white", fontsize=8)
            plt.text(i, so_tr[i] + so_te[i] / 2, so_te[i], ha="center", fontsize=8)
        else:                                                 # cột thấp: ghi "train+test" bên trên
            plt.text(i, so_tr[i] + so_te[i] + 1, f"{so_tr[i]}+{so_te[i]}", ha="center", fontsize=8)
    plt.xticks(x, [C.rut_gon(n) for n in nguoi], rotation=30, ha="right")
    ten_nguon = list(C.NGUON)
    for nhan, n in zip(plt.gca().get_xticklabels(), nguoi):      # màu chữ theo nguồn ảnh
        nhan.set_color(C.NGUON[ten_nguon[nhom[n]]][1])
    plt.ylabel("Số ảnh")
    dem = df["tap"].value_counts()
    plt.title(f"Chia dữ liệu: train {dem.get('train', 0)} | test {dem.get('test', 0)} | "
              f"người lạ (chỉ test) {dem.get('nguoi_la', 0)}\n(màu tên: cam = tự thu, "
              f"tím = chụp tay, xanh = LFW)", fontsize=10)
    plt.legend()
    C.luu_hinh("2_chia_train_test.png")

    print("\n2.5 Lưu")
    df.to_csv(C.FILE_DANH_SACH, index=False)
    print("  -> dataset/danh_sach.csv (đã thêm cột tap)")
    print("\nXong bước 2. Tiếp theo: python buoc3_huan_luyen.py")


if __name__ == "__main__":
    main()

"""
=====================================================================
 BƯỚC 3 – HUẤN LUYỆN MÔ HÌNH
 Phụ trách: 3.1 – 3.2 (PCA)         Thành viên 1 – Bùi Trung Hiếu
            3.3 – 3.5 (hồi quy logistic)   Thành viên 2 – Nguyễn Văn Lâm
=====================================================================
 Dùng PCA để giảm chiều và chọn tham số C của hồi quy logistic trên tập học.
 Đầu ra: mô hình đã học và kết quả kiểm tra chéo.
"""
import os
import time
import tempfile

import numpy as np
import pandas as pd
import joblib                                                   # lưu / nạp mô hình ra file
from sklearn.preprocessing import StandardScaler                # chuẩn hóa z = (x - mean) / std
from sklearn.decomposition import PCA                           # phân tích thành phần chính
from sklearn.pipeline import Pipeline                           # ghép nhiều khâu thành 1 mô hình
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import chung as C
from chung import plt

GIU_PHUONG_SAI = 0.95      # PCA giữ 95% phương sai
SO_FOLD = 5                # cross-validation 5-fold


# ============ 3.2 PHÂN TÍCH PCA  (Thành viên 1 – Bùi Trung Hiếu) ============
def phan_tich_pca(X_train, y_train):
    """Làm PCA đầy đủ trên tập train để chọn k và vẽ hình minh họa."""
    scaler = StandardScaler().fit(X_train)          # fit: tính mean, std từng pixel (tập train)
    Xs = scaler.transform(X_train)                  # transform: z = (x - mean) / std
    pca = PCA(svd_solver="full").fit(Xs)            # tính TẤT CẢ thành phần chính (vector riêng)
    ti_le = pca.explained_variance_ratio_           # tỉ lệ phương sai của từng thành phần
    tich_luy = np.cumsum(ti_le)
    k = int(np.searchsorted(tich_luy, GIU_PHUONG_SAI) + 1)            # k nhỏ nhất đạt 95%
    k = min(k, int(len(X_train) * (1 - 1 / SO_FOLD)) - 1)            # không vượt số ảnh của 1 fold
    print(f"  {X_train.shape[1]} chiều -> cần k = {k} thành phần chính "
          f"để giữ {GIU_PHUONG_SAI:.0%} phương sai")

    # (a) Biểu đồ kết hợp: phương sai từng thành phần (cột) + tích lũy (đường)
    m = min(len(ti_le), max(60, k + 10))
    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    so_cot = min(50, len(ti_le))                                     # cột: 50 thành phần đầu
    ax1.bar(np.arange(1, so_cot + 1), ti_le[:so_cot] * 100, color="steelblue")
    ax1.set(xlabel="Thành phần chính (PC)", ylabel="Phương sai từng thành phần (%)")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, m + 1), tich_luy[:m] * 100, color="red", marker=".")
    ax2.axhline(GIU_PHUONG_SAI * 100, color="gray", ls="--")
    ax2.axvline(k, color="gray", ls=":")
    ax2.set(ylabel="Phương sai tích lũy (%)", ylim=(0, 102))
    plt.title(f"PCA: {k} thành phần đầu giữ {tich_luy[k - 1] * 100:.1f}% phương sai")
    C.luu_hinh("3_pca_phuong_sai.png")

    # (b) Mặt trung bình + 15 eigenfaces (mỗi vector riêng 10.000 phần tử vẽ thành 1 ảnh)
    fig, ax = plt.subplots(2, 8, figsize=(14, 4))
    ax = ax.ravel()
    ax[0].imshow(scaler.mean_.reshape(C.KICH_THUOC, C.KICH_THUOC), cmap="gray")
    ax[0].set_title("Mặt trung bình", fontsize=9)
    for i in range(min(15, len(pca.components_))):
        ax[i + 1].imshow(pca.components_[i].reshape(C.KICH_THUOC, C.KICH_THUOC), cmap="gray")
        ax[i + 1].set_title(f"PC{i + 1} ({ti_le[i] * 100:.1f}%)", fontsize=8)
    for a in ax:
        a.axis("off")
    plt.suptitle("Eigenfaces: các vector riêng của tập khuôn mặt")
    C.luu_hinh("3_eigenfaces.png")

    # (c) Trước / sau PCA (giống code slide PCA): 2 pixel gốc vs 2 thành phần chính đầu
    Z = pca.transform(Xs)
    so = pd.factorize(y_train)[0]                                   # tên người -> mã số 0, 1, 2…
    p1, p2 = 46 * C.KICH_THUOC + 34, 46 * C.KICH_THUOC + 66          # 2 pixel ở vùng mắt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(Xs[:, p1], Xs[:, p2], c=so, cmap="tab20", edgecolor="k", s=15)
    plt.xlabel("Pixel mắt trái")
    plt.ylabel("Pixel mắt phải")
    plt.title("Trước PCA: 2 pixel gốc (đã chuẩn hóa)")
    plt.subplot(1, 2, 2)
    plt.scatter(Z[:, 0], Z[:, 1], c=so, cmap="tab20", edgecolor="k", s=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Sau PCA: chiếu lên PC1, PC2")
    plt.colorbar(label="Người (mã số)")
    C.luu_hinh("3_truoc_sau_pca.png")

    # (d) Khôi phục ảnh từ m thành phần: x ≈ mean + z_1*PC1 + ... + z_m*PCm
    cac_m = sorted({min(10, k), min(50, k), k})                    # vd 10, 50, k thành phần
    fig, ax = plt.subplots(4, len(cac_m) + 1, figsize=(2 * (len(cac_m) + 1), 8.2))
    for r in range(4):
        ax[r, 0].imshow(X_train[r].reshape(C.KICH_THUOC, C.KICH_THUOC), cmap="gray")
        for j, mm in enumerate(cac_m):
            anh = scaler.inverse_transform(Z[r:r + 1, :mm] @ pca.components_[:mm] + pca.mean_)
            ax[r, j + 1].imshow(anh.reshape(C.KICH_THUOC, C.KICH_THUOC), cmap="gray")
    for j, t in enumerate(["Ảnh gốc"] + [f"{mm} thành phần" for mm in cac_m]):
        ax[0, j].set_title(t, fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    C.luu_hinh("3_khoi_phuc_pca.png")
    return k


def huan_luyen(X_train, y_train, k):
    """Chọn mức điều chuẩn C cho hồi quy logistic bằng kiểm tra chéo."""
    cv = StratifiedKFold(SO_FOLD, shuffle=True, random_state=C.RANDOM_STATE)
    bat_dau = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cache_pca_") as bo_nho:
        pipeline = Pipeline([
            ("chuan_hoa", StandardScaler()),
            ("pca", PCA(n_components=k, random_state=C.RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ], memory=bo_nho)
        tim = GridSearchCV(pipeline, {"model__C": [0.001, 0.01, 0.1, 1]},
                           cv=cv, scoring={"accuracy": "accuracy", "neg_log_loss": "neg_log_loss"}, refit="neg_log_loss", n_jobs=1)
        tim.fit(X_train, y_train)
        mh = tim.best_estimator_.set_params(memory=None)
        idx = tim.best_index_
    bang = pd.DataFrame([{
        "mo_hinh": "Logistic Regression", "tham_so": str({"C": tim.best_params_["model__C"]}),
        "cv_%": round(100 * tim.cv_results_["mean_test_accuracy"][idx], 2),
        "cv_std_%": round(100 * tim.cv_results_["std_test_accuracy"][idx], 2),
        "log_loss": round(-tim.best_score_, 4),
        "thoi_gian_s": round(time.perf_counter() - bat_dau, 1),
    }])
    print(bang.to_string(index=False))
    return mh, bang


def main():
    print("=" * 70)
    print(" BƯỚC 3 – HUẤN LUYỆN MÔ HÌNH")
    print("=" * 70)

    # ---------------- 3.1 – 3.2: Thành viên 1 (Bùi Trung Hiếu) ----------------
    print("\n3.1 Lấy dữ liệu train")
    X, df = C.doc_du_lieu()
    if "tap" not in df:
        raise SystemExit("Chưa chia dữ liệu. Chạy buoc2_chia_du_lieu.py trước.")
    la_train = (df["tap"] == "train").to_numpy()
    X_train, y_train = X[la_train], df.loc[la_train, "ten"].to_numpy()
    print(f"  X_train: {X_train.shape[0]} ảnh x {X_train.shape[1]} chiều | "
          f"{len(set(y_train))} người")
    dem = pd.Series(y_train).value_counts()
    it = dem[dem < SO_FOLD]
    if len(it):                       # vẫn chạy được, nhưng CV của người đó kém tin cậy
        print(f"  ! {len(it)} người có < {SO_FOLD} ảnh train (nên chụp thêm): "
              + ", ".join(f"{t} ({n})" for t, n in it.items()))

    print("\n3.2 Phân tích PCA")
    k = phan_tich_pca(X_train, y_train)

    # ---------------- 3.3 – 3.5: Thành viên 2 (Nguyễn Văn Lâm) ----------------
    print(f"\n3.3 + 3.4 Huấn luyện hồi quy logistic, chọn tham số bằng GridSearchCV {SO_FOLD}-fold")
    mh, bang = huan_luyen(X_train, y_train, k)

    print("\n3.5 Lưu kết quả")
    ten_mo_hinh = "Logistic Regression"
    os.makedirs(C.MO_HINH, exist_ok=True)
    # người trong lớp (ảnh tự thu): được hiện tên + điểm danh ở bước 6
    trong_lop = df["nguon"].isin(C.TRONG_LOP) & (df["vai_tro"] == "biet")
    nguoi_diem_danh = sorted(df.loc[trong_lop, "ten"].unique())
    joblib.dump({"mo_hinh": mh, "ten_mo_hinh": ten_mo_hinh, "so_thanh_phan_pca": k,
                 "nguoi_diem_danh": nguoi_diem_danh}, C.FILE_HOI_QUY_LOGISTIC, compress=3)
    bang.to_csv(C.FILE_KET_QUA_CV, index=False, encoding="utf-8-sig")
    print(f"  Mô hình đã huấn luyện: {ten_mo_hinh}")
    print("  -> mo_hinh/hoi_quy_logistic.joblib\n  -> ket_qua/3_ket_qua_cv.csv")
    print("\nXong bước 3. Tiếp theo: python buoc4_du_doan.py")


if __name__ == "__main__":
    main()

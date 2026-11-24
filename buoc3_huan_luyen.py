"""
=====================================================================
 BƯỚC 3 – HUẤN LUYỆN MÔ HÌNH
 Phụ trách: 3.1 – 3.2 (HOG + PCA)          Thành viên 1 – Bùi Trung Hiếu
            3.3 – 3.5 (hồi quy logistic)   Thành viên 2 – Nguyễn Văn Lâm
=====================================================================
 Trích đặc trưng HOG (2.916 chiều), chuẩn hóa StandardScaler, dùng PCA để giảm chiều
 (giữ 95% phương sai) và chọn tham số C của hồi quy logistic bằng kiểm tra chéo 5-fold
 (tối ưu theo log_loss để xác suất predict_proba phản ánh đúng độ tự tin).
 Đầu ra: mô hình đã học (mo_hinh/hoi_quy_logistic.joblib), kết quả CV (ket_qua/3_ket_qua_cv.csv)
         và các biểu đồ PCA + chọn tham số C (ket_qua/3_*.png).
=====================================================================
"""
import os
import time
import tempfile

import numpy as np
import pandas as pd
import joblib                                                   # lưu / nạp mô hình ra file
from sklearn.preprocessing import StandardScaler, FunctionTransformer                # chuẩn hóa z = (x - mean) / std
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
    """Phân tích PCA trên đặc trưng HOG để chọn số thành phần chính k (giữ 95% phương sai)
    và kết hợp PCA trên ảnh pixel gốc để vẽ Eigenfaces + khôi phục ảnh minh họa."""
    # PCA trên pixel gốc (10.000 chiều): dùng để vẽ mặt trung bình, Eigenfaces và khôi phục ảnh
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    pca = PCA(svd_solver="full").fit(Xs)
    ti_le = pca.explained_variance_ratio_

    # PCA trên đặc trưng HOG (2.916 chiều): dùng trong pipeline huấn luyện thực tế
    X_hog = C.trich_hog(X_train)
    Xs_hog = StandardScaler().fit_transform(X_hog)
    pca_hog = PCA(svd_solver="full", random_state=C.RANDOM_STATE).fit(Xs_hog)
    ti_le_hog = pca_hog.explained_variance_ratio_
    tich_luy_hog = np.cumsum(ti_le_hog)
    k = min(int(np.searchsorted(tich_luy_hog, GIU_PHUONG_SAI) + 1),
            int(len(X_train) * (1 - 1 / SO_FOLD)) - 1)              # không vượt số ảnh của 1 fold
    print(f"  Ảnh {X_train.shape[1]} pixel -> HOG {X_hog.shape[1]} chiều -> cần k = {k} thành phần chính "
          f"để giữ {GIU_PHUONG_SAI:.0%} phương sai")

    # (a) Biểu đồ kết hợp: phương sai từng thành phần (cột) + tích lũy (đường) của PCA trên HOG
    m, so_cot = min(len(ti_le_hog), max(60, k + 10)), min(50, len(ti_le_hog))
    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    ax1.bar(np.arange(1, so_cot + 1), ti_le_hog[:so_cot] * 100, color="steelblue")
    ax1.set(xlabel="Thành phần chính (PC)", ylabel="Phương sai từng thành phần (%)")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, m + 1), tich_luy_hog[:m] * 100, color="red", marker=".")
    ax2.axhline(GIU_PHUONG_SAI * 100, color="gray", ls="--")
    ax2.axvline(k, color="gray", ls=":")
    ax2.set(ylabel="Phương sai tích lũy (%)", ylim=(0, 102))
    plt.title(f"PCA trên HOG: {k} thành phần đầu giữ {tich_luy_hog[k - 1] * 100:.1f}% phương sai")
    C.luu_hinh("3_pca_phuong_sai.png")

    # (b) Mặt trung bình + 15 eigenfaces
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

    # (c) Trước / sau PCA trên HOG: 2 đặc trưng HOG đầu vs 2 thành phần chính đầu (PC1, PC2)
    Z, Z_hog = pca.transform(Xs), pca_hog.transform(Xs_hog)
    so = pd.factorize(y_train)[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, D, xl, yl, td in (
        (ax[0], Xs_hog, "Đặc trưng HOG 1", "Đặc trưng HOG 2", "Trước PCA: 2 đặc trưng HOG đầu (đã chuẩn hóa)"),
        (ax[1], Z_hog, "Principal Component 1", "Principal Component 2", "Sau PCA: chiếu lên PC1, PC2 (từ HOG)"),
    ):
        sc = a.scatter(D[:, 0], D[:, 1], c=so, cmap="tab20", edgecolor="k", s=15)
        a.set(xlabel=xl, ylabel=yl, title=td)
    fig.colorbar(sc, ax=ax[1], label="Người (mã số)")
    C.luu_hinh("3_truoc_sau_pca.png")

    # (d) Khôi phục ảnh từ m thành phần (PCA pixel): x ≈ mean + z_1*PC1 + ... + z_m*PCm
    cac_m = sorted({min(10, k), min(50, k), k})
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
    """Huấn luyện Pipeline (HOG -> StandardScaler -> PCA -> LogisticRegression)
    và chọn mức điều chuẩn C bằng kiểm tra chéo 5-fold (chọn mô hình có log_loss nhỏ nhất)."""
    cv = StratifiedKFold(SO_FOLD, shuffle=True, random_state=C.RANDOM_STATE)
    bat_dau, cac_c = time.perf_counter(), [0.001, 0.01, 0.1, 1]
    with tempfile.TemporaryDirectory(prefix="cache_pca_") as bo_nho:
        pipeline = Pipeline([
            ("hog", FunctionTransformer(C.trich_hog)),                      # trích đặc trưng HOG (2.916 chiều)
            ("chuan_hoa", StandardScaler()),                                # chuẩn hóa z = (x - mean) / std
            ("pca", PCA(n_components=k, random_state=C.RANDOM_STATE)),      # giảm chiều về k thành phần chính
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ], memory=bo_nho)
        tim = GridSearchCV(pipeline, {"model__C": cac_c}, cv=cv,
                           scoring={"accuracy": "accuracy", "neg_log_loss": "neg_log_loss"},
                           refit="neg_log_loss", n_jobs=1).fit(X_train, y_train)
        mh, idx = tim.best_estimator_.set_params(memory=None), tim.best_index_
        cv_acc = tim.cv_results_["mean_test_accuracy"] * 100
        cv_std = tim.cv_results_["std_test_accuracy"] * 100
        cv_loss = -tim.cv_results_["mean_test_neg_log_loss"]

    c_tot = tim.best_params_["model__C"]
    print("  Kết quả CV 5-fold theo tham số điều chuẩn C:")
    print(pd.DataFrame({"C": cac_c, "cv_%": np.round(cv_acc, 2),
                        "cv_std_%": np.round(cv_std, 2), "log_loss": np.round(cv_loss, 4)}).to_string(index=False))

    # Vẽ biểu đồ chọn tham số C (Accuracy và Log-loss)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(cac_c, cv_acc, yerr=cv_std, marker="o", color="steelblue", capsize=4, lw=2)
    ax[1].plot(cac_c, cv_loss, marker="s", color="darkorange", lw=2)
    for a, yl, td, nhan in (
        (ax[0], "Độ chính xác CV (%)", "Độ chính xác CV 5-fold theo C", f"Chọn C = {c_tot}"),
        (ax[1], "Log-loss (càng nhỏ càng tốt)", "Log-loss CV 5-fold theo C",
         f"Chọn C = {c_tot} (log_loss = {cv_loss[idx]:.4f})"),
    ):
        a.axvline(c_tot, color="tomato", ls="--", label=nhan)
        a.set_xscale("log")
        a.set(xlabel="Tham số C (thang log)", ylabel=yl, title=td)
        a.grid(True, alpha=0.3)
        a.legend()
    ax[0].set_ylim(max(0, cv_acc.min() - 5), 102)
    plt.suptitle("Chọn tham số điều chuẩn C cho Hồi quy logistic (GridSearchCV 5-fold)")
    C.luu_hinh("3_chon_tham_so_C.png")

    bang = pd.DataFrame([{
        "mo_hinh": "Logistic Regression", "tham_so": str({"C": c_tot}),
        "cv_%": round(float(cv_acc[idx]), 2), "cv_std_%": round(float(cv_std[idx]), 2),
        "log_loss": round(float(cv_loss[idx]), 4), "thoi_gian_s": round(time.perf_counter() - bat_dau, 1),
    }])
    print("\n  Cấu hình được chọn:\n" + bang.to_string(index=False))
    return mh, bang


def main():
    print("=" * 70 + "\n BƯỚC 3 – HUẤN LUYỆN MÔ HÌNH\n" + "=" * 70)

    # ---------------- 3.1 – 3.2: Thành viên 1 (Bùi Trung Hiếu) ----------------
    print("\n3.1 Lấy dữ liệu train")
    X, df = C.doc_du_lieu()
    if "tap" not in df:
        raise SystemExit("Chưa chia dữ liệu. Chạy buoc2_chia_du_lieu.py trước.")
    la_train = (df["tap"] == "train").to_numpy()
    X_train, y_train = X[la_train], df.loc[la_train, "ten"].to_numpy()
    print(f"  X_train: {X_train.shape[0]} ảnh x {X_train.shape[1]} chiều | {len(set(y_train))} người")
    it = pd.Series(y_train).value_counts()
    it = it[it < SO_FOLD]
    if len(it):
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
    joblib.dump({"mo_hinh": mh, "ten_mo_hinh": ten_mo_hinh, "so_thanh_phan_pca": k,
                 "nguoi_diem_danh": sorted(df["ten"].unique())}, C.FILE_HOI_QUY_LOGISTIC, compress=3)
    bang.to_csv(C.FILE_KET_QUA_CV, index=False, encoding="utf-8-sig")
    print(f"  Mô hình đã huấn luyện: {ten_mo_hinh}")
    print("  -> mo_hinh/hoi_quy_logistic.joblib\n  -> ket_qua/3_ket_qua_cv.csv")
    print("\nXong bước 3. Tiếp theo: python buoc4_du_doan.py")


if __name__ == "__main__":
    main()

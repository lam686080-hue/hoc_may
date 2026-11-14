"""
=====================================================================
 BƯỚC 3 – HUẤN LUYỆN MÔ HÌNH                              (Thành viên 2)
=====================================================================
 Làm gì   : Giảm chiều ảnh bằng PCA rồi huấn luyện 7 mô hình của học phần,
            mỗi mô hình tự chọn tham số tốt nhất bằng cross-validation 5-fold
            TRÊN TẬP TRAIN (tập test không được dùng ở bước này).
 Mỗi mô hình là 1 Pipeline 3 khâu:
     StandardScaler (chuẩn hóa)  ->  PCA (giảm 10.000 chiều)  ->  bộ phân lớp
 7 bộ phân lớp: Logistic Regression, KNN, SVM linear / poly / rbf, Naive Bayes, Decision Tree
 Đầu vào  : dataset/du_lieu_anh.npz, dataset/danh_sach.csv (bước 1, 2)
 Đầu ra   : mo_hinh/cac_mo_hinh.joblib   7 mô hình đã huấn luyện
            ket_qua/3_ket_qua_cv.csv      tham số tốt nhất + độ chính xác CV từng mô hình
            ket_qua/3_*.png               biểu đồ PCA và kết quả huấn luyện
 Các bước : 3.1 Lấy dữ liệu train
            3.2 Phân tích PCA: chọn số thành phần k giữ 95% phương sai
            3.3 Tạo 7 mô hình + lưới tham số
            3.4 Huấn luyện + chọn tham số bằng GridSearchCV (5-fold)
            3.5 Vẽ kết quả, lưu mô hình
 Chạy     : python buoc3_huan_luyen.py
=====================================================================
"""
import os
import time
import shutil
import tempfile

import numpy as np
import pandas as pd
import joblib                                                   # lưu / nạp mô hình ra file
from sklearn.preprocessing import StandardScaler                # chuẩn hóa z = (x - mean) / std
from sklearn.decomposition import PCA                           # phân tích thành phần chính
from sklearn.pipeline import Pipeline                           # ghép nhiều khâu thành 1 mô hình
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
import chung as C
from chung import plt

GIU_PHUONG_SAI = 0.95      # PCA giữ 95% phương sai
SO_FOLD = 5                # cross-validation 5-fold
try:
    import seaborn as sns  # vẽ bản đồ nhiệt (heatmap)
except ImportError:
    sns = None


# ======================= 3.2 PHÂN TÍCH PCA =======================
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
    for i in range(15):
        ax[i + 1].imshow(pca.components_[i].reshape(C.KICH_THUOC, C.KICH_THUOC), cmap="gray")
        ax[i + 1].set_title(f"PC{i + 1} ({ti_le[i] * 100:.1f}%)", fontsize=8)
    for a in ax:
        a.axis("off")
    plt.suptitle("Eigenfaces: các vector riêng của tập khuôn mặt")
    C.luu_hinh("3_eigenfaces.png")

    # (c) Trước / sau PCA (như slide PCA): 2 pixel gốc vs 2 thành phần chính đầu
    Z = pca.transform(Xs)
    nguoi = sorted(set(y_train))
    mau = C.mau_lop(len(nguoi))
    p1, p2 = 46 * C.KICH_THUOC + 34, 46 * C.KICH_THUOC + 66          # 2 pixel ở vùng mắt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, n in enumerate(nguoi):
        c = y_train == n
        ax[0].scatter(Xs[c, p1], Xs[c, p2], color=mau[i], s=12, edgecolor="k", lw=0.2)
        ax[1].scatter(Z[c, 0], Z[c, 1], color=mau[i], s=12, edgecolor="k", lw=0.2,
                      label=C.rut_gon(n))
    ax[0].set(title="Trước PCA: 2 pixel gốc (đã chuẩn hóa)",
              xlabel="Pixel mắt trái", ylabel="Pixel mắt phải")
    ax[1].set(title="Sau PCA: chiếu lên PC1, PC2",
              xlabel="Principal Component 1", ylabel="Principal Component 2")
    if len(nguoi) <= 12:
        ax[1].legend(fontsize=7, markerscale=1.5)
    else:                                                       # nhiều người: chú thích ra ngoài
        ax[1].legend(fontsize=6, markerscale=1.2, loc="upper left", bbox_to_anchor=(1.01, 1),
                     ncol=int(np.ceil(len(nguoi) / 25)))
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


# ======================= 3.3 TẠO 7 MÔ HÌNH =======================
def tao_mo_hinh(k, bo_nho):
    """Mỗi mô hình = Pipeline [StandardScaler -> PCA(k) -> bộ phân lớp] + lưới tham số để thử.
    Pipeline bảo đảm StandardScaler và PCA chỉ học từ dữ liệu train."""
    def ong(bo_phan_lop):
        return Pipeline([("chuan_hoa", StandardScaler()),
                         ("pca", PCA(n_components=k, random_state=C.RANDOM_STATE)),
                         ("model", bo_phan_lop)],
                        memory=bo_nho)          # nhớ kết quả PCA, các mô hình sau không tính lại
    rs = C.RANDOM_STATE
    # class_weight="balanced": người có ít ảnh được tăng trọng số để không bị lấn át
    # probability=True: SVM tính được xác suất predict_proba (cần cho bước 4, 5)
    # gamma SVM rbf: sau chuẩn hóa + PCA, mặc định "scale" ≈ 1/9500 ≈ 1e-4 -> thử quanh đó
    return {
        "Logistic Regression": (ong(LogisticRegression(max_iter=5000, class_weight="balanced")),
                                {"model__C": [0.001, 0.01, 0.1, 1]}),
        "KNN": (ong(KNeighborsClassifier(weights="distance")),
                {"model__n_neighbors": [1, 3, 5, 7]}),
        "SVM linear": (ong(SVC(kernel="linear", probability=True, class_weight="balanced",
                               random_state=rs)),
                       {"model__C": [0.0001, 0.001, 0.01, 0.1]}),
        "SVM poly": (ong(SVC(kernel="poly", gamma="scale", probability=True,
                             class_weight="balanced", random_state=rs)),
                     {"model__degree": [2, 3], "model__C": [1, 10]}),
        "SVM rbf": (ong(SVC(kernel="rbf", probability=True, class_weight="balanced",
                            random_state=rs)),
                    {"model__C": [1, 10, 100], "model__gamma": [1e-5, 1e-4, 1e-3]}),
        "Naive Bayes": (ong(GaussianNB()),
                        {"model__var_smoothing": [1e-9, 1e-2]}),
        "Decision Tree": (ong(DecisionTreeClassifier(random_state=rs)),
                          {"model__criterion": ["gini", "entropy"],
                           "model__max_depth": [10, 20, None]}),
    }


# ======================= 3.4 HUẤN LUYỆN + GRIDSEARCHCV =======================
def huan_luyen(X_train, y_train, k):
    """GridSearchCV: với mỗi bộ tham số, chia tập train thành 5 phần, lần lượt lấy 1 phần để
    kiểm tra, 4 phần để học -> độ chính xác trung bình. Chọn bộ tốt nhất rồi học lại
    trên cả tập train."""
    bo_nho = tempfile.mkdtemp(prefix="cache_pca_")
    # chia 5 phần (xáo trộn trước), mỗi phần giữ đúng tỉ lệ ảnh của từng người
    cv = StratifiedKFold(SO_FOLD, shuffle=True, random_state=C.RANDOM_STATE)
    mo_hinh, bang = {}, []
    try:
        for ten, (ong, luoi) in tao_mo_hinh(k, bo_nho).items():
            t0 = time.time()
            gs = GridSearchCV(ong, luoi, cv=cv, scoring="accuracy", n_jobs=1)
            gs.fit(X_train, y_train)                                 # thử hết lưới tham số
            mo_hinh[ten] = gs.best_estimator_.set_params(memory=None)  # bộ tham số tốt nhất
            tham_so = {p.replace("model__", ""): v for p, v in gs.best_params_.items()}
            do_lech = gs.cv_results_["std_test_score"][gs.best_index_]  # độ lệch chuẩn giữa 5 fold
            bang.append({"mo_hinh": ten, "tham_so": str(tham_so),
                         "cv_%": round(gs.best_score_ * 100, 2),        # độ chính xác CV trung bình
                         "cv_std_%": round(do_lech * 100, 2),
                         "thoi_gian_s": round(time.time() - t0, 1)})
            if ten == "SVM rbf":                               # giữ kết quả mọi cặp (C, gamma)
                luoi_rbf = pd.DataFrame(gs.cv_results_["params"])
                luoi_rbf["cv"] = gs.cv_results_["mean_test_score"]
            b = bang[-1]
            print(f"  {ten:<20} CV {b['cv_%']:5.1f}% ± {b['cv_std_%']:4.1f} | {b['tham_so']} "
                  f"| {b['thoi_gian_s']}s")
    finally:
        shutil.rmtree(bo_nho, ignore_errors=True)
    return mo_hinh, pd.DataFrame(bang), luoi_rbf


# ======================= 3.5 VẼ KẾT QUẢ =======================
def ve_ket_qua(bang, luoi_rbf, mo_hinh):
    # (a) Độ chính xác CV (cột) ± độ lệch chuẩn giữa 5 fold (thanh sai số)
    plt.figure(figsize=(11, 4.6))
    tot = bang["cv_%"].max()
    mau = ["tomato" if v == tot else "steelblue" for v in bang["cv_%"]]        # đỏ = tốt nhất
    cot = plt.bar(bang["mo_hinh"], bang["cv_%"], yerr=bang["cv_std_%"], capsize=5,
                  edgecolor="black", color=mau)
    for c, v in zip(cot, bang["cv_%"]):
        plt.text(c.get_x() + c.get_width() / 2, 2, f"{v:.1f}%", ha="center", color="white",
                 fontweight="bold")
    plt.ylim(0, 105)
    plt.ylabel("Độ chính xác CV (%)")
    plt.xticks(rotation=15)
    plt.title(f"Độ chính xác {SO_FOLD}-fold cross-validation trên tập train (đỏ = tốt nhất)")
    C.luu_hinh("3_cv_cac_mo_hinh.png")

    # (b) SVM rbf: độ chính xác CV theo C và gamma (bản đồ nhiệt)
    M = luoi_rbf.pivot(index="model__C", columns="model__gamma", values="cv") * 100
    plt.figure(figsize=(6, 4))
    if sns is not None:
        sns.heatmap(M, annot=True, fmt=".1f", cmap="coolwarm")
    else:
        plt.imshow(M.values, cmap="coolwarm")
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.title("SVM rbf: độ chính xác CV (%) theo C và gamma")
    C.luu_hinh("3_svm_rbf_C_gamma.png")

    # (c) 3 tầng đầu của cây quyết định (plot_tree như bài giảng)
    cay = mo_hinh["Decision Tree"].named_steps["model"]
    plt.figure(figsize=(22, 8))
    plot_tree(cay, max_depth=2, filled=True, fontsize=8, precision=2,
              feature_names=[f"PC{i + 1}" for i in range(cay.n_features_in_)],
              class_names=[C.rut_gon(c) for c in cay.classes_])
    plt.title(f"Decision Tree – 3 tầng đầu (độ sâu thật = {cay.get_depth()})")
    C.luu_hinh("3_cay_quyet_dinh.png")


# =====================================================================
def main():
    print("=" * 70)
    print(" BƯỚC 3 – HUẤN LUYỆN MÔ HÌNH")
    print("=" * 70)

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

    print(f"\n3.3 + 3.4 Huấn luyện 7 mô hình, chọn tham số bằng GridSearchCV {SO_FOLD}-fold")
    mo_hinh, bang, luoi_rbf = huan_luyen(X_train, y_train, k)

    print("\n3.5 Vẽ kết quả và lưu")
    ve_ket_qua(bang, luoi_rbf, mo_hinh)
    tot_nhat = bang.loc[bang["cv_%"].idxmax(), "mo_hinh"]     # chọn theo CV, KHÔNG theo tập test
    os.makedirs(C.MO_HINH, exist_ok=True)
    # người trong lớp (ảnh tự thu / chụp tay, không phải LFW): được hiện tên + điểm danh ở bước 6
    trong_lop = df["nguon"].isin(C.TRONG_LOP) & (df["vai_tro"] == "biet")
    nguoi_diem_danh = sorted(df.loc[trong_lop, "ten"].unique())
    joblib.dump({"mo_hinh": mo_hinh, "tot_nhat": tot_nhat, "so_thanh_phan_pca": k,
                 "nguoi_diem_danh": nguoi_diem_danh}, C.FILE_CAC_MO_HINH, compress=3)
    bang.to_csv(C.FILE_KET_QUA_CV, index=False, encoding="utf-8-sig")
    print(f"  Mô hình tốt nhất theo cross-validation: {tot_nhat}")
    print("  -> mo_hinh/cac_mo_hinh.joblib\n  -> ket_qua/3_ket_qua_cv.csv")
    print("\nXong bước 3. Tiếp theo: python buoc4_du_doan.py")


if __name__ == "__main__":
    main()

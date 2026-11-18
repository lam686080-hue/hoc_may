"""
=====================================================================
 BƯỚC 5 – ĐÁNH GIÁ MÔ HÌNH
 Phụ trách: Thành viên 3 – Lê Thanh Cao
=====================================================================
 Làm gì   : So kết quả dự đoán (bước 4) với tên thật để đánh giá hồi quy logistic,
            chọn ngưỡng "người lạ", rồi lưu hồi quy logistic cho bước 6.
 Đầu vào  : ket_qua/du_doan.npz (bước 4), ket_qua/3_ket_qua_cv.csv và
            mo_hinh/hoi_quy_logistic.joblib (bước 3), dữ liệu bước 1-2
 Đầu ra   : ket_qua/5_*.png, 5_danh_gia_logistic.csv, 5_bao_cao_phan_loai.txt
            mo_hinh/mo_hinh_tot_nhat.joblib + thong_tin.json   (cho bước 6)
            ket_qua/xem_ket_qua.html     trang xem TẤT CẢ hình của 5 bước
 Các bước : 5.1 Đọc kết quả dự đoán
            5.2 Tính độ đo: accuracy, precision, recall, F1 -> bảng đánh giá
            5.3 Ma trận nhầm lẫn
            5.4 Đường ROC và AUC
            5.5 Phân tích lỗi: theo nguồn ảnh, theo tư thế, từng người, ảnh nhận sai
            5.6 Người lạ: chọn ngưỡng để FAR và FRR gần nhau nhất trên bộ ảnh dùng chọn ngưỡng
            5.7 Học lại hồi quy logistic trên toàn bộ dữ liệu, lưu cho bước 6
            5.8 Tạo trang xem kết quả
 Chạy     : python buoc5_danh_gia.py
=====================================================================
"""
import os
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import seaborn as sns                                   # vẽ ma trận nhầm lẫn (heatmap)
from sklearn.base import clone                          # tạo bản sao mô hình chưa học
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize       # đổi tên người -> dạng 0/1 cho ROC
import chung as C
from chung import plt

rng = np.random.default_rng(C.RANDOM_STATE)


def ve_cot(nhan, gia_tri, mau, ten_file, tieu_de):
    """Biểu đồ cột độ chính xác (%), ghi số lên đầu cột."""
    plt.figure(figsize=(7, 4))
    plt.bar_label(plt.bar(nhan, gia_tri, color=mau, edgecolor="black"), fmt="%.0f%%")
    plt.ylim(0, 110)
    plt.ylabel("Độ chính xác (%)")
    plt.title(tieu_de)
    C.luu_hinh(ten_file)


# ======================= 5.2 ĐỘ ĐO + BẢNG SO SÁNH =======================
def tinh_do_do(ten, du_doan, y_test, ms, cv):
    acc = accuracy_score(y_test, du_doan)
    p, r, f, _ = precision_recall_fscore_support(y_test, du_doan, average="macro",
                                                zero_division=0)
    diem = cv.loc[cv["mo_hinh"] == ten].iloc[0]
    bang = pd.DataFrame([{"mo_hinh": ten, "tham_so": diem["tham_so"], "cv_%": diem["cv_%"],
                          "test_%": round(acc * 100, 2), "precision": round(p, 3),
                          "recall": round(r, 3), "f1": round(f, 3), "ms_moi_anh": round(ms, 3)}])
    print(bang.to_string(index=False))
    bang.to_csv(os.path.join(C.KET_QUA, "5_danh_gia_logistic.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(C.KET_QUA, "5_bao_cao_phan_loai.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, du_doan, zero_division=0))
    # Biểu đồ cột nhóm: CV (trên train) cạnh test cho hồi quy logistic, kèm thời gian dự đoán
    ax = bang.set_index("mo_hinh")[["cv_%", "test_%"]].plot.bar(
        figsize=(12, 4.8), color=["steelblue", "darkorange"], edgecolor="black", rot=0)
    for cot in ax.containers:
        ax.bar_label(cot, fmt="%.0f", fontsize=8)
    ax.set_xticklabels([f"{t}\n{m:.2f} ms/ảnh" for t, m in zip(bang["mo_hinh"],
                                                                bang["ms_moi_anh"])], fontsize=8)
    ax.legend(["Cross-validation (tập train)", "Tập test"], loc="upper center", ncol=2)
    ax.set(xlabel="", ylabel="Độ chính xác (%)", ylim=(0, 122), title="Đánh giá hồi quy logistic")
    C.luu_hinh("5_danh_gia_logistic.png")
    return bang


# ======================= 5.3 MA TRẬN NHẦM LẪN =======================
def ve_ma_tran(ax, cm, nhan, tieu_de, lon=False):
    """Heatmap ma trận nhầm lẫn như slide: annot=True, fmt='d', cmap='Blues'."""
    sns.heatmap(cm, annot=len(nhan) <= (30 if lon else 15), fmt="d", cmap="Blues", ax=ax,
                xticklabels=nhan, yticklabels=nhan, cbar=lon, annot_kws={"size": 9 if lon else 6})
    ax.set(xlabel="Predicted Label (dự đoán)", ylabel="True Label (thực tế)", title=tieu_de)
    ax.tick_params(labelsize=9 if lon else 7)


def ve_ma_tran_logistic(du_doan, y_test, lop, do_chinh_xac):
    nhan = [C.rut_gon(n) for n in lop]
    co = min(len(lop), 26)                                        # giới hạn cỡ hình khi nhiều người
    fig, ax = plt.subplots(figsize=(co + 3, 0.85 * co + 2.5))
    ve_ma_tran(ax, confusion_matrix(y_test, du_doan, labels=lop), nhan,
               f"Ma trận nhầm lẫn – Hồi quy logistic ({do_chinh_xac:.1f}%)", lon=True)
    C.luu_hinh("5_ma_tran_nham_lan_tot_nhat.png")


# ======================= 5.4 ROC – AUC =======================
def ve_roc(y_test, P_test, lop, ten_mo_hinh):
    """ROC one-vs-rest: với mỗi người, coi 'là người đó' là lớp dương, còn lại là lớp âm."""
    Y = label_binarize(y_test, classes=lop)
    if Y.shape[1] == 1:                                           # chỉ 2 người: thêm cột còn lại
        Y = np.hstack([1 - Y, Y])
    plt.figure(figsize=(7.5, 6))
    for i, n in enumerate(lop):
        if 0 < Y[:, i].sum() < len(Y):                            # người có cả ảnh dương và âm
            fpr, tpr, _ = roc_curve(Y[:, i], P_test[:, i])        # TPR, FPR khi thay đổi ngưỡng
            plt.plot(fpr, tpr, lw=1.2, label=f"{C.rut_gon(n)} (AUC={auc(fpr, tpr):.3f})"
                     if len(lop) <= 15 else None)
    fpr, tpr, _ = roc_curve(Y.ravel(), P_test.ravel())           # gộp tất cả: trung bình micro
    auc_tb = auc(fpr, tpr)
    plt.plot(fpr, tpr, "k--", lw=2, label=f"Trung bình micro (AUC={auc_tb:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", ls=":", label="Ngẫu nhiên")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (one-vs-rest) – {ten_mo_hinh}")
    plt.legend(fontsize=7, loc="lower right")
    C.luu_hinh("5_roc_tot_nhat.png")
    print(f"  AUC trung bình (micro) = {auc_tb:.3f}")


# ======================= 5.5 PHÂN TÍCH LỖI =======================
def phan_tich_loi(X_test, y_test, y_doan, tu_the, nguon, ten_mo_hinh):
    # (a) độ chính xác theo nguồn ảnh: tự thu / chụp tay / LFW
    ds = [n for n in C.NGUON if (nguon == n).any()]
    acc = [accuracy_score(y_test[nguon == n], y_doan[nguon == n]) * 100 for n in ds]
    print("  Theo nguồn ảnh: " + ", ".join(f"{C.NGUON[n][0]} {a:.0f}%" for n, a in zip(ds, acc)))
    ve_cot([C.NGUON[n][0] for n in ds], acc, [C.NGUON[n][1] for n in ds], "5_theo_nguon_anh.png",
           f"Độ chính xác trên tập test theo nguồn ảnh – {ten_mo_hinh}")
    # (b) người trong lớp bị nhận đúng ít nhất (nên chụp thêm ảnh cho những người này)
    lop_hoc = np.isin(nguon, C.TRONG_LOP)
    if lop_hoc.any():
        dung = pd.Series(y_doan[lop_hoc] == y_test[lop_hoc]).groupby(y_test[lop_hoc]).mean()
        print("  Người trong lớp nhận đúng ít nhất: "
              + ", ".join(f"{t} {v * 100:.0f}%" for t, v in dung.sort_values().head(5).items()))
    # (c) độ chính xác theo tư thế (chỉ ảnh tự thu mới có tư thế)
    tt = [t for t in C.TEN_TU_THE if ((tu_the == t) & (nguon == "tu_thu")).any()]
    if tt:
        acc = [accuracy_score(y_test[tu_the == t], y_doan[tu_the == t]) * 100 for t in tt]
        print("  Theo tư thế: " + ", ".join(f"{C.TEN_TU_THE[t]} {a:.0f}%" for t, a in zip(tt, acc)))
        ve_cot([C.TEN_TU_THE[t] for t in tt], acc, "seagreen", "5_theo_tu_the.png",
               f"Ảnh tự thu trong tập kiểm tra: độ chính xác theo tư thế – {ten_mo_hinh}")
    # (d) một số ảnh bị nhận sai
    sai = np.where(y_doan != y_test)[0]
    print(f"  Số ảnh nhận sai: {len(sai)}/{len(y_test)}")
    if len(sai):
        chon = rng.choice(sai, size=min(16, len(sai)), replace=False)
        C.ve_luoi_anh(X_test[chon], [f"Thật: {C.rut_gon(t)}" for t in y_test[chon]],
                      [f"Đoán: {C.rut_gon(d)}" for d in y_doan[chon]], [False] * len(chon),
                      "5_anh_nhan_sai.png", f"Một số ảnh bị nhận sai – {ten_mo_hinh}")


# ======================= 5.6 NGƯỠNG NGƯỜI LẠ =======================
def chon_nguong(P_test, dung, P_la, X_la, lop):
    """Xác suất cao nhất < ngưỡng -> "Người lạ".
    FRR = tỉ lệ người QUEN bị từ chối | FAR = tỉ lệ người LẠ bị nhận nhầm. Chọn ngưỡng có chênh lệch FAR và FRR nhỏ nhất."""
    if len(P_la) == 0:
        print("  Không có ảnh người lạ -> dùng ngưỡng mặc định 0.6")
        return 0.6, None, None
    p_quen, p_la = P_test.max(1), P_la.max(1)                  # xác suất cao nhất của mỗi ảnh
    cac_nguong = np.round(np.linspace(0, 1, 101), 2)           # thử 101 ngưỡng 0.00 … 1.00
    frr = np.array([(p_quen < t).mean() for t in cac_nguong])
    far = np.array([(p_la >= t).mean() for t in cac_nguong])
    nhan_dung = np.array([((p_quen >= t) & dung).mean() for t in cac_nguong])   # quen: nhận + đúng
    i = int(np.argmin(np.abs(far - frr)))                      # chỗ 2 đường cắt nhau
    t = float(cac_nguong[i])
    print(f"  Ngưỡng người lạ = {t:.2f}: FAR {far[i] * 100:.1f}% | FRR {frr[i] * 100:.1f}% | "
          f"người quen nhận đúng {nhan_dung[i] * 100:.1f}%")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    kieu = dict(bins=20, range=(0, 1), alpha=0.6)
    ax[0].hist(p_quen[dung], color="seagreen", label="Người quen – đoán đúng", **kieu)
    ax[0].hist(p_quen[~dung], color="tomato", label="Người quen – đoán sai", **kieu)
    ax[0].hist(p_la, color="gray", label="Người lạ", **kieu)
    ax[0].axvline(t, color="k", ls="--", label=f"Ngưỡng {t:.2f}")
    ax[0].set(xlabel="Xác suất cao nhất (predict_proba)", ylabel="Số ảnh",
              title="Phân bố xác suất cao nhất")
    ax[1].plot(cac_nguong, far * 100, color="tomato", lw=2, label="FAR – người lạ bị nhận nhầm")
    ax[1].plot(cac_nguong, frr * 100, color="steelblue", lw=2, label="FRR – người quen bị từ chối")
    ax[1].plot(cac_nguong, nhan_dung * 100, color="seagreen", ls="--", label="Người quen nhận đúng")
    ax[1].axvline(t, color="k", ls=":")
    ax[1].set(xlabel="Ngưỡng", ylabel="%", title=f"Ngưỡng cân bằng FAR và FRR: {t:.2f}")
    for a in ax:
        a.legend(fontsize=8)
    C.luu_hinh("5_nguoi_la.png")
    chon = rng.choice(len(X_la), size=min(16, len(X_la)), replace=False)
    doan = [lop[j] if p >= t else "Người lạ" for j, p in zip(P_la[chon].argmax(1), p_la[chon])]
    C.ve_luoi_anh(X_la[chon], ["Thật: Người lạ"] * len(chon),
                  [f"Đoán: {C.rut_gon(d)} ({p:.0%})" for d, p in zip(doan, p_la[chon])],
                  [d == "Người lạ" for d in doan], "5_nguoi_la_anh.png",
                  f"Ảnh người lạ (không có trong tập huấn luyện) – ngưỡng {t:.2f}")
    return t, float(far[i]), float(frr[i])


# ======================= 5.8 TRANG XEM KẾT QUẢ =======================
MUC = [("1_", "Bước 1 – Chuẩn bị dữ liệu (Thành viên 1 – Bùi Trung Hiếu)"),
       ("2_", "Bước 2 – Chia train/test (Thành viên 2 – Nguyễn Văn Lâm)"),
       (("3_pca", "3_eigenfaces", "3_truoc_sau_pca", "3_khoi_phuc_pca"),
        "Bước 3.2 – Phân tích PCA (Thành viên 1 – Bùi Trung Hiếu)"),
       ("3_", "Bước 3.3 – 3.5 – Huấn luyện hồi quy logistic (Thành viên 2 – Nguyễn Văn Lâm)"),
       ("4_", "Bước 4 – Dự đoán (Thành viên 2 – Nguyễn Văn Lâm)"),
       ("5_", "Bước 5 – Đánh giá (Thành viên 3 – Lê Thanh Cao)")]


def tao_trang_xem(bang, tom_tat):
    """Trang HTML: tóm tắt + bảng đánh giá + mọi hình, chia mục theo bước và người phụ trách."""
    hinh_cu = {"3_cv_cac_mo_hinh.png", "3_svm_rbf_C_gamma.png", "3_cay_quyet_dinh.png",
               "5_ma_tran_nham_lan_cac_mo_hinh.png", "5_so_sanh_mo_hinh.png"}
    con_lai = sorted(f for f in os.listdir(C.KET_QUA) if f.endswith(".png") and f not in hinh_cu)
    h = ["<html><head><meta charset='utf-8'><title>Kết quả nhận diện khuôn mặt</title></head>"
         "<body style='font-family:Arial;max-width:1200px;margin:auto'><h1>Kết quả</h1><p>"
         + "<br>".join(f"<b>{k}:</b> {v}" for k, v in tom_tat.items()) + "</p>",
         bang.to_html(index=False)]                               # pandas đổi bảng thành HTML
    for tien_to, tieu_de in MUC:
        ds = [f for f in con_lai if f.startswith(tien_to)]       # startswith nhận cả tuple
        con_lai = [f for f in con_lai if f not in ds]             # mỗi hình chỉ vào 1 mục
        h += [f"<h2>{tieu_de}</h2>"] + [f"<p><b>{f}</b><br><img src='{f}' style='max-width:100%'>"
                                        for f in ds]
    p = os.path.join(C.KET_QUA, "xem_ket_qua.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(h) + "</body></html>")
    print("  -> ket_qua/xem_ket_qua.html  (mở bằng trình duyệt để xem tất cả hình)")
    if os.name == "nt":                                           # laptop Windows: tự mở trang
        webbrowser.open(Path(p).as_uri())


# =====================================================================
def main():
    print("=" * 70)
    print(" BƯỚC 5 – ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 70)

    print("\n5.1 Đọc kết quả dự đoán (bước 4)")
    if not os.path.exists(C.FILE_DU_DOAN):
        raise SystemExit("Chưa có kết quả dự đoán. Chạy buoc4_du_doan.py trước.")
    d = np.load(C.FILE_DU_DOAN)
    ten_mo_hinh = "Logistic Regression"
    if d["ten_mo_hinh"].shape != () or d["ten_mo_hinh"].item() != ten_mo_hinh:
        raise SystemExit("Hãy chạy lại bước 4 với bản hồi quy logistic.")
    y_doan, ms = d["du_doan"], float(d["ms_moi_anh"])
    lop, y_test = list(d["lop"]), d["y_test"]
    P_test, P_la = d["P_test"], d["P_la"]
    X, df = C.doc_du_lieu()
    vi_tri = {f: i for i, f in enumerate(df["file"])}            # tên file -> số thứ tự hàng
    i_test = np.array([vi_tri[f] for f in d["file_test"]], dtype=int)
    i_la = np.array([vi_tri[f] for f in d["file_la"]], dtype=int)
    cv = pd.read_csv(C.FILE_KET_QUA_CV, encoding="utf-8-sig")
    print(f"  {ten_mo_hinh} | {len(y_test)} ảnh kiểm tra | {len(i_la)} ảnh người lạ")

    print("\n5.2 Đánh giá trên tập kiểm tra")
    bang = tinh_do_do(ten_mo_hinh, y_doan, y_test, ms, cv)
    print(f"\n  Báo cáo chi tiết – {ten_mo_hinh}:")
    print(classification_report(y_test, y_doan, zero_division=0))

    print("5.3 Ma trận nhầm lẫn")
    ve_ma_tran_logistic(y_doan, y_test, lop, bang["test_%"].iloc[0])

    print("\n5.4 Đường ROC và AUC")
    ve_roc(y_test, P_test, lop, ten_mo_hinh)

    print("\n5.5 Phân tích lỗi")
    phan_tich_loi(X[i_test], y_test, y_doan, df["tu_the"].to_numpy()[i_test],
                  df["nguon"].to_numpy()[i_test], ten_mo_hinh)

    print("\n5.6 Chọn ngưỡng người lạ")
    dung = np.array(lop)[P_test.argmax(1)] == y_test               # ảnh người quen đoán đúng
    nguong, far, frr = chon_nguong(P_test, dung, P_la, X[i_la], lop)

    print("\n5.7 Học lại hồi quy logistic trên toàn bộ ảnh người quen, lưu cho bước 6")
    goi = joblib.load(C.FILE_HOI_QUY_LOGISTIC)
    biet = (df["vai_tro"] == "biet").to_numpy()
    mh = clone(goi["mo_hinh"])                    # cùng tham số tốt nhất, chưa học
    mh.fit(X[biet], df.loc[biet, "ten"].to_numpy())          # học trên train + test (hết dữ liệu)
    acc = float(bang.loc[bang["mo_hinh"] == ten_mo_hinh, "test_%"].iloc[0])
    joblib.dump({"mo_hinh": mh, "ten_mo_hinh": ten_mo_hinh, "nguoi_diem_danh": goi["nguoi_diem_danh"],
                 "nguong_nguoi_la": nguong, "do_chinh_xac_test": acc},
                C.FILE_MO_HINH_TOT, compress=3)
    tom_tat = {"Mô hình": ten_mo_hinh,
               "Tham số": bang.loc[bang["mo_hinh"] == ten_mo_hinh, "tham_so"].iloc[0],
               "Độ chính xác test (%)": acc, "Số thành phần PCA": goi["so_thanh_phan_pca"],
               "Ngưỡng người lạ": nguong,
               "FAR / FRR trên ảnh dùng chọn ngưỡng (%)": "-" if far is None
               else f"{far * 100:.1f} / {frr * 100:.1f}",
               "Người được điểm danh (trong lớp)": ", ".join(goi["nguoi_diem_danh"]) or "-"}
    with open(os.path.join(C.MO_HINH, "thong_tin.json"), "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in tom_tat.items()}, f, ensure_ascii=False, indent=2)
    print(f"  {ten_mo_hinh}: học lại trên {int(biet.sum())} ảnh -> mo_hinh/mo_hinh_tot_nhat.joblib")

    print("\n5.8 Tạo trang xem kết quả")
    tao_trang_xem(bang, tom_tat)
    print("\nXong bước 5. Tiếp theo: python buoc6_nhan_dien.py (hoặc --web trên Jetson)")


if __name__ == "__main__":
    main()

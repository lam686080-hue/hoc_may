"""
=====================================================================
 BƯỚC 5 – ĐÁNH GIÁ MÔ HÌNH                                 (Thành viên 3)
=====================================================================
 Làm gì   : So kết quả dự đoán (bước 4) với tên thật để đánh giá 7 mô hình,
            chọn ngưỡng "người lạ", rồi lưu mô hình tốt nhất cho bước 6.
 Đầu vào  : ket_qua/du_doan.npz (bước 4), ket_qua/3_ket_qua_cv.csv và
            mo_hinh/cac_mo_hinh.joblib (bước 3), dữ liệu bước 1-2
 Đầu ra   : ket_qua/5_*.png, 5_so_sanh_mo_hinh.csv, 5_bao_cao_phan_loai.txt
            mo_hinh/mo_hinh_tot_nhat.joblib + thong_tin.json   (cho bước 6)
            ket_qua/xem_ket_qua.html     trang xem TẤT CẢ hình của 5 bước
 Các bước : 5.1 Đọc kết quả dự đoán
            5.2 Tính độ đo: accuracy, precision, recall, F1 -> bảng so sánh
            5.3 Ma trận nhầm lẫn
            5.4 Đường ROC và AUC
            5.5 Phân tích lỗi: theo tư thế, theo nguồn ảnh, từng người, ảnh nhận sai
            5.6 Người lạ: chọn ngưỡng tại FAR = FRR
            5.7 Học lại mô hình tốt nhất trên toàn bộ dữ liệu, lưu cho bước 6
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
from sklearn.base import clone                          # tạo bản sao mô hình chưa học
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize       # đổi nhãn tên người -> dạng 0/1 cho ROC
import chung as C
from chung import plt

try:
    import seaborn as sns
except ImportError:
    sns = None
rng = np.random.default_rng(C.RANDOM_STATE)


def ve_ma_tran(ax, cm, nhan, tieu_de, lon=False):
    """Vẽ ma trận nhầm lẫn bằng heatmap (như bài giảng: annot=True, fmt='d', cmap='Blues')."""
    ghi_so = len(nhan) <= (30 if lon else 15)          # quá nhiều người thì không ghi số vào ô
    if sns is not None:
        sns.heatmap(cm, annot=ghi_so, fmt="d", cmap="Blues", xticklabels=nhan, yticklabels=nhan,
                    cbar=lon, ax=ax, annot_kws={"size": 9 if lon else 6})
    else:
        ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted Label (dự đoán)", fontsize=9)
    ax.set_ylabel("True Label (thực tế)", fontsize=9)
    ax.set_title(tieu_de, fontsize=12 if lon else 10)
    ax.tick_params(labelsize=9 if lon else 7)


# ======================= 5.2 ĐỘ ĐO + BẢNG SO SÁNH =======================
def tinh_do_do(ten_mh, du_doan, y_test, ms, cv):
    bang, bao_cao = [], []
    for i, ten in enumerate(ten_mh):
        acc = accuracy_score(y_test, du_doan[i])                       # tỉ lệ đoán đúng
        # precision, recall, F1 của từng người rồi lấy trung bình (macro)
        p, r, f, _ = precision_recall_fscore_support(y_test, du_doan[i], average="macro",
                                                     zero_division=0)
        dong_cv = cv[cv["mo_hinh"] == ten].iloc[0]
        bang.append({"mo_hinh": ten, "tham_so": dong_cv["tham_so"], "cv_%": dong_cv["cv_%"],
                     "test_%": round(acc * 100, 2), "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(f, 3), "ms_moi_anh": round(float(ms[i]), 3)})
        bao_cao.append(f"===== {ten}  (test {acc * 100:.2f}%) =====\n"
                       + classification_report(y_test, du_doan[i], zero_division=0))
        print(f"  {ten:<20} test {acc * 100:5.1f}% | CV {dong_cv['cv_%']:5.1f}% | "
              f"P {p:.3f} R {r:.3f} F1 {f:.3f}")
    bang = pd.DataFrame(bang)
    bang.to_csv(os.path.join(C.KET_QUA, "5_so_sanh_mo_hinh.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(C.KET_QUA, "5_bao_cao_phan_loai.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(bao_cao))
    # Biểu đồ cột nhóm: CV (trên train) cạnh test cho từng mô hình
    x = np.arange(len(bang))
    plt.figure(figsize=(12, 4.8))
    c1 = plt.bar(x - 0.2, bang["cv_%"], 0.4, label="Cross-validation (tập train)",
                 color="steelblue", edgecolor="black")
    c2 = plt.bar(x + 0.2, bang["test_%"], 0.4, label="Tập test",
                 color="darkorange", edgecolor="black")
    for c in list(c1) + list(c2):
        plt.text(c.get_x() + c.get_width() / 2, c.get_height() + 1, f"{c.get_height():.0f}",
                 ha="center", fontsize=8)
    nhan = [f"{t}\n{m:.2f} ms/ảnh" for t, m in zip(bang["mo_hinh"], bang["ms_moi_anh"])]
    plt.xticks(x, nhan, fontsize=8)
    plt.ylim(0, 122)
    plt.ylabel("Độ chính xác (%)")
    plt.title("So sánh các mô hình")
    plt.legend(loc="upper center", ncol=2)
    C.luu_hinh("5_so_sanh_mo_hinh.png")
    return bang


# ======================= 5.3 MA TRẬN NHẦM LẪN =======================
def ve_cac_ma_tran(ten_mh, du_doan, y_test, lop, bang, tot_nhat):
    nhan = [C.rut_gon(n) for n in lop]
    hang = int(np.ceil(len(ten_mh) / 4))
    fig, ax = plt.subplots(hang, 4, figsize=(21.6, 4.8 * hang), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, (i, ten) in zip(ax.ravel(), enumerate(ten_mh)):
        a.axis("on")
        cm = confusion_matrix(y_test, du_doan[i], labels=lop)          # hàng: thật, cột: dự đoán
        ve_ma_tran(a, cm, nhan, f"{ten} ({bang['test_%'][i]:.1f}%)")
    C.luu_hinh("5_ma_tran_nham_lan_cac_mo_hinh.png")
    i = ten_mh.index(tot_nhat)
    co = min(len(lop), 26)                                      # giới hạn cỡ hình khi nhiều người
    fig, ax = plt.subplots(figsize=(co + 3, 0.85 * co + 2.5))
    ve_ma_tran(ax, confusion_matrix(y_test, du_doan[i], labels=lop), nhan,
               f"Confusion Matrix – {tot_nhat} ({bang['test_%'][i]:.1f}%)", lon=True)
    C.luu_hinh("5_ma_tran_nham_lan_tot_nhat.png")


# ======================= 5.4 ROC – AUC =======================
def ve_roc(y_test, P_test, lop, tot_nhat):
    """ROC one-vs-rest: với mỗi người, coi 'là người đó' là lớp dương, còn lại là lớp âm."""
    Y = label_binarize(y_test, classes=lop)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    plt.figure(figsize=(7.5, 6))
    mau = C.mau_lop(len(lop))
    it_nguoi = len(lop) <= 15                    # nhiều người quá: vẽ mờ, không ghi chú từng người
    for i, n in enumerate(lop):
        if Y[:, i].any() and not Y[:, i].all():
            fpr, tpr, _ = roc_curve(Y[:, i], P_test[:, i])            # TPR, FPR khi thay đổi ngưỡng
            if it_nguoi:
                plt.plot(fpr, tpr, color=mau[i], lw=1.4,
                         label=f"{C.rut_gon(n)} (AUC={auc(fpr, tpr):.3f})")
            else:
                plt.plot(fpr, tpr, color="lightgray", lw=0.8)
    fpr, tpr, _ = roc_curve(Y.ravel(), P_test.ravel())
    auc_tb = auc(fpr, tpr)
    plt.plot(fpr, tpr, "k--", lw=2, label=f"Trung bình micro (AUC={auc_tb:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", ls=":", label="Ngẫu nhiên")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (one-vs-rest) – {tot_nhat}")
    plt.legend(fontsize=7, loc="lower right")
    C.luu_hinh("5_roc_tot_nhat.png")
    print(f"  AUC trung bình (micro) = {auc_tb:.3f}")


# ======================= 5.5 PHÂN TÍCH LỖI =======================
def phan_tich_loi(X_test, y_test, y_doan, P_test, tu_the, nguon, tot_nhat):
    # (a) độ chính xác theo nguồn ảnh: tự thu / chụp tay / LFW
    ds_nguon = [n for n in C.NGUON if (nguon == n).any()]
    acc_nguon = [accuracy_score(y_test[nguon == n], y_doan[nguon == n]) * 100 for n in ds_nguon]
    so_anh = [int((nguon == n).sum()) for n in ds_nguon]
    print("  Theo nguồn ảnh: " + ", ".join(f"{C.NGUON[n][0]} {a:.0f}% ({s} ảnh)"
                                          for n, a, s in zip(ds_nguon, acc_nguon, so_anh)))
    plt.figure(figsize=(7, 4))
    cot = plt.bar([C.NGUON[n][0] for n in ds_nguon], acc_nguon, edgecolor="black",
                  color=[C.NGUON[n][1] for n in ds_nguon])
    for c, v, s in zip(cot, acc_nguon, so_anh):
        plt.text(c.get_x() + c.get_width() / 2, v + 1, f"{v:.0f}%\n({s} ảnh)", ha="center",
                 fontsize=9)
    plt.ylim(0, 118)
    plt.ylabel("Độ chính xác (%)")
    plt.title(f"Độ chính xác trên tập test theo nguồn ảnh – {tot_nhat}")
    C.luu_hinh("5_theo_nguon_anh.png")
    # (b) người trong lớp bị nhận đúng ít nhất (nên chụp thêm ảnh cho những người này)
    lop_hoc = np.isin(nguon, C.TRONG_LOP)
    if lop_hoc.any():
        bang = pd.DataFrame({"ten": y_test[lop_hoc], "dung": y_doan[lop_hoc] == y_test[lop_hoc]})
        tung_nguoi = bang.groupby("ten")["dung"].agg(["mean", "size"]).sort_values("mean")
        print("  Người trong lớp nhận đúng ít nhất: " + ", ".join(
            f"{t} {r['mean'] * 100:.0f}% ({int(r['size'])} ảnh test)"
            for t, r in tung_nguoi.head(5).iterrows()))
    # (c) độ chính xác theo tư thế (chỉ ảnh thu_anh.py mới có tư thế)
    tt = [t for t in C.TEN_TU_THE if ((tu_the == t) & (nguon == "tu_thu")).any()]
    if tt:                                   # độ chính xác theo tư thế (chỉ ảnh tự thu có tư thế)
        acc = [accuracy_score(y_test[(tu_the == t) & (nguon == "tu_thu")],
                              y_doan[(tu_the == t) & (nguon == "tu_thu")]) * 100 for t in tt]
        plt.figure(figsize=(7, 4))
        cot = plt.bar([C.TEN_TU_THE[t] for t in tt], acc, color="seagreen", edgecolor="black")
        for c, v in zip(cot, acc):
            plt.text(c.get_x() + c.get_width() / 2, v + 1, f"{v:.0f}%", ha="center")
        plt.ylim(0, 110)
        plt.ylabel("Độ chính xác (%)")
        plt.title(f"Ảnh tự thu (lượt 2): độ chính xác theo tư thế – {tot_nhat}")
        C.luu_hinh("5_theo_tu_the.png")
        print("  Theo tư thế: " + ", ".join(f"{C.TEN_TU_THE[t]} {a:.0f}%" for t, a in zip(tt, acc)))
    sai = np.where(y_doan != y_test)[0]
    print(f"  Số ảnh nhận sai: {len(sai)}/{len(y_test)}")
    if len(sai):
        chon = rng.choice(sai, size=min(16, len(sai)), replace=False)
        xs = P_test[chon].max(1)
        C.ve_luoi_anh(X_test[chon], [f"Thật: {C.rut_gon(t)}" for t in y_test[chon]],
                      [f"Đoán: {C.rut_gon(d)} ({p:.0%})" for d, p in zip(y_doan[chon], xs)],
                      [False] * len(chon), "5_anh_nhan_sai.png",
                      f"Một số ảnh bị nhận sai – {tot_nhat}")


# ======================= 5.6 NGƯỠNG NGƯỜI LẠ =======================
def chon_nguong(P_test, dung, P_la, X_la, lop):
    """Xác suất cao nhất < ngưỡng -> "Người lạ".
    FRR = tỉ lệ người QUEN bị từ chối | FAR = tỉ lệ người LẠ bị nhận nhầm.
    Chọn ngưỡng tại FAR = FRR."""
    if len(P_la) == 0:
        print("  Không có ảnh người lạ -> dùng ngưỡng mặc định 0.6")
        return 0.6, None, None
    p_quen, p_la = P_test.max(1), P_la.max(1)
    cac_nguong = np.round(np.linspace(0, 1, 101), 2)
    frr = np.array([(p_quen < t).mean() for t in cac_nguong])
    far = np.array([(p_la >= t).mean() for t in cac_nguong])
    nhan_dung = np.array([((p_quen >= t) & dung).mean() for t in cac_nguong])
    i = int(np.argmin(np.abs(far - frr)))
    t = float(cac_nguong[i])
    print(f"  Ngưỡng người lạ = {t:.2f}: FAR {far[i] * 100:.1f}% | FRR {frr[i] * 100:.1f}% | "
          f"người quen nhận đúng {nhan_dung[i] * 100:.1f}%")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    kieu = dict(bins=20, range=(0, 1), alpha=0.6)
    ax[0].hist(p_quen[dung], color="seagreen", label="Người quen – đoán đúng", **kieu)
    if (~dung).any():
        ax[0].hist(p_quen[~dung], color="tomato", label="Người quen – đoán sai", **kieu)
    ax[0].hist(p_la, color="gray", label="Người lạ", **kieu)
    ax[0].axvline(t, color="k", ls="--", label=f"Ngưỡng {t:.2f}")
    ax[0].set(xlabel="Xác suất cao nhất (predict_proba)", ylabel="Số ảnh",
              title="Phân bố xác suất cao nhất")
    ax[0].legend(fontsize=8)
    ax[1].plot(cac_nguong, far * 100, color="tomato", lw=2, label="FAR – người lạ bị nhận nhầm")
    ax[1].plot(cac_nguong, frr * 100, color="steelblue", lw=2, label="FRR – người quen bị từ chối")
    ax[1].plot(cac_nguong, nhan_dung * 100, color="seagreen", lw=2, ls="--",
               label="Người quen nhận đúng")
    ax[1].axvline(t, color="k", ls=":")
    ax[1].set(xlabel="Ngưỡng", ylabel="%", title=f"Chọn ngưỡng tại FAR = FRR: {t:.2f}")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    C.luu_hinh("5_nguoi_la.png")
    chon = rng.choice(len(X_la), size=min(16, len(X_la)), replace=False)
    doan = [lop[j] if p >= t else "Người lạ" for j, p in zip(P_la[chon].argmax(1), p_la[chon])]
    C.ve_luoi_anh(X_la[chon], ["Thật: Người lạ"] * len(chon),
                  [f"Đoán: {C.rut_gon(d)} ({p:.0%})" for d, p in zip(doan, p_la[chon])],
                  [d == "Người lạ" for d in doan], "5_nguoi_la_anh.png",
                  f"Ảnh người lạ (không có trong tập huấn luyện) – ngưỡng {t:.2f}")
    return t, float(far[i]), float(frr[i])


# ======================= 5.8 TRANG XEM KẾT QUẢ =======================
MO_TA = {
    "1_so_anh_moi_nguoi": "Số ảnh của mỗi người (cam: tự thu, xanh: LFW)",
    "1_anh_mau": "Ảnh mẫu sau tiền xử lý",
    "1_can_bang_sang": "Trước/sau tiền xử lý (lọc Gaussian + CLAHE) và histogram mức xám",
    "2_chia_train_test": "Số ảnh train/test của mỗi người",
    "3_pca_phuong_sai": "Phương sai giữ lại theo số thành phần chính",
    "3_eigenfaces": "Mặt trung bình và các eigenfaces",
    "3_truoc_sau_pca": "Dữ liệu trước và sau PCA",
    "3_khoi_phuc_pca": "Ảnh khôi phục từ một số thành phần chính",
    "3_cv_cac_mo_hinh": "Độ chính xác cross-validation của 7 mô hình",
    "3_svm_rbf_C_gamma": "Ảnh hưởng của C và gamma tới SVM rbf",
    "3_cay_quyet_dinh": "3 tầng đầu của cây quyết định",
    "4_du_doan_tap_test": "Ảnh tập test kèm tên thật / tên dự đoán",
    "4_xac_suat_du_doan": "Xác suất predict_proba (như ví dụ AR Face)",
    "5_so_sanh_mo_hinh": "So sánh độ chính xác CV và test",
    "5_ma_tran_nham_lan_cac_mo_hinh": "Ma trận nhầm lẫn của 7 mô hình",
    "5_ma_tran_nham_lan_tot_nhat": "Ma trận nhầm lẫn của mô hình tốt nhất",
    "5_roc_tot_nhat": "Đường ROC và AUC",
    "5_theo_tu_the": "Độ chính xác theo tư thế (ảnh tự thu, lượt 2)",
    "5_theo_nguon_anh": "Độ chính xác theo nguồn ảnh (tự thu / chụp tay / LFW)",
    "5_anh_nhan_sai": "Các ảnh bị nhận sai",
    "5_nguoi_la": "Chọn ngưỡng người lạ (FAR/FRR)",
    "5_nguoi_la_anh": "Ảnh người lạ và kết luận của hệ thống",
}
BUOC = [("1_", "Bước 1 – Chuẩn bị dữ liệu (Thành viên 1)"),
        ("2_", "Bước 2 – Chia train/test (Thành viên 1)"),
        ("3_", "Bước 3 – Huấn luyện (Thành viên 2)"),
        ("4_", "Bước 4 – Dự đoán (Thành viên 2)"),
        ("5_", "Bước 5 – Đánh giá (Thành viên 3)")]
CSS = ("body{font-family:Arial,sans-serif;max-width:1200px;margin:auto;padding:16px;"
       "background:#f6f7f9} h1{color:#1f3b73} h2{border-bottom:2px solid #1f3b73;padding-top:16px}"
       "table{border-collapse:collapse;background:#fff} th{background:#1f3b73;color:#fff}"
       "td,th{border:1px solid #bbb;padding:4px 8px;font-size:14px} img{max-width:100%}"
       "figure{background:#fff;margin:12px 0;padding:8px;border:1px solid #ddd}")


def tao_trang_xem(bang, tom_tat):
    hinh = sorted(f for f in os.listdir(C.KET_QUA) if f.endswith(".png"))
    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
         f"<title>Kết quả nhận diện khuôn mặt</title><style>{CSS}</style></head><body>",
         "<h1>Kết quả nhận diện khuôn mặt</h1><p>",
         "<br>".join(f"<b>{k}:</b> {v}" for k, v in tom_tat.items()) + "</p>",
         bang.to_html(index=False)]                            # pandas đổi bảng thành HTML
    for tien_to, tieu_de in BUOC:
        ds = [f for f in hinh if f.startswith(tien_to)]
        if ds:
            h.append(f"<h2>{tieu_de}</h2>")
            for f in ds:
                h.append(f"<figure><img src='{f}'><figcaption><b>{f}</b> – "
                         f"{MO_TA.get(f[:-4], '')}</figcaption></figure>")
    h.append("</body></html>")
    p = os.path.join(C.KET_QUA, "xem_ket_qua.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    print(f"  -> {os.path.relpath(p, C.THU_MUC)}  (mở bằng trình duyệt để xem tất cả hình)")
    if os.name == "nt" or os.environ.get("DISPLAY"):           # máy có màn hình thì tự mở
        try:
            webbrowser.open(Path(p).as_uri())
        except Exception:
            pass


# =====================================================================
def main():
    print("=" * 70)
    print(" BƯỚC 5 – ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 70)

    print("\n5.1 Đọc kết quả dự đoán (bước 4)")
    if not os.path.exists(C.FILE_DU_DOAN):
        raise SystemExit("Chưa có kết quả dự đoán. Chạy buoc4_du_doan.py trước.")
    d = np.load(C.FILE_DU_DOAN)
    ten_mh, du_doan, ms = list(d["ten_mo_hinh"]), d["du_doan"], d["ms_moi_anh"]
    tot_nhat, lop, y_test = str(d["tot_nhat"]), list(d["lop"]), d["y_test"]
    P_test, P_la = d["P_test"], d["P_la"]
    X, df = C.doc_du_lieu()
    vi_tri = {f: i for i, f in enumerate(df["file"])}
    i_test = np.array([vi_tri[f] for f in d["file_test"]], dtype=int)
    i_la = np.array([vi_tri[f] for f in d["file_la"]], dtype=int)
    cv = pd.read_csv(C.FILE_KET_QUA_CV, encoding="utf-8-sig")
    print(f"  {len(ten_mh)} mô hình | {len(y_test)} ảnh test | {len(i_la)} ảnh người lạ | "
          f"tốt nhất (CV): {tot_nhat}")

    print("\n5.2 Độ đo trên tập test (P, R, F1 trung bình macro)")
    bang = tinh_do_do(ten_mh, du_doan, y_test, ms, cv)
    y_doan = du_doan[ten_mh.index(tot_nhat)]
    print(f"\n  Báo cáo chi tiết – {tot_nhat}:")
    print(classification_report(y_test, y_doan, zero_division=0))

    print("5.3 Ma trận nhầm lẫn")
    ve_cac_ma_tran(ten_mh, du_doan, y_test, lop, bang, tot_nhat)

    print("\n5.4 Đường ROC và AUC")
    ve_roc(y_test, P_test, lop, tot_nhat)

    print("\n5.5 Phân tích lỗi")
    phan_tich_loi(X[i_test], y_test, y_doan, P_test, df["tu_the"].to_numpy()[i_test],
                  df["nguon"].to_numpy()[i_test], tot_nhat)

    print("\n5.6 Chọn ngưỡng người lạ")
    dung = np.array(lop)[P_test.argmax(1)] == y_test
    nguong, far, frr = chon_nguong(P_test, dung, P_la, X[i_la], lop)

    print("\n5.7 Học lại mô hình tốt nhất trên toàn bộ ảnh người quen, lưu cho bước 6")
    goi = joblib.load(C.FILE_CAC_MO_HINH)
    biet = (df["vai_tro"] == "biet").to_numpy()
    mh = clone(goi["mo_hinh"][tot_nhat])                    # cùng tham số tốt nhất, chưa học
    mh.fit(X[biet], df.loc[biet, "ten"].to_numpy())          # học trên train + test (hết dữ liệu)
    acc = float(bang.loc[bang["mo_hinh"] == tot_nhat, "test_%"].iloc[0])
    joblib.dump({"mo_hinh": mh, "ten_mo_hinh": tot_nhat, "lop": list(mh.classes_),
                 "nguoi_diem_danh": goi["nguoi_diem_danh"], "nguong_nguoi_la": nguong,
                 "do_chinh_xac_test": acc}, C.FILE_MO_HINH_TOT, compress=3)
    far_frr = "-" if far is None else f"{far * 100:.1f} / {frr * 100:.1f}"
    tom_tat = {"Mô hình tốt nhất (theo CV)": tot_nhat,
               "Tham số": bang.loc[bang["mo_hinh"] == tot_nhat, "tham_so"].iloc[0],
               "Độ chính xác test (%)": acc, "Số thành phần PCA": goi["so_thanh_phan_pca"],
               "Ngưỡng người lạ": nguong,
               "FAR / FRR tại ngưỡng (%)": far_frr,
               "Người được điểm danh (trong lớp)": ", ".join(goi["nguoi_diem_danh"]) or "-"}
    with open(os.path.join(C.MO_HINH, "thong_tin.json"), "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in tom_tat.items()}, f, ensure_ascii=False, indent=2)
    print(f"  {tot_nhat}: học lại trên {int(biet.sum())} ảnh -> mo_hinh/mo_hinh_tot_nhat.joblib")

    print("\n5.8 Tạo trang xem kết quả")
    tao_trang_xem(bang, tom_tat)
    print("\nXong bước 5. Tiếp theo: python buoc6_nhan_dien.py (hoặc --web trên Jetson)")


if __name__ == "__main__":
    main()

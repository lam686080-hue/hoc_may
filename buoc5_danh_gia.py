"""
=====================================================================
 BƯỚC 5 – ĐÁNH GIÁ MÔ HÌNH
 Phụ trách: Thành viên 3 – Lê Thanh Cao
=====================================================================
 Làm gì   : So kết quả dự đoán (bước 4) với tên thật để đánh giá hồi quy logistic,
            chọn ngưỡng "người lạ", rồi cập nhật ngưỡng vào mô hình cho bước 6.
 Đầu vào  : ket_qua/du_doan.npz (bước 4), ket_qua/3_ket_qua_cv.csv và
            mo_hinh/hoi_quy_logistic.joblib (bước 3), dữ liệu bước 1-2
 Đầu ra   : ket_qua/5_*.png, 5_danh_gia_logistic.csv, 5_bao_cao_phan_loai.txt, test_nguoi_la.csv
            mo_hinh/hoi_quy_logistic.joblib (cập nhật thêm ngưỡng) + thong_tin.json (cho bước 6)
            ket_qua/xem_ket_qua.html     trang xem TẤT CẢ kết quả và hình của 5 bước
 Các bước : 5.1 Đọc kết quả dự đoán
            5.2 Tính độ đo: accuracy, precision, recall, F1 -> bảng đánh giá
            5.3 Ma trận nhầm lẫn
            5.4 Đường ROC và AUC
            5.5 Phân tích lỗi: theo tư thế và ảnh nhận sai (nếu có)
            5.6 Người lạ: chọn ngưỡng cân bằng FAR/FRR trên tập hiệu chỉnh, rồi kiểm tra độc lập
            5.7 Cập nhật ngưỡng đã chọn vào mo_hinh/hoi_quy_logistic.joblib cho bước 6
            5.8 Tạo trang xem kết quả tổng hợp (xem_ket_qua.html)
 Chạy     : python buoc5_danh_gia.py
=====================================================================
"""
import html
import os
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import seaborn as sns                                   # vẽ ma trận nhầm lẫn (heatmap)
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize       # đổi tên người -> dạng 0/1 cho ROC
import chung as C
from chung import plt

rng = np.random.default_rng(C.RANDOM_STATE)


# ======================= 5.2 ĐỘ ĐO ĐÁNH GIÁ =======================
def tinh_do_do(ten, du_doan, y_test, ms, ms_don, cv):
    acc = accuracy_score(y_test, du_doan)
    p, r, f, _ = precision_recall_fscore_support(y_test, du_doan, average="macro", zero_division=0)
    diem = cv.iloc[0]
    dong = {"mo_hinh": ten, "tham_so": diem["tham_so"], "cv_%": diem["cv_%"],
            "test_%": round(acc * 100, 2), "precision": round(p, 3),
            "recall": round(r, 3), "f1": round(f, 3), "ms_batch": round(ms, 3)}
    if ms_don is not None:
        dong["ms_don_le"] = round(ms_don, 3)
    bang = pd.DataFrame([dong])
    print(bang.to_string(index=False))
    bang.to_csv(os.path.join(C.KET_QUA, "5_danh_gia_logistic.csv"), index=False, encoding="utf-8-sig")
    bao_cao = classification_report(y_test, du_doan, zero_division=0)
    Path(C.KET_QUA, "5_bao_cao_phan_loai.txt").write_text(bao_cao, encoding="utf-8")

    ax = bang.set_index("mo_hinh")[["cv_%", "test_%"]].plot.bar(
        figsize=(6.2, 4.4), width=0.45, color=["steelblue", "darkorange"], edgecolor="black", rot=0)
    for cot in ax.containers:
        ax.bar_label(cot, fmt="%.1f%%", fontsize=9)
    toc_do = f"{ms:.2f} ms/ảnh (batch)" + (f" | {ms_don:.2f} ms/ảnh (đơn lẻ)" if ms_don is not None else "")
    ax.set_xticklabels([f"{ten}\n{toc_do}"], fontsize=9)
    ax.legend(["Cross-validation (tập train)", "Tập test"], loc="upper center", ncol=2)
    ax.set(xlabel="", ylabel="Độ chính xác (%)", ylim=(0, 122), title="Đánh giá hồi quy logistic")
    C.luu_hinh("5_danh_gia_logistic.png")
    return bang, bao_cao


# ======================= 5.3 MA TRẬN NHẦM LẪN =======================
def ve_ma_tran_logistic(du_doan, y_test, lop, do_chinh_xac):
    """Heatmap ma trận nhầm lẫn của hồi quy logistic trên tập test."""
    nhan, co = [C.rut_gon(n) for n in lop], min(len(lop), 26)
    _, ax = plt.subplots(figsize=(co + 3, 0.85 * co + 2.5))
    sns.heatmap(confusion_matrix(y_test, du_doan, labels=lop), annot=len(nhan) <= 30, fmt="d",
                cmap="Blues", ax=ax, xticklabels=nhan, yticklabels=nhan, annot_kws={"size": 9})
    ax.set(xlabel="Predicted Label (dự đoán)", ylabel="True Label (thực tế)",
           title=f"Ma trận nhầm lẫn – Hồi quy logistic ({do_chinh_xac:.1f}%)")
    ax.tick_params(labelsize=9)
    C.luu_hinh("5_ma_tran_nham_lan.png")


# ======================= 5.4 ROC – AUC =======================
def ve_roc(y_test, P_test, lop, ten_mo_hinh):
    """ROC one-vs-rest: với mỗi người, coi 'là người đó' là lớp dương, còn lại là lớp âm."""
    Y = label_binarize(y_test, classes=lop)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    plt.figure(figsize=(7.5, 6))
    for i, n in enumerate(lop):
        if 0 < Y[:, i].sum() < len(Y):
            fpr, tpr, _ = roc_curve(Y[:, i], P_test[:, i])
            plt.plot(fpr, tpr, lw=1.2, label=f"{C.rut_gon(n)} (AUC={auc(fpr, tpr):.3f})" if len(lop) <= 15 else None)
    fpr, tpr, _ = roc_curve(Y.ravel(), P_test.ravel())
    auc_tb = auc(fpr, tpr)
    plt.plot(fpr, tpr, "k--", lw=2, label=f"Trung bình micro (AUC={auc_tb:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", ls=":", label="Ngẫu nhiên")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (one-vs-rest) – {ten_mo_hinh}")
    plt.legend(fontsize=7, loc="lower right")
    C.luu_hinh("5_roc_auc.png")
    print(f"  AUC trung bình (micro) = {auc_tb:.3f}")


# ======================= 5.5 PHÂN TÍCH LỖI =======================
def phan_tich_loi(X_test, y_test, y_doan, tu_the, ten_mo_hinh):
    tt = [t for t in C.TEN_TU_THE if (tu_the == t).any()]
    if tt:
        acc = [accuracy_score(y_test[tu_the == t], y_doan[tu_the == t]) * 100 for t in tt]
        print("  Theo tư thế: " + ", ".join(f"{C.TEN_TU_THE[t]} {a:.0f}%" for t, a in zip(tt, acc)))
        plt.figure(figsize=(7, 4))
        plt.bar_label(plt.bar([C.TEN_TU_THE[t] for t in tt], acc, color="seagreen", edgecolor="black"), fmt="%.0f%%")
        plt.ylim(0, 110)
        plt.ylabel("Độ chính xác (%)")
        plt.title(f"Ảnh tự thu trong tập kiểm tra: độ chính xác theo tư thế – {ten_mo_hinh}")
        C.luu_hinh("5_theo_tu_the.png")
    sai = np.where(y_doan != y_test)[0]
    print(f"  Số ảnh nhận sai: {len(sai)}/{len(y_test)}")
    if len(sai):
        chon = rng.choice(sai, size=min(16, len(sai)), replace=False)
        C.ve_luoi_anh(X_test[chon], [f"Thật: {C.rut_gon(t)}" for t in y_test[chon]],
                      [f"Đoán: {C.rut_gon(d)}" for d in y_doan[chon]], [False] * len(chon),
                      "5_anh_nhan_sai.png", f"Một số ảnh bị nhận sai – {ten_mo_hinh}")
    else:
        (Path(C.KET_QUA) / "5_anh_nhan_sai.png").unlink(missing_ok=True)


# ======================= 5.6 NGƯỠNG NGƯỜI LẠ =======================
def chon_nguong(P_test, dung, P_la, X_la, lop):
    """Xác suất cao nhất < ngưỡng -> 'Người lạ'. Chọn ngưỡng có |FAR - FRR| nhỏ nhất."""
    if len(P_la) == 0:
        for ten in ("5_nguoi_la.png", "5_nguoi_la_anh.png"):
            (Path(C.KET_QUA) / ten).unlink(missing_ok=True)
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
    for du_lieu, mau, nhan in ((p_quen[dung], "seagreen", "Người quen – đoán đúng"),
                               (p_quen[~dung], "tomato", "Người quen – đoán sai"),
                               (p_la, "gray", "Người lạ")):
        ax[0].hist(du_lieu, color=mau, label=nhan, **kieu)
    ax[0].axvline(t, color="k", ls="--", label=f"Ngưỡng {t:.2f}")
    ax[0].set(xlabel="Xác suất cao nhất (predict_proba)", ylabel="Số ảnh", title="Phân bố xác suất cao nhất")
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


def kiem_tra_nguoi_la(mh, nguong):
    """Đánh giá độc lập khả năng từ chối người lạ trên bộ ảnh dataset/nguoi_la."""
    X_la, names, status = C.doc_nguoi_la("nguoi_la")
    xs = iter(mh.predict_proba(X_la)) if len(X_la) else iter(())
    rows, doan_ok, p_ok, dung_ok = [], [], [], []
    for name, state in zip(names, status):
        label, score = "", ""
        if state == "ok":
            probs = next(xs)
            i = int(probs.argmax())
            score = float(probs[i])
            nhan_nham = score >= nguong
            state, label = ("nhan_nham", mh.classes_[i]) if nhan_nham else ("tu_choi_dung", "Nguoi la")
            doan_ok.append(label)
            p_ok.append(score)
            dung_ok.append(not nhan_nham)
        rows.append((name, state, label, round(score, 4) if isinstance(score, float) else score))
    bang = pd.DataFrame(rows, columns=["file", "trang_thai", "du_doan", "xac_suat_cao_nhat"])
    bang.to_csv(Path(C.KET_QUA, "test_nguoi_la.csv"), index=False, encoding="utf-8-sig")
    if len(X_la):
        n = min(16, len(X_la))
        C.ve_luoi_anh(X_la[:n], ["Thật: Người lạ"] * n,
                      [f"Đoán: {'Người lạ' if d == 'Nguoi la' else C.rut_gon(d)} ({p:.0%})"
                       for d, p in zip(doan_ok[:n], p_ok[:n])],
                      dung_ok[:n], "5_nguoi_la_kiem_tra.png",
                      f"Kiểm tra độc lập ảnh người lạ (dataset/nguoi_la) – ngưỡng {nguong:.2f}")
    else:
        Path(C.KET_QUA, "5_nguoi_la_kiem_tra.png").unlink(missing_ok=True)
    sai = int((bang.trang_thai == "nhan_nham").sum())
    print(f"  Kiểm tra người lạ: nhận nhầm {sai}/{len(X_la)}, bỏ qua {len(names) - len(X_la)} ảnh")
    return (f"{sai}/{len(X_la)}" if len(X_la) else "Chưa có ảnh hợp lệ"), bang


# ======================= 5.8 TRANG XEM KẾT QUẢ =======================
MUC = [("1_", "Bước 1 – Chuẩn bị dữ liệu (Thành viên 1 – Bùi Trung Hiếu)"),
       ("2_", "Bước 2 – Chia train/test (Thành viên 2 – Nguyễn Văn Lâm)"),
       (("3_pca", "3_eigenfaces", "3_truoc_sau_pca", "3_khoi_phuc_pca"),
        "Bước 3.2 – Phân tích PCA (Thành viên 1 – Bùi Trung Hiếu)"),
       ("3_", "Bước 3.3 – 3.5 – Huấn luyện hồi quy logistic (Thành viên 2 – Nguyễn Văn Lâm)"),
       ("4_", "Bước 4 – Dự đoán (Thành viên 2 – Nguyễn Văn Lâm)"),
       ("5_", "Bước 5 – Đánh giá (Thành viên 3 – Lê Thanh Cao)")]


def tao_trang_xem(bang, tom_tat, bao_cao, bang_nguoi_la):
    """Trang HTML: tóm tắt + bảng đánh giá + mọi hình + báo cáo chi tiết + kiểm tra người lạ."""
    con_lai = sorted(f for f in os.listdir(C.KET_QUA) if f.endswith(".png"))
    css = ("body{font-family:Arial,sans-serif;max-width:1200px;margin:20px auto;padding:0 16px;line-height:1.5}"
           "table{border-collapse:collapse;margin:12px 0;font-size:14px}th,td{border:1px solid #ccc;padding:6px 10px}"
           "th{background:#f3f6fa}pre{background:#f7f9fb;border:1px solid #ddd;padding:12px;overflow-x:auto}"
           "h2{margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}")
    h = [f"<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'><title>Kết quả nhận diện khuôn mặt</title>"
         f"<style>{css}</style></head><body><h1>Kết quả đánh giá hệ thống nhận diện khuôn mặt</h1><p>"
         + "<br>".join(f"<b>{html.escape(str(k))}:</b> {html.escape(str(v))}" for k, v in tom_tat.items()) + "</p>",
         bang.to_html(index=False)]
    for tien_to, tieu_de in MUC:
        ds = [f for f in con_lai if f.startswith(tien_to)]
        con_lai = [f for f in con_lai if f not in ds]
        if ds:
            h += [f"<h2>{tieu_de}</h2>"] + [f"<p><b>{f}</b><br><img src='{f}' style='max-width:100%'></p>" for f in ds]
    h += ["<h3>Báo cáo phân loại chi tiết từng người (5_bao_cao_phan_loai.txt)</h3>",
          f"<pre>{html.escape(bao_cao)}</pre>"]
    if not bang_nguoi_la.empty:
        h += ["<h3>Bảng kiểm tra độc lập ảnh người lạ (test_nguoi_la.csv)</h3>", bang_nguoi_la.to_html(index=False)]
    p = Path(C.KET_QUA, "xem_ket_qua.html")
    p.write_text("\n".join(h) + "\n</body></html>", encoding="utf-8")
    print("  -> ket_qua/xem_ket_qua.html  (mở bằng trình duyệt để xem tất cả kết quả)")
    if os.name == "nt":
        webbrowser.open(p.as_uri())


# =====================================================================
def main():
    print("=" * 70 + "\n BƯỚC 5 – ĐÁNH GIÁ MÔ HÌNH\n" + "=" * 70)

    print("\n5.1 Đọc kết quả dự đoán (bước 4)")
    if not os.path.exists(C.FILE_DU_DOAN):
        raise SystemExit("Chưa có kết quả dự đoán. Chạy buoc4_du_doan.py trước.")
    d = np.load(C.FILE_DU_DOAN)
    ten_mo_hinh = "Logistic Regression"
    y_doan, ms = d["du_doan"], float(d["ms_moi_anh"])
    ms_don = float(d["ms_don_le"]) if "ms_don_le" in d else None
    lop, y_test, P_test = list(d["lop"]), d["y_test"], d["P_test"]
    X, df = C.doc_du_lieu()
    vi_tri = {f: i for i, f in enumerate(df["file"])}
    i_test = np.array([vi_tri[f] for f in d["file_test"]], dtype=int)
    cv = pd.read_csv(C.FILE_KET_QUA_CV, encoding="utf-8-sig")
    print(f"  {ten_mo_hinh} | {len(y_test)} ảnh kiểm tra")

    print("\n5.2 Đánh giá trên tập kiểm tra")
    bang, bao_cao = tinh_do_do(ten_mo_hinh, y_doan, y_test, ms, ms_don, cv)
    print(f"\n  Báo cáo chi tiết – {ten_mo_hinh}:\n{bao_cao}")

    print("5.3 Ma trận nhầm lẫn")
    ve_ma_tran_logistic(y_doan, y_test, lop, bang["test_%"].iloc[0])

    print("\n5.4 Đường ROC và AUC")
    ve_roc(y_test, P_test, lop, ten_mo_hinh)

    print("\n5.5 Phân tích lỗi")
    phan_tich_loi(X[i_test], y_test, y_doan, df["tu_the"].to_numpy()[i_test], ten_mo_hinh)

    print("\n5.6 Chọn ngưỡng người lạ")
    goi = joblib.load(C.FILE_HOI_QUY_LOGISTIC)
    mh = goi["mo_hinh"]
    X_la, _, _ = C.doc_nguoi_la("nguoi_la_chon_nguong")
    P_la = mh.predict_proba(X_la) if len(X_la) else np.empty((0, len(lop)))
    # Giữ lại một nửa ảnh test của từng người để đánh giá ngưỡng độc lập.
    chia = np.random.default_rng(C.RANDOM_STATE)
    hieu_chinh = np.zeros(len(y_test), dtype=bool)
    for ten in lop:
        indices = chia.permutation(np.flatnonzero(y_test == ten))
        hieu_chinh[indices[:len(indices) // 2]] = True
    dung = np.asarray(lop)[P_test.argmax(1)] == y_test
    nguong, far, frr = chon_nguong(P_test[hieu_chinh], dung[hieu_chinh], P_la, X_la, lop)
    nhan_nham_la, bang_nguoi_la = kiem_tra_nguoi_la(mh, nguong)
    kiem_tra = ~hieu_chinh
    dung_sau_nguong = int(((P_test.max(1) >= nguong) & dung & kiem_tra).sum())
    print(f"  Kiểm tra người quen sau ngưỡng: {dung_sau_nguong}/{kiem_tra.sum()} ảnh nhận đúng")

    print("\n5.7 Cập nhật ngưỡng đã chọn vào mo_hinh/hoi_quy_logistic.joblib cho bước 6")
    acc = float(bang["test_%"].iloc[0])
    goi.update({"nguong_nguoi_la": nguong, "do_chinh_xac_test": acc})
    joblib.dump(goi, C.FILE_HOI_QUY_LOGISTIC, compress=3)
    tom_tat = {
        "Mô hình": ten_mo_hinh, "Tham số": bang["tham_so"].iloc[0],
        "Độ chính xác test (%)": acc, "Số thành phần PCA": goi["so_thanh_phan_pca"],
        "Tốc độ dự đoán batch / đơn lẻ (ms/ảnh)": f"{ms:.3f} / {ms_don:.3f}" if ms_don is not None else f"{ms:.3f}",
        "Ngưỡng người lạ": nguong,
        "Cách chọn ngưỡng": "Cân bằng FAR/FRR" if len(X_la) else "Mặc định (chưa hiệu chỉnh)",
        "Ảnh lạ dùng chọn ngưỡng": len(X_la), "Ảnh lạ kiểm tra bị nhận nhầm": nhan_nham_la,
        "Ảnh quen kiểm tra nhận đúng sau ngưỡng": f"{dung_sau_nguong}/{kiem_tra.sum()}",
        "FAR / FRR trên ảnh dùng chọn ngưỡng (%)": "-" if far is None else f"{far * 100:.1f} / {frr * 100:.1f}",
        "Người được điểm danh (trong lớp)": ", ".join(goi["nguoi_diem_danh"]) or "-",
    }
    Path(C.MO_HINH, "thong_tin.json").write_text(
        json.dumps({k: str(v) for k, v in tom_tat.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {ten_mo_hinh}: đã huấn luyện trên {(df.tap == 'train').sum()} ảnh -> mo_hinh/hoi_quy_logistic.joblib")

    print("\n5.8 Tạo trang xem kết quả")
    tao_trang_xem(bang, tom_tat, bao_cao, bang_nguoi_la)
    print("\nXong bước 5. Tiếp theo: python buoc6_nhan_dien.py (hoặc --web trên Jetson)")


if __name__ == "__main__":
    main()

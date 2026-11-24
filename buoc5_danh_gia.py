"""Bước 5 – Đánh giá Logistic Regression. Phụ trách: Lê Thanh Cao.

Đọc dự đoán bước 4, lập báo cáo, chọn và kiểm tra ngưỡng người lạ.
Lưu ngưỡng vào mô hình của bước 3 cho camera bước 6.
Chạy: python buoc5_danh_gia.py
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
import chung as C
from chung import plt

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
    return bang


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


def chon_nguong(P_test, P_la):
    """Xác suất cao nhất < ngưỡng -> 'Người lạ'. Chọn ngưỡng có |FAR - FRR| nhỏ nhất."""
    if len(P_la) == 0:
        print("  Không có ảnh người lạ -> dùng ngưỡng mặc định 0.6")
        return 0.6, None, None
    p_quen, p_la = P_test.max(1), P_la.max(1)
    cac_nguong = np.round(np.linspace(0, 1, 101), 2)
    frr = np.array([(p_quen < t).mean() for t in cac_nguong])
    far = np.array([(p_la >= t).mean() for t in cac_nguong])
    i = int(np.argmin(np.abs(far - frr)))
    t = float(cac_nguong[i])
    print(f"  Ngưỡng {t:.2f}, FAR {far[i]:.1%}, FRR {frr[i]:.1%}")
    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cac_nguong, far * 100, label="FAR – người lạ bị nhận nhầm")
    ax.plot(cac_nguong, frr * 100, label="FRR – người quen bị từ chối")
    ax.axvline(t, color="black", linestyle="--", label=f"Ngưỡng {t:.2f}")
    ax.set(xlabel="Ngưỡng", ylabel="Tỷ lệ (%)", title="Chọn ngưỡng người lạ")
    ax.legend()
    C.luu_hinh("5_nguoi_la.png")
    return t, float(far[i]), float(frr[i])


def kiem_tra_nguoi_la(mh, nguong):
    """Đánh giá độc lập khả năng từ chối người lạ trên bộ ảnh dataset/nguoi_la."""
    X_la, names, status = C.doc_nguoi_la("nguoi_la")
    xs = iter(mh.predict_proba(X_la)) if len(X_la) else iter(())
    rows = []
    for name, state in zip(names, status):
        label, score = "", ""
        if state == "ok":
            probs = next(xs)
            i = int(probs.argmax())
            score = float(probs[i])
            nhan_nham = score >= nguong
            state, label = ("nhan_nham", mh.classes_[i]) if nhan_nham else ("tu_choi_dung", "Nguoi la")
        rows.append((name, state, label, round(score, 4) if isinstance(score, float) else score))
    bang = pd.DataFrame(rows, columns=["file", "trang_thai", "du_doan", "xac_suat_cao_nhat"])
    bang.to_csv(Path(C.KET_QUA, "test_nguoi_la.csv"), index=False, encoding="utf-8-sig")
    sai = int((bang.trang_thai == "nhan_nham").sum())
    print(f"  Kiểm tra người lạ: nhận nhầm {sai}/{len(X_la)}, bỏ qua {len(names) - len(X_la)} ảnh")
    return f"{sai}/{len(X_la)}" if len(X_la) else "Chưa có ảnh hợp lệ"


def tao_trang_xem(bang, tom_tat, bao_cao, co_anh_hieu_chinh):
    muc = ["<h1>Đánh giá mô hình nhận diện</h1>", bang.to_html(index=False),
           "<h2>Classification Report</h2>", f"<pre>{html.escape(bao_cao)}</pre>",
           "<h2>Confusion Matrix</h2>", "<img src='5_ma_tran_nham_lan.png' alt='Ma trận nhầm lẫn'>",
           "<h2>Ngưỡng và kết quả kiểm tra độc lập</h2>"]
    for k in ("Ngưỡng người lạ", "Cách chọn ngưỡng", "Ảnh lạ dùng chọn ngưỡng",
              "FAR / FRR trên ảnh dùng chọn ngưỡng (%)", "Ảnh quen kiểm tra nhận đúng sau ngưỡng",
              "Ảnh lạ kiểm tra bị nhận nhầm"):
        muc.append(f"<p>{html.escape(k)}: {html.escape(str(tom_tat[k]))}</p>")
    if co_anh_hieu_chinh:
        muc.append("<img src='5_nguoi_la.png' alt='FAR và FRR theo ngưỡng'>")
    muc.append("<p><a href='test_nguoi_la.csv'>Kết quả từng ảnh người lạ (CSV)</a></p>")
    trang = Path(C.KET_QUA, "xem_ket_qua.html")
    trang.write_text("<!doctype html><html lang='vi'><meta charset='utf-8'><title>Kết quả bước 5</title>"
                     "<style>body{font:16px Arial;max-width:960px;margin:30px auto}img{max-width:100%}"
                     "th,td{padding:5px}pre{overflow:auto}</style><body>"
                     + "\n".join(muc) + "</body></html>", encoding="utf-8")
    if os.name == "nt":
        webbrowser.open(trang.as_uri())


def main():
    print("=" * 70 + "\n BƯỚC 5 – ĐÁNH GIÁ MÔ HÌNH\n" + "=" * 70)

    print("\n5.1 Accuracy, Precision, Recall và F1-score")
    if not os.path.exists(C.FILE_DU_DOAN):
        raise SystemExit("Chưa có kết quả dự đoán. Chạy buoc4_du_doan.py trước.")
    ten_mo_hinh = "Logistic Regression"
    with np.load(C.FILE_DU_DOAN) as d:
        y_doan, y_test, P_test = d["du_doan"], d["y_test"], d["P_test"]
        lop = list(d["lop"])
        ms = float(d["ms_moi_anh"])
        ms_don = float(d["ms_don_le"]) if "ms_don_le" in d else None
    cv = pd.read_csv(C.FILE_KET_QUA_CV, encoding="utf-8-sig")
    print(f"  {ten_mo_hinh} | {len(y_test)} ảnh kiểm tra")

    bang = tinh_do_do(ten_mo_hinh, y_doan, y_test, ms, ms_don, cv)

    print("\n5.2 Classification Report")
    bao_cao = classification_report(y_test, y_doan, zero_division=0)
    print(bao_cao)
    Path(C.KET_QUA, "5_bao_cao_phan_loai.txt").write_text(bao_cao, encoding="utf-8")

    print("5.3 Ma trận nhầm lẫn")
    ve_ma_tran_logistic(y_doan, y_test, lop, bang["test_%"].iloc[0])

    print("\n5.4 Chọn và kiểm tra ngưỡng người lạ")
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
    nguong, far, frr = chon_nguong(P_test[hieu_chinh], P_la)
    nhan_nham_la = kiem_tra_nguoi_la(mh, nguong)
    kiem_tra = ~hieu_chinh
    dung_sau_nguong = int(((P_test.max(1) >= nguong) & dung & kiem_tra).sum())
    print(f"  Kiểm tra người quen sau ngưỡng: {dung_sau_nguong}/{kiem_tra.sum()} ảnh nhận đúng")

    print("\n5.5 Cập nhật ngưỡng đã chọn vào mo_hinh/hoi_quy_logistic.joblib cho bước 6")
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
    tao_trang_xem(bang, tom_tat, bao_cao, far is not None)
    print("  -> ket_qua/xem_ket_qua.html")
    print("\nXong bước 5. Tiếp theo: python buoc6_nhan_dien.py (hoặc --web trên Jetson)")


if __name__ == "__main__":
    main()

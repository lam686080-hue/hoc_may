"""
=====================================================================
 BƯỚC 4 – DỰ ĐOÁN TRÊN TẬP TEST
 Phụ trách: Thành viên 2 – Nguyễn Văn Lâm
=====================================================================
 Làm gì   : Cho hồi quy logistic đã huấn luyện dự đoán ảnh tập TEST (ảnh mô hình chưa từng thấy)
            và ảnh người lạ kiểm tra, đo thời gian dự đoán, lấy xác suất predict_proba,
            vẽ ảnh mặt kèm tên dự đoán để quan sát.
 Đầu vào  : mo_hinh/hoi_quy_logistic.joblib (bước 3), dữ liệu bước 1-2
 Đầu ra   : ket_qua/du_doan.npz            kết quả dự đoán (dùng cho bước 5)
            ket_qua/4_du_doan_tap_test.csv tên thật + tên dự đoán cho từng ảnh
            ket_qua/4_*.png                ảnh test kèm dự đoán, biểu đồ xác suất
 Các bước : 4.1 Nạp mô hình và tập test
            4.2 Hồi quy logistic dự đoán tập test (predict_proba) + đo thời gian (batch và đơn lẻ)
            4.3 Hồi quy logistic: thống kê xác suất cao nhất (predict_proba)
            4.4 Lưu kết quả dự đoán
            4.5 Vẽ ảnh test kèm dự đoán + biểu đồ xác suất (như ví dụ AR Face)
 Chạy     : python buoc4_du_doan.py
=====================================================================
"""
import os
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score              # độ chính xác (như slide)
import chung as C
from chung import plt

rng = np.random.default_rng(C.RANDOM_STATE)      # bộ sinh số ngẫu nhiên (chọn ảnh để vẽ)


# ======================= 4.5 (b) BIỂU ĐỒ XÁC SUẤT =======================
def ve_xac_suat(X, that, P, lop):
    """Mỗi ảnh mặt kèm thanh predict_proba của 5 người có xác suất cao nhất (xanh = đúng, đỏ = sai)."""
    hang = int(np.ceil(len(X) / 2))
    fig, ax = plt.subplots(hang, 4, figsize=(13, 2.5 * hang), squeeze=False,
                           gridspec_kw={"width_ratios": [1, 1.7, 1, 1.7]})
    for a in ax.ravel():
        a.axis("off")
    for i, (x, t, p) in enumerate(zip(X, that, P)):
        a_anh, a_cot = ax[i // 2, (i % 2) * 2], ax[i // 2, (i % 2) * 2 + 1]
        a_anh.imshow(x.reshape(C.KICH_THUOC, C.KICH_THUOC), cmap="gray")
        a_anh.set_title(f"Thật: {C.rut_gon(t)}", fontsize=9)
        top = np.argsort(p)[::-1][:5]
        dung, y = lop[top[0]] == t, np.arange(len(top))[::-1]
        a_cot.axis("on")
        a_cot.barh(y, p[top], color=["seagreen" if dung else "tomato"] + ["steelblue"] * (len(top) - 1))
        a_cot.set(yticks=y, yticklabels=[C.rut_gon(lop[j]) for j in top], xlim=(0, 1))
        a_cot.set_title(f"Dự đoán: {C.rut_gon(lop[top[0]])} ({p[top[0]]:.0%})", fontsize=9,
                        color="green" if dung else "red")
        a_cot.tick_params(labelsize=8)
        for yy, v in zip(y, p[top]):
            a_cot.text(min(v + 0.02, 0.85), yy, f"{v:.2f}", va="center", fontsize=8)
    plt.suptitle("predict_proba – xác suất ảnh thuộc về từng người (người lạ: không có cột đúng)")
    C.luu_hinh("4_xac_suat_du_doan.png")


# =====================================================================
def main():
    print("=" * 70 + "\n BƯỚC 4 – DỰ ĐOÁN TRÊN TẬP TEST\n" + "=" * 70)

    print("\n4.1 Nạp mô hình và tập test")
    if not os.path.exists(C.FILE_HOI_QUY_LOGISTIC):
        raise SystemExit("Chưa có mô hình. Chạy buoc3_huan_luyen.py trước.")
    ten = "Logistic Regression"
    mh = joblib.load(C.FILE_HOI_QUY_LOGISTIC)["mo_hinh"]
    X, df = C.doc_du_lieu()
    te = (df["tap"] == "test").to_numpy()
    X_test, y_test = X[te], df.loc[te, "ten"].to_numpy()
    if len(X_test) == 0:
        raise SystemExit("Tập test rỗng. Kiểm tra lại bước 2.")
    X_la, file_la_all, st_la = C.doc_nguoi_la("nguoi_la")
    file_la = file_la_all[st_la == "ok"]
    print(f"  {ten} | test: {len(X_test)} ảnh người quen | người lạ: {len(X_la)} ảnh")

    print("\n4.2 Hồi quy logistic dự đoán tập test")
    lop = np.array(mh.classes_)
    t0 = time.perf_counter()
    P_test = mh.predict_proba(X_test)
    du_doan = lop[np.argmax(P_test, axis=1)]
    ms = (time.perf_counter() - t0) * 1000 / len(X_test)
    # Đo thêm tốc độ dự đoán từng ảnh đơn lẻ (giống thực tế khi chạy camera ở bước 6)
    so_mau = min(50, len(X_test))
    t0 = time.perf_counter()
    for i in range(so_mau):
        mh.predict_proba(X_test[i:i + 1])
    ms_don = (time.perf_counter() - t0) * 1000 / so_mau
    print(f"  Độ chính xác {accuracy_score(y_test, du_doan):.1%} | batch {ms:.2f} ms/ảnh | đơn lẻ {ms_don:.2f} ms/ảnh")

    print("\n4.3 Xác suất của hồi quy logistic")
    P_la = mh.predict_proba(X_la) if len(X_la) else np.zeros((0, len(lop)))
    p_doan = P_test.max(axis=1)
    print(f"  Xác suất cao nhất trung bình: người quen {p_doan.mean():.2f}"
          + (f" | người lạ {P_la.max(1).mean():.2f}" if len(P_la) else ""))

    print("\n4.4 Lưu kết quả dự đoán")
    os.makedirs(C.KET_QUA, exist_ok=True)
    np.savez(C.FILE_DU_DOAN, ten_mo_hinh=ten, du_doan=du_doan.astype(str), ms_moi_anh=ms, ms_don_le=ms_don,
             lop=lop.astype(str), y_test=y_test.astype(str),
             file_test=df.loc[te, "file"].to_numpy().astype(str),
             P_test=P_test, P_la=P_la, file_la=file_la.astype(str))
    pd.DataFrame({"file": df.loc[te, "file"].to_numpy(), "nguoi_that": y_test,
                  "du_doan": du_doan, "xac_suat": p_doan.round(3), "dung": du_doan == y_test}).to_csv(
        os.path.join(C.KET_QUA, "4_du_doan_tap_test.csv"), index=False, encoding="utf-8-sig")
    print("  -> ket_qua/du_doan.npz\n  -> ket_qua/4_du_doan_tap_test.csv")

    print("\n4.5 Vẽ ảnh test kèm dự đoán")
    chon = rng.choice(len(X_test), size=min(24, len(X_test)), replace=False)
    doan = du_doan[chon]
    C.ve_luoi_anh(X_test[chon], [f"Thật: {C.rut_gon(t)}" for t in y_test[chon]],
                  [f"Đoán: {C.rut_gon(d)} ({p:.0%})" for d, p in zip(doan, p_doan[chon])],
                  doan == y_test[chon], "4_du_doan_tap_test.png",
                  f"Dự đoán trên tập test – {ten} (xanh: đúng, đỏ: sai)")
    mau = chon[:5]
    X_m, y_m, P_m = X_test[mau], list(y_test[mau]), P_test[mau]
    if len(P_la):                                            # thêm 1 ảnh người lạ để so sánh
        j = int(rng.integers(len(P_la)))
        X_m, y_m, P_m = np.vstack([X_m, X_la[j:j + 1]]), y_m + ["Người lạ"], np.vstack([P_m, P_la[j:j + 1]])
    ve_xac_suat(X_m, y_m, P_m, lop)
    print("\nXong bước 4. Tiếp theo: python buoc5_danh_gia.py")


if __name__ == "__main__":
    main()

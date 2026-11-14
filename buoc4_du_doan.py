"""
=====================================================================
 BƯỚC 4 – DỰ ĐOÁN TRÊN TẬP TEST                            (Thành viên 2)
=====================================================================
 Làm gì   : Cho 7 mô hình đã huấn luyện dự đoán ảnh tập TEST (ảnh chúng chưa từng thấy),
            đo thời gian dự đoán, lấy xác suất predict_proba của mô hình tốt nhất,
            vẽ ảnh mặt kèm tên dự đoán để quan sát.
 Đầu vào  : mo_hinh/cac_mo_hinh.joblib (bước 3), dữ liệu bước 1-2
 Đầu ra   : ket_qua/du_doan.npz            kết quả dự đoán (dùng cho bước 5)
            ket_qua/4_du_doan_tap_test.csv tên thật + tên dự đoán của từng mô hình cho từng ảnh
            ket_qua/4_*.png                ảnh test kèm dự đoán, biểu đồ xác suất
 Các bước : 4.1 Nạp mô hình và tập test
            4.2 Mỗi mô hình dự đoán tập test (predict) + đo thời gian
            4.3 Mô hình tốt nhất: xác suất từng người (predict_proba), cả cho người lạ
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
import chung as C
from chung import plt

rng = np.random.default_rng(C.RANDOM_STATE)


def ve_xac_suat(X, that, P, lop):
    """Như ví dụ AR Face trong bài giảng: ảnh mặt + thanh predict_proba (5 người cao nhất)."""
    hang = int(np.ceil(len(X) / 2))
    fig, ax = plt.subplots(hang, 4, figsize=(13, 2.5 * hang), squeeze=False,
                           gridspec_kw={"width_ratios": [1, 1.7, 1, 1.7]})
    for a in ax.ravel():
        a.axis("off")
    for i in range(len(X)):
        a_anh, a_cot = ax[i // 2, (i % 2) * 2], ax[i // 2, (i % 2) * 2 + 1]
        a_anh.imshow(X[i].reshape(C.KICH_THUOC, C.KICH_THUOC), cmap="gray")
        a_anh.set_title(f"Thật: {C.rut_gon(that[i])}", fontsize=9)
        top = np.argsort(P[i])[::-1][:5]                       # 5 người có xác suất cao nhất
        dung = lop[top[0]] == that[i]
        a_cot.axis("on")
        a_cot.barh([C.rut_gon(lop[j]) for j in top][::-1], P[i, top][::-1],
                   color=(["steelblue"] * (len(top) - 1) + ["seagreen" if dung else "tomato"]))
        a_cot.set_xlim(0, 1)
        for y, v in enumerate(P[i, top][::-1]):
            a_cot.text(min(v + 0.02, 0.85), y, f"{v:.2f}", va="center", fontsize=8)
        a_cot.set_title(f"Dự đoán: {C.rut_gon(lop[top[0]])} ({P[i, top[0]]:.0%})", fontsize=9,
                        color="green" if dung else "red")
        a_cot.tick_params(labelsize=8)
    plt.suptitle("predict_proba – xác suất ảnh thuộc về từng người (người lạ: không có cột đúng)")
    C.luu_hinh("4_xac_suat_du_doan.png")


def main():
    print("=" * 70)
    print(" BƯỚC 4 – DỰ ĐOÁN TRÊN TẬP TEST")
    print("=" * 70)

    print("\n4.1 Nạp mô hình và tập test")
    if not os.path.exists(C.FILE_CAC_MO_HINH):
        raise SystemExit("Chưa có mô hình. Chạy buoc3_huan_luyen.py trước.")
    goi = joblib.load(C.FILE_CAC_MO_HINH)                   # nạp lại 7 mô hình đã lưu ở bước 3
    mo_hinh, tot_nhat = goi["mo_hinh"], goi["tot_nhat"]
    X, df = C.doc_du_lieu()
    te = (df["tap"] == "test").to_numpy()
    la = (df["tap"] == "nguoi_la").to_numpy()
    X_test, y_test = X[te], df.loc[te, "ten"].to_numpy()
    if len(X_test) == 0:
        raise SystemExit("Tập test rỗng. Kiểm tra lại bước 2.")
    print(f"  {len(mo_hinh)} mô hình | test: {len(X_test)} ảnh người quen | "
          f"người lạ: {int(la.sum())} ảnh")

    print("\n4.2 Mỗi mô hình dự đoán tập test")
    du_doan, ms = {}, {}
    for ten, mh in mo_hinh.items():
        t0 = time.perf_counter()
        du_doan[ten] = mh.predict(X_test)                    # predict: trả về tên người dự đoán
        ms[ten] = (time.perf_counter() - t0) / len(X_test) * 1000
        dung = (du_doan[ten] == y_test).sum()
        print(f"  {ten:<20} đoán đúng {dung}/{len(y_test)} ảnh | {ms[ten]:.2f} ms/ảnh")

    print(f"\n4.3 Xác suất của mô hình tốt nhất ({tot_nhat})")
    mh = mo_hinh[tot_nhat]
    lop = np.array(mh.classes_)                              # thứ tự người ứng với các cột xác suất
    P_test = mh.predict_proba(X_test)                        # xác suất thuộc từng người (tổng = 1)
    P_la = mh.predict_proba(X[la]) if la.any() else np.zeros((0, len(lop)))
    print(f"  Xác suất cao nhất trung bình: người quen {P_test.max(1).mean():.2f}"
          + (f" | người lạ {P_la.max(1).mean():.2f}" if len(P_la) else ""))

    print("\n4.4 Lưu kết quả dự đoán")
    os.makedirs(C.KET_QUA, exist_ok=True)
    ten_mh = list(mo_hinh)
    # .astype(str): lưu chữ dạng chuỗi thường (không cần pickle khi đọc lại ở bước 5)
    np.savez(C.FILE_DU_DOAN, ten_mo_hinh=np.array(ten_mh, dtype=str),
             du_doan=np.array([du_doan[t] for t in ten_mh]).astype(str),
             ms_moi_anh=np.array([ms[t] for t in ten_mh]), tot_nhat=np.array(tot_nhat, dtype=str),
             lop=lop.astype(str), y_test=y_test.astype(str),
             file_test=df.loc[te, "file"].to_numpy().astype(str),
             P_test=P_test, P_la=P_la, file_la=df.loc[la, "file"].to_numpy().astype(str))
    bang = pd.DataFrame({"file": df.loc[te, "file"].to_numpy(), "nguoi_that": y_test})
    for t in ten_mh:
        bang[t] = du_doan[t]
    bang["xac_suat_" + tot_nhat] = P_test.max(1).round(3)
    bang.to_csv(os.path.join(C.KET_QUA, "4_du_doan_tap_test.csv"), index=False,
                encoding="utf-8-sig")                        # utf-8-sig: Excel đọc đúng tiếng Việt
    print("  -> ket_qua/du_doan.npz\n  -> ket_qua/4_du_doan_tap_test.csv")

    print("\n4.5 Vẽ ảnh test kèm dự đoán")
    nguon = df.loc[te, "nguon"].to_numpy()
    lop_hoc = np.where(np.isin(nguon, C.TRONG_LOP))[0]       # người trong lớp (tự thu, chụp tay)
    lfw = np.where(nguon == "lfw")[0]
    n_lop = min(8, len(lop_hoc))                             # 8 ảnh người trong lớp + 16 ảnh LFW
    n_lfw = min(24 - n_lop, len(lfw))
    chon = np.concatenate([rng.choice(lop_hoc, n_lop, replace=False),
                           rng.choice(lfw, n_lfw, replace=False)]).astype(int)
    doan, xs = du_doan[tot_nhat][chon], P_test[chon].max(1)
    C.ve_luoi_anh(X_test[chon], [f"Thật: {C.rut_gon(t)}" for t in y_test[chon]],
                  [f"Đoán: {C.rut_gon(d)} ({p:.0%})" for d, p in zip(doan, xs)],
                  doan == y_test[chon], "4_du_doan_tap_test.png",
                  f"Dự đoán trên tập test – {tot_nhat} (xanh: đúng, đỏ: sai)")
    mau = np.concatenate([rng.choice(lop_hoc, min(2, len(lop_hoc)), replace=False),
                          rng.choice(lfw, min(3, len(lfw)), replace=False)]).astype(int)
    X_m, y_m, P_m = X_test[mau], list(y_test[mau]), P_test[mau]
    if len(P_la):                                            # thêm 1 ảnh người lạ để so sánh
        j = int(rng.integers(len(P_la)))
        X_m = np.vstack([X_m, X[la][j:j + 1]])
        y_m.append("Người lạ")
        P_m = np.vstack([P_m, P_la[j:j + 1]])
    ve_xac_suat(X_m, y_m, P_m, lop)
    print("\nXong bước 4. Tiếp theo: python buoc5_danh_gia.py")


if __name__ == "__main__":
    main()

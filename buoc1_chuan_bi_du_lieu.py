"""
=====================================================================
 BƯỚC 1 – CHUẨN BỊ DỮ LIỆU                                (Thành viên 1)
=====================================================================
 Làm gì   : Gom ảnh mặt từ 3 nguồn, đưa về CÙNG một dạng (căn chỉnh 5 điểm, xám 100x100,
            lọc Gaussian + CLAHE), thống kê + vẽ biểu đồ, lưu lại cho các bước sau.
 Đầu vào  : dataset/mat_xam_100/<Tên>/*.png   ảnh nhóm tự thu bằng thu_anh.py (đã căn chỉnh)
            dataset/anh_chup_tay/<Tên>/*.jpg  ảnh chụp tay bằng điện thoại (ảnh gốc, cỡ bất kỳ)
                 ảnh để thẳng trong <Tên>/  -> bước 2 tự chia 75% train / 25% test
                 ảnh trong <Tên>/train/ hoặc <Tên>/test/ -> chỉ dùng để train / chỉ để test
            Bộ LFW (tự tải lần đầu, ~230 MB)
 Đầu ra   : dataset/du_lieu_anh.npz           tất cả ảnh đã tiền xử lý (N x 100 x 100)
            dataset/danh_sach.csv             mỗi dòng 1 ảnh: file, tên, nguồn, lượt, tư thế
            dataset/chup_tay_100/             ảnh chụp tay đã căn chỉnh + bao_cao.csv
            ket_qua/1_*.png                   biểu đồ cho báo cáo
 Các bước : 1.1 Đọc ảnh tự thu (thu_anh.py)
            1.2 Ảnh chụp tay: căn chỉnh ảnh mới, đọc; người quá ít ảnh -> người lạ
            1.3 Chuẩn bị ảnh LFW (tải + căn chỉnh giống hệt ảnh tự thu)
            1.4 Tiền xử lý: lọc Gaussian + cân bằng sáng CLAHE
            1.5 Thống kê và trực quan hóa
            1.6 Lưu dữ liệu
 Chạy     : python buoc1_chuan_bi_du_lieu.py
            python buoc1_chuan_bi_du_lieu.py --chuan-bi   (chỉ làm 1.2 + 1.3: căn chỉnh ảnh
                                                           chụp tay mới, chuẩn bị LFW – trên laptop)
=====================================================================
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import chung as C
from chung import plt, cv2

# ======================= CẤU HÌNH BƯỚC 1 =======================
DUNG_LFW = True               # False: không dùng LFW (cần >= 2 người trong lớp)
LFW_SO_ANH_TOI_THIEU = 70     # lấy người có >= 70 ảnh trong LFW -> 7 người
LFW_TOI_DA_MOI_NGUOI = 150    # mỗi người LFW lấy tối đa 150 ảnh (cân bằng với 130 ảnh tự thu)
LFW_SO_NGUOI_LA = 30          # 30 người LFW khác làm "người lạ" (chỉ để test)
LFW_ANH_MOI_NGUOI_LA = 5      # mỗi người lạ lấy 5 ảnh
SO_ANH_TOI_THIEU = 5          # người trong lớp có < 5 ảnh: chưa đủ để học -> làm người lạ (test)
CHI_LAM_NGUOI_LA = []         # tên người chỉ dùng làm người lạ, vd ["Tran_Van_A"]
DUOI_ANH = (".jpg", ".jpeg", ".png", ".bmp", ".webp")   # loại ảnh chụp tay đọc được
KC_MAT_TOI_THIEU = 20         # ảnh chụp tay: 2 mắt cách nhau < 20 px (mặt quá nhỏ) -> bỏ
KC_MAT_CHUAN = 70             # ảnh lớn được thu nhỏ để 2 mắt cách nhau ~70 px (như ảnh camera)
# ===============================================================


# ======================= 1.1 ĐỌC ẢNH TỰ THU =======================
def doc_anh_tu_thu():
    """Tên ảnh do thu_anh.py đặt: <Tên>_<số>_L<lượt>_<tư thế>.png
    -> tách tên file ra được: người, lượt chụp (1 hoặc 2), tư thế."""
    # ảnh bị kiem_tra.py đánh dấu LOI thì bỏ qua
    p_loi = os.path.join(C.DATASET, "kiem_tra", "bao_cao.csv")
    loi = set()
    if os.path.exists(p_loi):
        bc = pd.read_csv(p_loi, encoding="utf-8-sig", dtype=str)
        if {"file", "trang_thai"} <= set(bc.columns):
            loi = set(bc.loc[bc["trang_thai"] == "LOI", "file"])
    dong, sai_ten = [], 0
    if not os.path.isdir(C.THU_MUC_TU_THU):
        print("  (không có thư mục dataset/mat_xam_100)")
        return dong
    for ten in sorted(os.listdir(C.THU_MUC_TU_THU)):
        thu_muc = os.path.join(C.THU_MUC_TU_THU, ten)
        if not os.path.isdir(thu_muc):
            continue
        for fn in sorted(os.listdir(thu_muc)):
            goc, duoi = os.path.splitext(fn)
            phan = goc.rsplit("_", 3)                 # tách 3 dấu "_" cuối: [tên, số, L1, tư thế]
            hop_le = duoi.lower() == ".png" and len(phan) == 4 and phan[2][:1] == "L" \
                and phan[2][1:].isdigit()
            if not hop_le:
                sai_ten += 1                          # không phải ảnh do thu_anh.py chụp
                continue
            if goc in loi:
                continue
            dong.append({"duong_dan": os.path.join(thu_muc, fn), "file": goc, "ten": ten,
                         "nguon": "tu_thu", "luot": int(phan[2][1:]), "tu_the": phan[3],
                         "vai_tro": "biet", "tap_co_dinh": ""})
    print(f"  Ảnh tự thu: {len(dong)} ảnh ({len(loi)} ảnh LOI bị bỏ qua)")
    if sai_ten:
        print(f"  ! Bỏ qua {sai_ten} file không đúng tên của thu_anh.py "
              "(ảnh chụp tay hãy để trong dataset/anh_chup_tay/<Tên>/)")
    return dong


# ======================= 1.2 ẢNH CHỤP TAY =======================
def doc_anh_dien_thoai(duong_dan):
    """Đọc ảnh chụp bằng điện thoại, xoay đúng chiều theo thông tin EXIF (ảnh dọc / ngang)."""
    from PIL import Image, ImageOps                    # Pillow: đọc ảnh + thông tin xoay EXIF
    with Image.open(duong_dan) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def tao_bo_tim_mat(so_mat):
    """MediaPipe Face Landmarker chế độ IMAGE (ảnh tĩnh): tìm mặt + 478 điểm mốc."""
    import thu_anh as T
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=T.tai_model()),
        running_mode=vision.RunningMode.IMAGE, num_faces=so_mat,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5))


def tim_mat(lm, bgr):
    """Trả về danh sách các mặt; mỗi mặt là mảng 478 điểm (tọa độ pixel trên ảnh bgr)."""
    import mediapipe as mp
    H, W = bgr.shape[:2]
    s = min(1.0, 1600 / max(H, W))                   # ảnh rất lớn: thu nhỏ cho MediaPipe chạy nhanh
    nho = bgr
    if s < 1:
        nho = cv2.resize(bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(nho, cv2.COLOR_BGR2RGB)          # MediaPipe cần ảnh RGB
    kq = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    # tọa độ MediaPipe là tỉ lệ 0..1 -> nhân với kích thước ảnh GỐC ra pixel
    return [np.array([(p.x * W, p.y * H) for p in m]) for m in kq.face_landmarks]


def can_chinh_anh_chup_tay():
    """dataset/anh_chup_tay/<Tên>/*.jpg -> tìm mặt -> căn chỉnh 5 điểm -> xám 100x100
    -> dataset/chup_tay_100/<Tên>/. Chỉ xử lý ảnh MỚI; kết quả từng ảnh ghi vào bao_cao.csv."""
    if not os.path.isdir(C.THU_MUC_CHUP_TAY):
        print("  (không có ảnh gốc dataset/anh_chup_tay ở máy này"
              + (" – dùng ảnh đã căn chỉnh sẵn)" if os.path.isdir(C.THU_MUC_CHUP_TAY_100) else ")"))
        return
    import thu_anh as T                                   # dùng lại: chuẩn hóa tên, căn chỉnh mặt
    # (a) liệt kê ảnh gốc: (tên người, ảnh gốc, ảnh 100x100 sẽ lưu)
    #     ảnh trong thư mục con train/ hoặc test/ được lưu vào thư mục con cùng tên
    viec = []
    for thu_muc in sorted(os.listdir(C.THU_MUC_CHUP_TAY)):
        p = os.path.join(C.THU_MUC_CHUP_TAY, thu_muc)
        if not os.path.isdir(p):
            continue
        ten = T.chuan_hoa_ten(thu_muc)                    # 'Trần Văn Á' -> 'Tran_Van_A'
        con = [x for x in sorted(os.listdir(p))
               if x.lower() in ("train", "test") and os.path.isdir(os.path.join(p, x))]
        for tm_con in [""] + con:
            pc = os.path.join(p, tm_con)
            for fn in sorted(os.listdir(pc)):
                goc, duoi = os.path.splitext(fn)
                if not os.path.isfile(os.path.join(pc, fn)):
                    continue
                if duoi.lower() in DUOI_ANH:
                    # IMG_01.jpg -> IMG_01_jpg.png (giữ cả đuôi để 2 ảnh cùng tên không đè nhau)
                    ra = T.chuan_hoa_ten(goc + "_" + duoi[1:]) + ".png"
                    dich = os.path.join(C.THU_MUC_CHUP_TAY_100, ten, tm_con.lower(), ra)
                    viec.append((ten, os.path.join(pc, fn), dich))
                elif duoi.lower() in (".heic", ".heif"):
                    print(f"  ! {thu_muc}/{fn}: ảnh HEIC (iPhone) chưa đọc được -> đổi sang JPG")
    # (b) ảnh gốc đã bị xóa -> xóa luôn ảnh căn chỉnh tương ứng
    can_giu = {os.path.normcase(d) for _, _, d in viec}
    for tm, _, files in os.walk(C.THU_MUC_CHUP_TAY_100):
        for fn in files:
            p = os.path.join(tm, fn)
            if fn.endswith(".png") and os.path.normcase(p) not in can_giu:
                os.remove(p)
    # (c) căn chỉnh ảnh mới (ảnh 100x100 chưa có hoặc cũ hơn ảnh gốc)
    p_bc = os.path.join(C.THU_MUC_CHUP_TAY_100, "bao_cao.csv")
    cu = {}
    if os.path.exists(p_bc):
        cu = {r["anh_goc"]: r for r in pd.read_csv(p_bc, encoding="utf-8-sig", dtype=str)
              .fillna("").to_dict("records")}
    def da_co(nguon, dich):        # đã có ảnh căn chỉnh, mới hơn ảnh gốc -> không làm lại
        return os.path.exists(dich) and os.path.getmtime(dich) >= os.path.getmtime(nguon)
    moi = [v for v in viec if not da_co(v[1], v[2])]
    print(f"  Ảnh chụp tay: {len(viec)} ảnh gốc, cần căn chỉnh mới {len(moi)} ảnh")
    lm = tao_bo_tim_mat(so_mat=5) if moi else None
    bao_cao = []
    for ten, nguon, dich in viec:
        ten_goc = os.path.relpath(nguon, C.THU_MUC_CHUP_TAY)
        if (ten, nguon, dich) not in moi:                 # đã căn chỉnh ở lần chạy trước
            bao_cao.append(cu.get(ten_goc) or {"anh_goc": ten_goc, "ten": ten, "trang_thai": "OK",
                                               "so_mat": "", "kc_mat_px": "", "ghi_chu": ""})
            continue
        dong = {"anh_goc": ten_goc, "ten": ten, "trang_thai": "OK", "so_mat": 0, "kc_mat_px": "",
                "ghi_chu": ""}
        bao_cao.append(dong)
        try:
            bgr = doc_anh_dien_thoai(nguon)
        except Exception as e:
            dong.update(trang_thai="KHONG_DOC_DUOC",
                        ghi_chu=f"không mở được ảnh ({type(e).__name__})")
            continue
        cac_mat = tim_mat(lm, bgr)
        dong["so_mat"] = len(cac_mat)
        if not cac_mat:
            dong.update(trang_thai="KHONG_THAY_MAT", ghi_chu="chụp lại: rõ mặt, đủ sáng")
            continue
        # nhiều mặt -> lấy mặt TO nhất (người được chụp thường ở gần nhất)
        diem = max(cac_mat, key=lambda d: np.ptp(d[:, 0]) * np.ptp(d[:, 1]))
        d5 = T.diem_5(diem)                               # 5 điểm: 2 mắt, mũi, 2 khóe miệng
        kc_mat = float(np.linalg.norm(d5[1] - d5[0]))     # khoảng cách 2 mắt (pixel)
        dong["kc_mat_px"] = round(kc_mat)
        if kc_mat < KC_MAT_TOI_THIEU:
            dong.update(trang_thai="MAT_NHO", ghi_chu="mặt quá nhỏ: chụp gần hơn")
            continue
        if kc_mat > KC_MAT_CHUAN:                         # ảnh lớn: thu nhỏ trước (chống răng cưa)
            k = KC_MAT_CHUAN / kc_mat
            H, W = bgr.shape[:2]
            bgr = cv2.resize(bgr, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA)
            d5 = d5 * [bgr.shape[1] / W, bgr.shape[0] / H]
        mat = T.can_chinh_mat(bgr, d5, C.KICH_THUOC)      # căn chỉnh giống hệt ảnh tự thu
        C.luu_png(dich, cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY))
        if len(cac_mat) > 1:
            dong.update(trang_thai="NHIEU_MAT",
                        ghi_chu="có nhiều mặt, đã lấy mặt to nhất – xem lại")
    if lm is not None:
        lm.close()
    if viec:
        bc = pd.DataFrame(bao_cao)
        os.makedirs(C.THU_MUC_CHUP_TAY_100, exist_ok=True)
        bc.to_csv(p_bc, index=False, encoding="utf-8-sig")
        loi = bc[~bc["trang_thai"].isin(["OK", "NHIEU_MAT"])]
        print(f"  Căn chỉnh được {len(bc) - len(loi)}/{len(bc)} ảnh -> dataset/chup_tay_100")
        for r in loi.head(15).itertuples():
            print(f"    ! {r.anh_goc}: {r.trang_thai} ({r.ghi_chu})")
        if len(loi) > 15:
            print(f"    ... và {len(loi) - 15} ảnh khác – xem dataset/chup_tay_100/bao_cao.csv")


def doc_anh_chup_tay():
    """Đọc ảnh chụp tay đã căn chỉnh. file = <Tên>__<tên ảnh> để không trùng giữa các người.
    tap_co_dinh = "train"/"test" nếu ảnh nằm trong thư mục con train/ hoặc test/."""
    dong = []
    if not os.path.isdir(C.THU_MUC_CHUP_TAY_100):
        return dong
    for ten in sorted(os.listdir(C.THU_MUC_CHUP_TAY_100)):
        for tm_con in ("", "train", "test"):
            tm = os.path.join(C.THU_MUC_CHUP_TAY_100, ten, tm_con)
            if not os.path.isdir(tm):
                continue
            tien_to = tm_con + "_" if tm_con else ""
            for fn in sorted(os.listdir(tm)):
                goc, duoi = os.path.splitext(fn)
                if duoi.lower() == ".png":
                    dong.append({"duong_dan": os.path.join(tm, fn),
                                 "file": f"{ten}__{tien_to}{goc}", "ten": ten,
                                 "nguon": "chup_tay", "luot": 0, "tu_the": "",
                                 "vai_tro": "biet", "tap_co_dinh": tm_con})
    print(f"  Ảnh chụp tay đã căn chỉnh: {len(dong)} ảnh của {len({d['ten'] for d in dong})} người")
    return dong


def gan_vai_tro(dong):
    """Người trong lớp có < SO_ANH_TOI_THIEU ảnh (tự thu + chụp tay) chưa đủ để học
    -> chỉ dùng làm NGƯỜI LẠ (test). Người trong CHI_LAM_NGUOI_LA cũng vậy."""
    dem = pd.Series([d["ten"] for d in dong]).value_counts()
    it_anh = sorted(t for t, n in dem.items() if n < SO_ANH_TOI_THIEU)
    for d in dong:
        if d["ten"] in it_anh or d["ten"] in CHI_LAM_NGUOI_LA:
            d["vai_tro"] = "nguoi_la"
    if it_anh:
        print(f"  ! Chưa đủ {SO_ANH_TOI_THIEU} ảnh -> tạm làm người lạ: "
              + ", ".join(f"{t} ({dem[t]})" for t in it_anh))
    return dong


# ======================= 1.3 CHUẨN BỊ ẢNH LFW =======================
def tim_lfw_goc():
    """Tìm thư mục LFW gốc: thư mục chứa các thư mục tên người (vd George_W_Bush)."""
    for goc in (C.THU_MUC_LFW_GOC, os.path.join(os.path.expanduser("~"), "scikit_learn_data")):
        if not os.path.isdir(goc):
            continue
        for tm, con, _ in os.walk(goc):
            if "George_W_Bush" in con:
                return tm
            if tm[len(goc):].count(os.sep) >= 3:
                con[:] = []
    return None


def chuan_bi_lfw():
    """Tải LFW -> chọn người -> MediaPipe tìm mặt -> căn chỉnh 5 điểm GIỐNG HỆT ảnh tự thu
    -> lưu dataset/lfw_xam_100. Chỉ làm 1 lần, các lần sau dùng lại."""
    p_ds = os.path.join(C.THU_MUC_LFW, "danh_sach.csv")
    if not DUNG_LFW or os.path.exists(p_ds):
        print("  Đã có ảnh LFW căn chỉnh sẵn" if DUNG_LFW
              else "  DUNG_LFW = False -> không dùng LFW")
        return
    goc = tim_lfw_goc()
    if goc is None:
        print("  Đang tải bộ LFW (~230 MB, chỉ 1 lần)...")
        from sklearn.datasets import fetch_lfw_people      # hàm tải bộ LFW của scikit-learn
        try:
            fetch_lfw_people(data_home=C.THU_MUC_LFW_GOC, min_faces_per_person=LFW_SO_ANH_TOI_THIEU,
                             resize=0.1)
        except Exception as e:
            raise SystemExit(f"Không tải được LFW ({e}).\nTải tay lfw-funneled.tgz tại "
                             "https://ndownloader.figshare.com/files/5976015,\n"
                             f"giải nén vào {C.THU_MUC_LFW_GOC} rồi chạy lại.")
        goc = tim_lfw_goc()
        if goc is None:
            raise SystemExit(f"Không tìm thấy ảnh LFW trong {C.THU_MUC_LFW_GOC}.")
    import thu_anh as T                     # dùng lại hàm căn chỉnh mặt của thu_anh.py
    lm = tao_bo_tim_mat(so_mat=3)
    rng = np.random.default_rng(C.RANDOM_STATE)
    tat_ca = {n: sorted(os.listdir(os.path.join(goc, n))) for n in sorted(os.listdir(goc))
              if os.path.isdir(os.path.join(goc, n))}
    biet = [n for n, f in tat_ca.items() if len(f) >= LFW_SO_ANH_TOI_THIEU]
    con_lai = [n for n, f in tat_ca.items() if 5 <= len(f) < LFW_SO_ANH_TOI_THIEU]
    so_la = min(LFW_SO_NGUOI_LA, len(con_lai))
    la = sorted(str(n) for n in rng.choice(con_lai, size=so_la, replace=False))
    # kế hoạch: (tên người, vai trò, số ảnh tối đa lấy của người đó)
    ke_hoach = ([(n, "biet", LFW_TOI_DA_MOI_NGUOI) for n in biet]
                + [(n, "nguoi_la", LFW_ANH_MOI_NGUOI_LA) for n in la])
    dong, tong = [], 0
    for n, vai_tro, toi_da in ke_hoach:
        files = tat_ca[n]
        if len(files) > toi_da:                          # chọn ngẫu nhiên tối đa toi_da ảnh
            files = sorted(str(f) for f in rng.choice(files, size=toi_da, replace=False))
        dat = 0
        for fn in files:
            tong += 1
            bgr = cv2.imdecode(np.fromfile(os.path.join(goc, n, fn), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            cac_mat = tim_mat(lm, bgr)
            if not cac_mat:                              # không tìm thấy mặt -> bỏ
                continue
            H, W = bgr.shape[:2]
            # nhiều mặt -> chọn mặt gần tâm ảnh nhất (ảnh LFW luôn đặt người được gán nhãn ở giữa)
            diem = min(cac_mat, key=lambda d: np.hypot(*(d.mean(0) - (W / 2, H / 2))))
            mat = T.can_chinh_mat(bgr, T.diem_5(diem), C.KICH_THUOC)    # căn chỉnh 5 điểm
            xam = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)                 # ảnh xám 100x100
            ten_file = os.path.splitext(fn)[0]
            C.luu_png(os.path.join(C.THU_MUC_LFW, n, ten_file + ".png"), xam)
            dong.append({"file": ten_file, "ten": n, "vai_tro": vai_tro})
            dat += 1
        if vai_tro == "biet":
            print(f"    {n:<22} {dat}/{len(files)} ảnh tìm được mặt")
    lm.close()
    pd.DataFrame(dong).to_csv(p_ds, index=False)
    with open(os.path.join(C.THU_MUC_LFW, "thong_ke.json"), "w", encoding="utf-8") as f:
        json.dump({"so_anh_doc": tong, "so_anh_tim_duoc_mat": len(dong), "nguoi_biet": biet,
                   "nguoi_la": la}, f, ensure_ascii=False, indent=2)
    print(f"  LFW: {len(dong)}/{tong} ảnh tìm được mặt và căn chỉnh -> dataset/lfw_xam_100")


def doc_anh_lfw():
    p_ds = os.path.join(C.THU_MUC_LFW, "danh_sach.csv")
    if not DUNG_LFW or not os.path.exists(p_ds):
        return []
    lfw = pd.read_csv(p_ds, dtype=str)
    dong = [{"duong_dan": os.path.join(C.THU_MUC_LFW, r.ten, r.file + ".png"), "file": r.file,
             "ten": r.ten, "nguon": "lfw", "luot": 0, "tu_the": "", "vai_tro": r.vai_tro,
             "tap_co_dinh": ""} for r in lfw.itertuples()]
    print(f"  Ảnh LFW: {len(dong)} ảnh")
    return dong


# ======================= 1.4 TIỀN XỬ LÝ =======================
def tien_xu_ly_tat_ca(dong):
    """Đọc từng ảnh -> C.tien_xu_ly (lọc Gaussian + CLAHE). Ảnh hỏng/sai kích thước thì bỏ."""
    anh, giu = [], []
    for d in dong:
        a = C.doc_anh_xam(d["duong_dan"])
        if a is not None and a.shape == (C.KICH_THUOC, C.KICH_THUOC):
            anh.append(C.tien_xu_ly(a))
            giu.append(d)
    return np.array(anh, dtype=np.uint8), pd.DataFrame(giu)


# ======================= 1.5 THỐNG KÊ, TRỰC QUAN HÓA =======================
def thong_ke_va_ve(df, anh):
    biet = df[df["vai_tro"] == "biet"]
    bang = pd.crosstab(biet["ten"], biet["nguon"])            # số ảnh mỗi người theo từng nguồn
    nguon_ds = [n for n in C.NGUON if n in bang.columns]     # các nguồn có ảnh
    # thứ tự: người có ảnh tự thu -> chỉ chụp tay -> LFW (cùng nhóm thì theo tên)
    nhom = {t: min(i for i, n in enumerate(nguon_ds) if bang.loc[t, n] > 0) for t in bang.index}
    thu_tu = sorted(bang.index, key=lambda t: (nhom[t], t))
    tong = bang.sum(axis=1)
    print("\n  Số ảnh mỗi người (người quen):")
    for t in thu_tu:
        chi_tiet = ", ".join(f"{C.NGUON[n][0].split(' (')[0].lower()} {bang.loc[t, n]}"
                             for n in nguon_ds if bang.loc[t, n] > 0)
        print(f"    {t:<24} {tong[t]:4d} ảnh ({chi_tiet})")
    print(f"  Người lạ (chỉ test): {int((df['vai_tro'] == 'nguoi_la').sum())} ảnh của "
          f"{df.loc[df['vai_tro'] == 'nguoi_la', 'ten'].nunique()} người")

    # (a) Biểu đồ thanh ngang xếp chồng: số ảnh mỗi người, tô màu theo nguồn
    plt.figure(figsize=(9, 0.42 * len(thu_tu) + 1.8))
    trai = np.zeros(len(thu_tu))
    for n in nguon_ds:
        so = np.array([bang.loc[t, n] for t in thu_tu])
        plt.barh([C.rut_gon(t) for t in thu_tu], so, left=trai, color=C.NGUON[n][1],
                 edgecolor="black", label=C.NGUON[n][0])
        trai += so
    for i, t in enumerate(thu_tu):
        plt.text(tong[t] + 1, i, str(tong[t]), va="center", fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Số ảnh")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(nguon_ds), fontsize=9)
    plt.title(f"Số ảnh mỗi người: {len(biet)} ảnh, {len(thu_tu)} người", pad=26)
    C.luu_hinh("1_so_anh_moi_nguoi.png")

    # (b) Ảnh mẫu: mỗi người 1 hàng 8 ảnh
    rng = np.random.default_rng(C.RANDOM_STATE)
    fig, ax = plt.subplots(len(thu_tu), 8, figsize=(10.4, 1.45 * len(thu_tu)), squeeze=False)
    for i, t in enumerate(thu_tu):
        idx = biet.index[biet["ten"] == t].to_numpy()
        for j, k in enumerate(rng.choice(idx, size=min(8, len(idx)), replace=False)):
            ax[i, j].imshow(anh[k], cmap="gray")
        for a in ax[i]:
            a.set_xticks([])
            a.set_yticks([])
            a.axis("off")
        ax[i, 0].axis("on")
        for s in ax[i, 0].spines.values():
            s.set_visible(False)
        ax[i, 0].set_ylabel(C.rut_gon(t), fontsize=8, rotation=0, ha="right", va="center")
    plt.suptitle("Ảnh mẫu sau tiền xử lý (căn chỉnh 5 điểm, xám 100x100, lọc Gaussian + CLAHE)")
    C.luu_hinh("1_anh_mau.png")

    # (c) Trước / sau tiền xử lý: 4 ảnh lấy lần lượt từ các nguồn + histogram mức xám
    kho = {n: list(rng.permutation(df.index[df["nguon"] == n].to_numpy())) for n in nguon_ds}
    chon = []
    while len(chon) < 4 and any(kho.values()):
        for n in nguon_ds:
            if kho[n] and len(chon) < 4:
                chon.append(kho[n].pop())
    mau_hist = rng.choice(len(df), size=min(300, len(df)), replace=False)
    goc = np.concatenate([C.doc_anh_xam(df.loc[k, "duong_dan"]).ravel() for k in mau_hist])
    fig = plt.figure(figsize=(12, 4.6))
    for j, k in enumerate(chon):
        a = fig.add_subplot(2, 6, j + 1)
        a.imshow(C.doc_anh_xam(df.loc[k, "duong_dan"]), cmap="gray", vmin=0, vmax=255)
        a.set_title("Trước" if j == 0 else "", fontsize=9, loc="left")
        a.axis("off")
        a = fig.add_subplot(2, 6, j + 7)
        a.imshow(anh[k], cmap="gray", vmin=0, vmax=255)
        a.set_title("Sau lọc + CLAHE" if j == 0 else "", fontsize=9, loc="left")
        a.axis("off")
    a = fig.add_subplot(1, 3, 3)
    a.hist(goc, bins=64, alpha=0.6, label="Trước", color="gray", edgecolor="black")
    a.hist(anh[mau_hist].ravel(), bins=64, alpha=0.6, label="Sau lọc + CLAHE", color="steelblue",
           edgecolor="black")
    a.set(xlabel="Mức xám", ylabel="Tần số", title="Histogram mức xám")
    a.legend()
    C.luu_hinh("1_can_bang_sang.png")

    # (d) Tóm tắt số: độ sáng trung bình mỗi ảnh theo nguồn (mean, median, std – như bài giảng)
    df["do_sang"] = anh.reshape(len(anh), -1).mean(1)
    print("\n  Độ sáng trung bình mỗi ảnh (sau tiền xử lý):")
    bang_sang = df.groupby("nguon")["do_sang"].agg(["count", "mean", "median", "std"])
    print(bang_sang.round(1).to_string())


# =====================================================================
def main():
    chi_chuan_bi = "--chuan-bi" in sys.argv or "--chi-lfw" in sys.argv
    print("=" * 70)
    print(" BƯỚC 1 – CHUẨN BỊ DỮ LIỆU"
          + (" (chỉ chuẩn bị ảnh chụp tay + LFW)" if chi_chuan_bi else ""))
    print("=" * 70)
    print("\n1.1 Đọc ảnh tự thu (thu_anh.py)")
    dong = doc_anh_tu_thu()

    print("\n1.2 Ảnh chụp tay")
    can_chinh_anh_chup_tay()
    dong += doc_anh_chup_tay()
    dong = gan_vai_tro(dong)

    print("\n1.3 Chuẩn bị ảnh LFW")
    chuan_bi_lfw()
    if chi_chuan_bi:
        print("\nXong phần chuẩn bị (--chuan-bi).")
        return
    dong += doc_anh_lfw()
    if not dong:
        raise SystemExit("Không có ảnh nào. Kiểm tra thư mục dataset/.")

    print("\n1.4 Tiền xử lý (lọc Gaussian + CLAHE)")
    anh, df = tien_xu_ly_tat_ca(dong)
    print(f"  {len(anh)} ảnh -> mảng {anh.shape} "
          f"(mỗi ảnh sẽ thành vector {C.KICH_THUOC * C.KICH_THUOC} chiều)")
    so_nguoi = df.loc[df["vai_tro"] == "biet", "ten"].nunique()
    if so_nguoi < 2:
        raise SystemExit("Cần ít nhất 2 người quen để phân loại. Thêm ảnh hoặc bật DUNG_LFW.")

    print("\n1.5 Thống kê và trực quan hóa")
    thong_ke_va_ve(df, anh)

    print("\n1.6 Lưu dữ liệu")
    np.savez_compressed(C.FILE_ANH, anh=anh)                     # lưu mảng ảnh (nén)
    df.drop(columns=["duong_dan", "do_sang"]).to_csv(C.FILE_DANH_SACH, index=False)
    print("  -> dataset/du_lieu_anh.npz\n  -> dataset/danh_sach.csv")
    print("\nXong bước 1. Tiếp theo: python buoc2_chia_du_lieu.py")


if __name__ == "__main__":
    main()

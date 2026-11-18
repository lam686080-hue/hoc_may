"""
=====================================================================
 BƯỚC 1 – CHUẨN BỊ DỮ LIỆU
 Phụ trách: Thành viên 1 – Bùi Trung Hiếu
=====================================================================
 Làm gì   : Gom ảnh mặt từ 3 nguồn, đưa về CÙNG một dạng (căn chỉnh 5 điểm, xám 100x100,
            lọc Gaussian + CLAHE), thống kê + vẽ biểu đồ, lưu lại cho các bước sau.
 Đầu vào  : dataset_nhe/mat_xam_100/<Tên>/*.png    ảnh nhóm tự thu bằng thu_anh_nhe.py (đã căn chỉnh)
            dataset/anh_chup_tay/<Tên>/*.jpg   ảnh chụp tay bằng điện thoại (ảnh gốc)
                (ảnh trong <Tên>/train/ hoặc <Tên>/test/: chỉ dùng để train / chỉ để test)
            Bộ LFW (tùy chọn --lfw; tải nếu chưa có, ~230 MB)
 Đầu ra   : dataset/du_lieu_anh.npz   tất cả ảnh đã tiền xử lý (N x 100 x 100)
            dataset/danh_sach.csv     mỗi dòng 1 ảnh: file, tên, nguồn, lượt, tư thế, vai trò
            dataset/chup_tay_100/     ảnh chụp tay đã căn chỉnh + bao_cao.csv
            ket_qua/1_*.png           biểu đồ cho báo cáo
 Các bước : 1.1 Đọc ảnh tự thu
            1.2 Ảnh chụp tay: căn chỉnh ảnh mới, đọc; người quá ít ảnh -> người lạ
            1.3 Chuẩn bị ảnh LFW (tải + căn chỉnh giống hệt ảnh tự thu)
            1.4 Tiền xử lý: lọc Gaussian + cân bằng sáng CLAHE
            1.5 Thống kê và trực quan hóa
            1.6 Lưu dữ liệu
 Chạy     : python buoc1_chuan_bi_du_lieu.py
            Thêm --lfw nếu muốn dùng bộ ảnh LFW.
            python buoc1_chuan_bi_du_lieu.py --chuan-bi   (chỉ làm 1.2 + 1.3, trên laptop)
=====================================================================
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import chung as C
from chung import plt, cv2

# ======================= CẤU HÌNH BƯỚC 1 =======================
DUNG_LFW = "--lfw" in sys.argv     # chỉ lấy LFW khi người chạy chọn
LFW_SO_ANH_TOI_THIEU = 70     # lấy người có >= 70 ảnh trong LFW -> 7 người
LFW_TOI_DA_MOI_NGUOI = 150    # mỗi người LFW lấy tối đa 150 ảnh (cân bằng với 130 ảnh tự thu)
LFW_SO_NGUOI_LA = 30          # 30 người LFW khác làm "người lạ" (chỉ để test)
LFW_ANH_MOI_NGUOI_LA = 5      # mỗi người lạ lấy 5 ảnh
SO_ANH_TOI_THIEU = 5          # người trong lớp có < 5 ảnh: chưa đủ để học -> làm người lạ (test)
CHI_LAM_NGUOI_LA = []         # tên người chỉ dùng làm người lạ, vd ["Tran_Van_A"]
DUOI_ANH = (".jpg", ".jpeg", ".png", ".bmp", ".webp")   # loại ảnh chụp tay đọc được
KC_MAT_TOI_THIEU = 20         # ảnh chụp tay: 2 mắt cách nhau < 20 px (mặt quá nhỏ) -> bỏ
KC_MAT_CHUAN = 70             # mặt quá to thì thu nhỏ ảnh để 2 mắt cách nhau ~70 px (như camera)


# ======================= 1.1 ĐỌC ẢNH TỰ THU =======================
def doc_anh_tu_thu():
    """Tên ảnh do thu_anh_nhe.py đặt: <Tên>_<số>_L<lượt>_<tư thế>.png -> biết người, lượt, tư thế.
    Ảnh bị kiem_tra.py đánh dấu LOI thì bỏ qua."""
    thu_muc_nhe = os.path.join(C.THU_MUC, "dataset_nhe")
    thu_muc_anh = os.path.join(thu_muc_nhe, "mat_xam_100")
    print(f"  Đọc ảnh thu nhẹ tại: {thu_muc_anh}")
    p_loi = os.path.join(thu_muc_nhe, "kiem_tra", "bao_cao.csv")
    loi = set()
    if os.path.exists(p_loi):
        bc = pd.read_csv(p_loi, encoding="utf-8-sig", dtype=str)
        loi = set(bc.loc[bc["trang_thai"] == "LOI", "file"])
    dong, sai_ten = [], 0
    for p in sorted(glob.glob(os.path.join(thu_muc_anh, "*", "*.png"))):
        goc = os.path.splitext(os.path.basename(p))[0]
        phan = goc.rsplit("_", 3)                       # tách 3 dấu "_" cuối: [tên, số, L1, tư thế]
        if len(phan) < 4 or phan[2][:1] != "L" or not phan[2][1:].isdigit():
            sai_ten += 1                                # không phải ảnh do thu_anh_nhe.py chụp
        elif goc not in loi:
            dong.append({"duong_dan": p, "file": goc, "ten": os.path.basename(os.path.dirname(p)),
                         "nguon": "tu_thu", "luot": int(phan[2][1:]), "tu_the": phan[3],
                         "vai_tro": "biet", "tap_co_dinh": ""})
    print(f"  Ảnh tự thu: {len(dong)} ảnh ({len(loi)} ảnh LOI bị bỏ qua)")
    if sai_ten:
        print(f"  ! Bỏ qua {sai_ten} file không đúng tên của thu_anh_nhe.py "
              "(ảnh chụp tay hãy để trong dataset/anh_chup_tay/<Tên>/)")
    return dong


# ======================= 1.2 ẢNH CHỤP TAY =======================
def tao_bo_tim_mat(so_mat):
    """MediaPipe Face Landmarker chế độ IMAGE (ảnh tĩnh): tìm mặt + 478 điểm mốc."""
    import thu_anh_nhe as T
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
    s = min(1.0, 1600 / max(H, W))                      # ảnh rất lớn: thu nhỏ cho MediaPipe nhanh
    nho = bgr if s == 1 else cv2.resize(bgr, (round(W * s), round(H * s)),
                                        interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(nho, cv2.COLOR_BGR2RGB)          # MediaPipe cần ảnh RGB
    kq = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    # tọa độ MediaPipe là tỉ lệ 0..1 -> nhân với kích thước ảnh GỐC ra pixel
    return [np.array([(p.x * W, p.y * H) for p in m]) for m in kq.face_landmarks]


def can_chinh_1_anh(lm, nguon, dich):
    """1 ảnh điện thoại -> ảnh mặt xám 100x100 căn chỉnh 5 điểm. Trả về kết quả (OK hoặc lỗi)."""
    import thu_anh_nhe as T
    from PIL import Image, ImageOps                     # Pillow: đọc ảnh + xoay đúng chiều (EXIF)
    try:
        with Image.open(nguon) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception:
        return "KHONG_DOC_DUOC"                         # file hỏng hoặc định dạng lạ (HEIC…)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cac_mat = tim_mat(lm, bgr)
    if not cac_mat:
        return "KHONG_THAY_MAT"
    diem = max(cac_mat, key=lambda d: np.ptp(d[:, 0]) * np.ptp(d[:, 1]))   # nhiều mặt: lấy mặt to
    d5 = T.diem_5(diem)                                 # 5 điểm: 2 mắt, mũi, 2 khóe miệng
    kc_mat = np.linalg.norm(d5[1] - d5[0])              # khoảng cách 2 mắt (pixel)
    if kc_mat < KC_MAT_TOI_THIEU:
        return "MAT_NHO"
    if kc_mat > KC_MAT_CHUAN:                           # mặt quá to: thu nhỏ ảnh (đỡ răng cưa)
        k = KC_MAT_CHUAN / kc_mat
        bgr, d5 = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), d5 * k
    mat = T.can_chinh_mat(bgr, d5, C.KICH_THUOC)        # căn chỉnh giống hệt ảnh tự thu
    C.luu_png(dich, cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY))
    return "NHIEU_MAT" if len(cac_mat) > 1 else "OK"


def can_chinh_anh_chup_tay():
    """dataset/anh_chup_tay/<Tên>/[train|test/]ảnh -> dataset/chup_tay_100/<Tên>/[train|test/]
    Chỉ xử lý ảnh chưa căn chỉnh (hoặc vừa sửa); kết quả ghi vào chup_tay_100/bao_cao.csv."""
    if not os.path.isdir(C.THU_MUC_CHUP_TAY):
        print("  (máy này không có ảnh gốc dataset/anh_chup_tay – dùng ảnh đã căn chỉnh nếu có)")
        return
    import thu_anh_nhe as T
    # (a) danh sách việc: (tên người, ảnh gốc, ảnh 100x100 sẽ lưu)
    viec = []
    for p in sorted(glob.glob(os.path.join(C.THU_MUC_CHUP_TAY, "*", "**", "*.*"), recursive=True)):
        phan = os.path.relpath(p, C.THU_MUC_CHUP_TAY).split(os.sep)   # [người, (train|test,) ảnh]
        con = phan[1].lower() if len(phan) == 3 else ""
        goc, duoi = os.path.splitext(phan[-1])
        if duoi.lower() in (".heic", ".heif"):
            print(f"  ! {'/'.join(phan)}: ảnh HEIC chưa đọc được -> đổi sang JPG")
        if len(phan) > 3 or con not in ("", "train", "test") or duoi.lower() not in DUOI_ANH:
            continue
        ten = T.chuan_hoa_ten(phan[0])                  # 'Trần Văn Á' -> 'Tran_Van_A'
        ra = T.chuan_hoa_ten(goc + "_" + duoi[1:]) + ".png"   # IMG_1.jpg -> IMG_1_jpg.png
        viec.append((ten, p, os.path.join(C.THU_MUC_CHUP_TAY_100, ten, con, ra)))
    # (b) ảnh gốc đã bị xóa -> xóa luôn ảnh đã căn chỉnh của nó
    can_giu = {os.path.normcase(d) for _, _, d in viec}
    for p in glob.glob(os.path.join(C.THU_MUC_CHUP_TAY_100, "**", "*.png"), recursive=True):
        if os.path.normcase(p) not in can_giu:
            os.remove(p)
    # (c) căn chỉnh ảnh mới: ảnh 100x100 chưa có hoặc cũ hơn ảnh gốc
    moi = [(ten, p, d) for ten, p, d in viec
           if not os.path.exists(d) or os.path.getmtime(d) < os.path.getmtime(p)]
    print(f"  Ảnh chụp tay: {len(viec)} ảnh gốc, cần căn chỉnh {len(moi)} ảnh")
    if not moi:
        return
    lm = tao_bo_tim_mat(so_mat=5)
    bc = pd.DataFrame([{"anh_goc": os.path.relpath(p, C.THU_MUC_CHUP_TAY), "ten": ten,
                        "ket_qua": can_chinh_1_anh(lm, p, dich)} for ten, p, dich in moi])
    lm.close()
    os.makedirs(C.THU_MUC_CHUP_TAY_100, exist_ok=True)
    bc.to_csv(os.path.join(C.THU_MUC_CHUP_TAY_100, "bao_cao.csv"), index=False,
              encoding="utf-8-sig")
    loi = bc[~bc["ket_qua"].isin(["OK", "NHIEU_MAT"])]
    print(f"  Căn chỉnh được {len(bc) - len(loi)}/{len(bc)} ảnh -> dataset/chup_tay_100")
    for r in loi.itertuples():
        print(f"    ! {r.anh_goc}: {r.ket_qua}")


def doc_anh_chup_tay():
    """Đọc ảnh chụp tay đã căn chỉnh. file = <Tên>__<tên ảnh> để không trùng giữa các người;
    tap_co_dinh = "train"/"test" nếu ảnh nằm trong thư mục con train/ hoặc test/."""
    dong = []
    for p in sorted(glob.glob(os.path.join(C.THU_MUC_CHUP_TAY_100, "*", "**", "*.png"),
                              recursive=True)):
        phan = os.path.relpath(p, C.THU_MUC_CHUP_TAY_100).split(os.sep)   # [tên, (train|test,) ảnh]
        con = phan[1] if len(phan) == 3 else ""
        tien_to = con + "_" if con else ""
        dong.append({"duong_dan": p, "file": f"{phan[0]}__{tien_to}{phan[-1][:-4]}",
                     "ten": phan[0], "nguon": "chup_tay", "luot": 0, "tu_the": "",
                     "vai_tro": "biet", "tap_co_dinh": con})
    print(f"  Ảnh chụp tay đã căn chỉnh: {len(dong)} ảnh")
    return dong


def gan_vai_tro(dong):
    """Người trong lớp có < SO_ANH_TOI_THIEU ảnh (tự thu + chụp tay) chưa đủ để học
    -> chỉ dùng làm NGƯỜI LẠ (test). Người trong CHI_LAM_NGUOI_LA cũng vậy."""
    dem = pd.Series([d["ten"] for d in dong], dtype=str).value_counts()
    it_anh = sorted(t for t, n in dem.items() if n < SO_ANH_TOI_THIEU)
    for d in dong:
        if d["ten"] in it_anh or d["ten"] in CHI_LAM_NGUOI_LA:
            d["vai_tro"] = "nguoi_la"
    if it_anh:
        print(f"  ! Chưa đủ {SO_ANH_TOI_THIEU} ảnh -> tạm làm người lạ: "
              + ", ".join(f"{t} ({dem[t]})" for t in it_anh))
    return dong


# ======================= 1.3 CHUẨN BỊ ẢNH LFW =======================
def thu_muc_lfw_goc():
    """Thư mục ảnh LFW gốc (mỗi người 1 thư mục); None nếu chưa tải."""
    for p in (os.path.join(C.THU_MUC_LFW_GOC, "lfw_home", "lfw_funneled"),      # tải tự động
              os.path.join(C.THU_MUC_LFW_GOC, "lfw_funneled"),                  # giải nén tay
              os.path.join(os.path.expanduser("~"), "scikit_learn_data", "lfw_home",
                           "lfw_funneled")):
        if os.path.isdir(os.path.join(p, "George_W_Bush")):
            return p
    return None


def chuan_bi_lfw():
    """Tải LFW -> chọn người -> MediaPipe tìm mặt -> căn chỉnh 5 điểm GIỐNG HỆT ảnh tự thu
    -> lưu dataset/lfw_xam_100. Chỉ làm 1 lần, các lần sau dùng lại."""
    p_ds = os.path.join(C.THU_MUC_LFW, "danh_sach.csv")
    if not DUNG_LFW or os.path.exists(p_ds):
        print("  Đã có ảnh LFW căn chỉnh sẵn" if DUNG_LFW else "  DUNG_LFW = False: không dùng")
        return
    if thu_muc_lfw_goc() is None:
        print("  Đang tải bộ LFW (~230 MB, chỉ 1 lần)...")
        from sklearn.datasets import fetch_lfw_people   # hàm tải bộ LFW của scikit-learn
        try:
            fetch_lfw_people(data_home=C.THU_MUC_LFW_GOC, min_faces_per_person=LFW_SO_ANH_TOI_THIEU,
                             resize=0.1)
        except Exception as e:
            print(f"  ! Không tải được: {e}")
    goc = thu_muc_lfw_goc()
    if goc is None:
        raise SystemExit("Chưa có LFW: tải tay lfw-funneled.tgz tại "
                         "https://ndownloader.figshare.com/files/5976015, giải nén vào du_lieu_lfw")
    import thu_anh_nhe as T
    rng = np.random.default_rng(C.RANDOM_STATE)
    thu_muc_nguoi = sorted(n for n in os.listdir(goc) if os.path.isdir(os.path.join(goc, n)))
    so_anh = {n: len(os.listdir(os.path.join(goc, n))) for n in thu_muc_nguoi}
    biet = [n for n, s in so_anh.items() if s >= LFW_SO_ANH_TOI_THIEU]
    con_lai = [n for n, s in so_anh.items() if 5 <= s < LFW_SO_ANH_TOI_THIEU]
    la = sorted(rng.choice(con_lai, size=min(LFW_SO_NGUOI_LA, len(con_lai)), replace=False))
    # kế hoạch: (người, vai trò, số ảnh tối đa)
    ke_hoach = ([(n, "biet", LFW_TOI_DA_MOI_NGUOI) for n in biet]
                + [(n, "nguoi_la", LFW_ANH_MOI_NGUOI_LA) for n in la])
    lm, dong = tao_bo_tim_mat(so_mat=3), []
    for n, vai_tro, toi_da in ke_hoach:
        files = sorted(os.listdir(os.path.join(goc, n)))
        for fn in sorted(rng.choice(files, size=min(toi_da, len(files)), replace=False)):
            bgr = T.doc_anh(os.path.join(goc, n, fn))
            cac_mat = tim_mat(lm, bgr) if bgr is not None else []
            if not cac_mat:                             # không đọc được / không thấy mặt -> bỏ
                continue
            H, W = bgr.shape[:2]
            # nhiều mặt -> lấy mặt gần tâm ảnh nhất (LFW luôn đặt người được gán nhãn ở giữa)
            diem = min(cac_mat, key=lambda d: np.hypot(*(d.mean(0) - (W / 2, H / 2))))
            mat = T.can_chinh_mat(bgr, T.diem_5(diem), C.KICH_THUOC)
            ten_file = os.path.splitext(fn)[0]
            C.luu_png(os.path.join(C.THU_MUC_LFW, n, ten_file + ".png"),
                      cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY))
            dong.append({"file": ten_file, "ten": n, "vai_tro": vai_tro})
    lm.close()
    pd.DataFrame(dong).to_csv(p_ds, index=False)
    print(f"  LFW: {len(dong)} ảnh ({len(biet)} người quen, {len(la)} người lạ) -> lfw_xam_100")


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
    bang = pd.crosstab(biet["ten"], biet["nguon"])              # số ảnh mỗi người theo từng nguồn
    bang = bang[[n for n in C.NGUON if n in bang]]             # cột: tự thu, chụp tay, LFW
    bang = bang.sort_values(list(bang.columns), ascending=False)   # người tự thu lên đầu
    print(bang.assign(tong=bang.sum(axis=1)).to_string())
    print(f"  Người lạ (chỉ test): {(df['vai_tro'] == 'nguoi_la').sum()} ảnh")

    # (a) Số ảnh mỗi người: thanh ngang xếp chồng, mỗi nguồn một màu
    ax = bang.rename(columns={n: C.NGUON[n][0] for n in bang}).rename_axis(columns=None).plot.barh(
        stacked=True, figsize=(9, 0.4 * len(bang) + 1.5), edgecolor="black",
        color=[C.NGUON[n][1] for n in bang])
    ax.invert_yaxis()
    ax.set(xlabel="Số ảnh", ylabel="",
           title=f"Số ảnh mỗi người: {len(biet)} ảnh, {len(bang)} người")
    C.luu_hinh("1_so_anh_moi_nguoi.png")

    # (b) Ảnh mẫu: mỗi người 1 hàng 8 ảnh
    rng = np.random.default_rng(C.RANDOM_STATE)
    fig, ax = plt.subplots(len(bang), 8, figsize=(10.4, 1.45 * len(bang)), squeeze=False)
    for i, ten in enumerate(bang.index):
        idx = biet.index[biet["ten"] == ten]
        chon = rng.choice(idx, size=min(8, len(idx)), replace=False)
        for j in range(8):
            ax[i, j].axis("off")
            if j < len(chon):
                ax[i, j].imshow(anh[chon[j]], cmap="gray")
        ax[i, 0].text(-8, 50, C.rut_gon(ten), ha="right", va="center", fontsize=8)
    plt.suptitle("Ảnh mẫu sau tiền xử lý (căn chỉnh 5 điểm, xám 100x100, lọc Gaussian + CLAHE)")
    C.luu_hinh("1_anh_mau.png")

    # (c) Trước / sau tiền xử lý: 4 ảnh + histogram mức xám của 200 ảnh
    chon = rng.choice(len(df), size=200 if len(df) > 200 else len(df), replace=False)
    truoc = np.array([C.doc_anh_xam(p) for p in df.loc[chon, "duong_dan"]])
    fig = plt.figure(figsize=(12, 4.6))
    for j in range(min(4, len(chon))):
        for hang, a_, nhan in ((0, truoc[j], "Trước"), (1, anh[chon[j]], "Sau lọc + CLAHE")):
            a = fig.add_subplot(2, 6, hang * 6 + j + 1)
            a.imshow(a_, cmap="gray", vmin=0, vmax=255)
            a.set_title(nhan if j == 0 else "", fontsize=9, loc="left")
            a.axis("off")
    a = fig.add_subplot(1, 3, 3)
    a.hist(truoc.ravel(), bins=64, alpha=0.6, label="Trước", color="gray")
    a.hist(anh[chon].ravel(), bins=64, alpha=0.6, label="Sau lọc + CLAHE", color="steelblue")
    a.set(xlabel="Mức xám", ylabel="Tần số", title="Histogram mức xám")
    a.legend()
    C.luu_hinh("1_can_bang_sang.png")

    # (d) Độ sáng trung bình mỗi ảnh theo nguồn (count, mean, median, std – như bài giảng)
    do_sang = pd.Series(anh.reshape(len(anh), -1).mean(1), index=df.index)
    print("\n  Độ sáng trung bình mỗi ảnh (sau tiền xử lý):")
    print(do_sang.groupby(df["nguon"]).agg(["count", "mean", "median", "std"]).round(1).to_string())


# =====================================================================
def main():
    chi_chuan_bi = "--chuan-bi" in sys.argv
    print("=" * 70)
    print(" BƯỚC 1 – CHUẨN BỊ DỮ LIỆU")
    print("=" * 70)
    print("\n1.1 Đọc ảnh tự thu (thu_anh_nhe.py)")
    dong = doc_anh_tu_thu()

    print("\n1.2 Ảnh chụp tay")
    can_chinh_anh_chup_tay()
    dong = gan_vai_tro(dong + doc_anh_chup_tay())

    print("\n1.3 Chuẩn bị ảnh LFW")
    chuan_bi_lfw()
    if chi_chuan_bi:
        print("\nXong phần chuẩn bị (--chuan-bi).")
        return
    dong += doc_anh_lfw()

    print("\n1.4 Tiền xử lý (lọc Gaussian + CLAHE)")
    anh, df = tien_xu_ly_tat_ca(dong)
    print(f"  {len(anh)} ảnh -> mảng {anh.shape} (mỗi ảnh sẽ thành vector 10.000 chiều)")
    if df.empty:
        raise SystemExit("Không có ảnh hợp lệ. Hãy chạy thu_anh_nhe.py hoặc kiểm tra dataset_nhe/mat_xam_100.")
    if df.loc[df["vai_tro"] == "biet", "ten"].nunique() < 2:
        raise SystemExit("Cần ít nhất 2 người quen để phân loại. Thêm ảnh hoặc chạy với --lfw.")

    print("\n1.5 Thống kê và trực quan hóa")
    thong_ke_va_ve(df, anh)

    print("\n1.6 Lưu dữ liệu")
    os.makedirs(C.DATASET, exist_ok=True)
    np.savez_compressed(C.FILE_ANH, anh=anh)                   # lưu mảng ảnh (nén)
    df.drop(columns="duong_dan").to_csv(C.FILE_DANH_SACH, index=False)
    print("  -> dataset/du_lieu_anh.npz\n  -> dataset/danh_sach.csv")
    print("\nXong bước 1. Tiếp theo: python buoc2_chia_du_lieu.py")


if __name__ == "__main__":
    main()

@echo off
REM =====================================================================
REM  CHAY CA 6 BUOC TREN JETSON (container orin_workspace_w7_4)
REM    [0] Laptop: can chinh anh chup tay moi + chuan bi LFW (buoc 1.2, 1.3)
REM    [1] Jetson: Buoc 1 (TV1) -> 2 (TV2) -> 3 (PCA: TV1, 7 mo hinh: TV2) -> 4 (TV2) -> 5 (TV3)
REM    [2] Lay ket qua ve laptop   [3] Bat buoc 6 nhan dien chay nen, xem tren web
REM  Dat file nay trong thu muc hoc may (canh code + dataset), chay:   .\chay_jetson.bat
REM =====================================================================
set JETSON=w8-4@192.168.0.22
set C=orin_workspace_w7_4
set GOMP=/usr/local/lib/python3.8/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
set CHAY=cd /workspace/face ^&^& export LD_PRELOAD=%GOMP%
set CODE=chung.py buoc1_chuan_bi_du_lieu.py buoc2_chia_du_lieu.py buoc3_huan_luyen.py buoc4_du_doan.py buoc5_danh_gia.py buoc6_nhan_dien.py thu_anh.py face_landmarker.task
set BUOC=python3 -u buoc1_chuan_bi_du_lieu.py ^&^& python3 -u buoc2_chia_du_lieu.py ^&^& python3 -u buoc3_huan_luyen.py ^&^& python3 -u buoc4_du_doan.py ^&^& python3 -u buoc5_danh_gia.py

echo.
echo [0/3] Tren laptop: can chinh anh chup tay moi + chuan bi anh LFW...
python buoc1_chuan_bi_du_lieu.py --chuan-bi
if errorlevel 1 (
    echo !! Loi khi chuan bi du lieu tren laptop - xem thong bao phia tren.
    exit /b 1
)
set GUI=%CODE%
if exist "dataset\mat_xam_100" set GUI=%GUI% dataset/mat_xam_100
if exist "dataset\chup_tay_100" set GUI=%GUI% dataset/chup_tay_100
if exist "dataset\lfw_xam_100" set GUI=%GUI% dataset/lfw_xam_100
if exist "dataset\kiem_tra\bao_cao.csv" set GUI=%GUI% dataset/kiem_tra/bao_cao.csv

echo.
echo [1/3] Gui du lieu + code len Jetson, chay BUOC 1 -^> 5 (vai phut)...
tar -cf - %GUI% | ssh -o ServerAliveInterval=30 %JETSON% "rm -rf ~/face && mkdir -p ~/face && tar -xf - -C ~/face && docker exec %C% rm -rf /workspace/face/dataset /workspace/face/ket_qua /workspace/face/mo_hinh && docker exec %C% mkdir -p /workspace/face && docker cp ~/face/. %C%:/workspace/face/ && docker exec %C% bash -c '%CHAY% && %BUOC%' && docker cp %C%:/workspace/face/ket_qua ~/face/ && docker cp %C%:/workspace/face/mo_hinh ~/face/"
if errorlevel 1 (
    echo !! Loi o 1 trong cac buoc 1-5 - xem dong BUOC ... cuoi cung o phia tren.
    exit /b 1
)

echo.
echo [2/3] Lay bieu do + mo hinh ve laptop, mo trang ket qua...
if exist ket_qua rmdir /s /q ket_qua
scp -r %JETSON%:~/face/ket_qua %JETSON%:~/face/mo_hinh .
start "" "ket_qua\xem_ket_qua.html"

echo.
echo [3/3] Bat BUOC 6 - nhan dien chay nen tren Jetson (mat SSH cung khong tat)...
ssh %JETSON% "docker exec %C% pkill -f nhan_dien.py; sleep 1; docker exec -d %C% bash -c '%CHAY% && python3 -u buoc6_nhan_dien.py --web > nhan_dien.log 2>&1' && sleep 12 && docker exec %C% tail -n 6 /workspace/face/nhan_dien.log"
echo.
echo ==> Mo trinh duyet:  http://192.168.0.22:5004
echo     (bat IP Webcam tren dien thoai truoc)
echo     Tat nhan dien:     .\dung_nhan_dien.bat

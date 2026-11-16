@echo off
REM Tat buoc 6 (nhan dien dang chay nen tren Jetson) + lay file diem danh ve laptop
set JETSON=w8-4@192.168.0.22
set C=orin_workspace_w7_4
ssh %JETSON% "docker exec %C% pkill -f nhan_dien.py; docker cp %C%:/workspace/face/diem_danh.csv ~/face/ 2>/dev/null; echo Da tat nhan dien."
scp %JETSON%:~/face/diem_danh.csv . 2>nul && echo Da lay diem_danh.csv ve laptop.

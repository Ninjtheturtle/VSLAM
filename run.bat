@echo off
cd /d C:\Users\nengj\OneDrive\Desktop\VSLAM

build\vslam.exe --sequence data\dataset\sequences\00

:: Useful flags:
::   --no-viz     disable Rerun visualization (faster, benchmarking)
::   --start N    start at frame N
::   --end N      stop at frame N

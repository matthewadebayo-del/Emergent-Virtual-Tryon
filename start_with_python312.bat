@echo off
echo Starting VirtualFit Backend with Python 3.12 (MediaPipe enabled)
set PYTHONPATH=C:\Users\mat_a\AppData\Local\Programs\Python\Python312\Lib\site-packages
cd /d "C:\Virtual Try-On - AWS\Emergent-Virtual-Tryon\backend"
"C:\Users\mat_a\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
pause
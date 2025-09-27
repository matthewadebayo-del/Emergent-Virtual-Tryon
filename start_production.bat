@echo off
echo Starting VirtualFit Production System
echo =====================================

echo Starting MongoDB...
start "MongoDB" C:\mongodb\bin\mongod.exe --config C:\mongodb\mongod.conf

echo Waiting for MongoDB to start...
timeout /t 5 /nobreak > nul

echo Starting VirtualFit Backend...
cd backend
python production_server.py

pause

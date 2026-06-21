@echo off
echo Deploying to EV3 brick...
scp -r "%~dp0*" robot@172.16.114.35:~/
scp -r "%~dp0..\test\*" robot@172.16.114.35:~/test/
if %ERRORLEVEL% == 0 (
    echo Done. Now run on the brick: python3 ~/ev3_server.py
) else (
    echo Deploy failed. Is the brick connected and is ev3_server.py running?
)

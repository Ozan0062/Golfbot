@echo off
echo Deploying robot/ to EV3 brick...
scp -r robot/* robot@10.65.82.35:~/
if %ERRORLEVEL% == 0 (
    echo Done. Now run on the brick: python3 ~/ev3_server.py
) else (
    echo Deploy failed. Is the brick connected and is ev3_server.py running?
)

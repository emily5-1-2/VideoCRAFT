if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ -y
    apt-get install libavformat-dev libavdevice-dev unrar
elif command -v brew >/dev/null 2>&1; then
    brew update && brew install ffmpeg libsm libxext unar
    brew install carlocab/personal/unrar
    brew link unrar
else
    echo >&2 "Cannot complete setup"
fi

if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found"
else
    pip install -r requirements.txt
fi
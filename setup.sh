if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ -y
    apt-get install libavformat-dev libavdevice-dev unrar unzip
elif command -v brew >/dev/null 2>&1; then
    brew update && brew install ffmpeg libsm libxext unar
    brew install carlocab/personal/unrar
    brew link unrar
else
    echo >&2 "Cannot complete setup"
fi

if [ ! -f transforms.py ]; then
    wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py
else
    echo "transforms.py already downloaded"
fi

if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found"
else
    pip install -r requirements.txt
fi
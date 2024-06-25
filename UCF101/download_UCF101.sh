if [ ! -d "ucf101_video_data" ]; then
    wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
    wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
    mkdir -p ucf101_video_data ucf101_test_train_splits
    unzip UCF101TrainTestSplits-RecognitionTask.zip -d ucf101_test_train_splits
    rm UCF101TrainTestSplits-RecognitionTask.zip
    unrar e UCF101.rar
    rm UCF101.rar
    mv *.avi ucf101_video_data
    echo "Organizing UCF101 dataset files..."
    python3 UCF101_splitting.py
    echo "UCF101 dataset files organized"
else
    echo "ucf101_video_data already downloaded"
fi

if [ ! -f transforms.py ]; then
    wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py
else
    echo "transforms.py already downloaded"
fi
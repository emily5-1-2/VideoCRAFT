if [ ! -d "hmdb51_video_data" ]; then
    wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
    wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
    mkdir -p hmdb51_video_data hmdb51_test_train_splits
    unrar e test_train_splits.rar hmdb51_test_train_splits
    rm test_train_splits.rar
    unrar e hmdb51_org.rar
    rm hmdb51_org.rar
    mv *.rar hmdb51_video_data
    python HMDB51_splitting.py
    rm hmdb51_video_data/*.rar
fi

if [ ! -f transforms.py ]; then
    wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py
else
    echo "transforms.py already downloaded"
fi
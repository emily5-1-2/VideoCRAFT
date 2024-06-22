if [ ! -f "hmdb51_org.rar" ]; then
    wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
    wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
fi

if [ ! -d "hmdb51_video_data" ]; then
    mkdir -p hmdb51_video_data hmdb51_test_train_splits
    unrar e test_train_splits.rar hmdb51_test_train_splits
    rm test_train_splits.rar
    unrar e hmdb51_org.rar
    rm hmdb51_org.rar
    mv *.rar hmdb51_video_data
    python HMDB51_splitting.py
    rm hmdb51_video_data/*.rar
fi
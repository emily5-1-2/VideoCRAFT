import os

for files in os.listdir('ucf101_video_data'):
    foldername = files.split('_')[1]
    os.system("mkdir -p ucf101_video_data/" + foldername)
    os.system("mv ucf101_video_data/"+ files + " ucf101_video_data/"+foldername)
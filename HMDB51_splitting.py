import os

for files in os.listdir('hmdb51_video_data'):
    foldername = files.split('.')[0]
    os.system("mkdir -p hmdb51_video_data/" + foldername)
    os.system("unrar e hmdb51_video_data/"+ files + " hmdb51_video_data/"+foldername)
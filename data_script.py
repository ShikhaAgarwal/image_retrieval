import os
from shutil import copyfile

data_input = "/Users/shikha/Downloads/dataset/img/DRESSES/Skirt/"
data_output = "/Users/shikha/Documents/Fall2018/ComputerVision/Project/dataset/"
suffix_input = "_01.jpg"
files = ["shop", "comsumer"]

for root_, dirs, files_ in os.walk(data_input):
    for d in dirs:
        path_input = os.path.join(data_input, d)
        ids = d.split('_')[1]
        for f in files:
            src = os.path.join(path_input, f+suffix_input)
            dest = os.path.join(data_output, f)
            dest = os.path.join(dest, ids+".jpg")
            print src
            print dest
            copyfile(src, dest)
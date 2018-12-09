import cv2
import os
from collections import defaultdict
from shutil import copyfile

#data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
#dest_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/train/'
#anno_file = data_dir + 'meta/Anno/list_bbox_consumer2shop.txt'
#category = 'DRESSES'

data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/'
dest_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/processed_data/'
anno_file = data_dir + 'meta/Anno/list_bbox_consumer2shop.txt'
category = 'TROUSERS'

partition_file = "dataset/list_eval_partition_train.txt"
i=0
shop_bbox = defaultdict(int)
consumer_bbox = defaultdict(int)
path_dict = defaultdict(list)

with open(anno_file,'rb') as f_in:
    for line in f_in:
        if i < 2:
            i += 1
            continue
        splitLine = line.strip().split()
        path = splitLine[0].split('/')
        bbox = splitLine[-4:]
        #if path[1] != category:
        #    continue

        img_id = path[3]
        path_dict[splitLine[0]] = bbox
        key = '/'.join(path[:-1])
        if "shop" in path[-1]:
            shop_bbox[key] += 1
        else:
            consumer_bbox[key] += 1

# path is not a list here
#print path_dict.keys()
i = 0
with open(partition_file, "rb") as p_file:
    check_set = set()
    for line in p_file:
        i += 1
        line = line.strip().split("\t")
        p_1, p_2, _ = line
        p_1_img = os.path.join(data_dir,p_1)
        p_1_img = cv2.imread(p_1_img)
        p_2_img = os.path.join(data_dir,p_2)
        p_2_img = cv2.imread(p_2_img)
  
        if p_1 not in check_set:
            if p_1 in path_dict.keys():
                #print p_1
                #if p_1 not in check_set:
                x1 = int(path_dict[p_1][0])
                y1 = int(path_dict[p_1][1])
                x2 = int(path_dict[p_1][2])
                y2 = int(path_dict[p_1][3])
                crop_img_1 = p_1_img[y1:y2, x1:x2]
                check_set.add(p_1)
                fname = p_1.split("/")[-1]
                dir_ = p_1.split("/")[:-1]
                #print dir_, fname
                dest_ = os.path.join(dest_dir,"/".join(dir_))
                
                if not os.path.exists(dest_):
                os.makedirs(dest_)
                cv2.imwrite(dest_+ "/"+fname,crop_img_1)
                #print "cropped", p_1
            else:
                fname = p_1.split("/")[-1]
                dir_ = p_1.split("/")[:-1]
                dest_ = os.path.join(dest_dir,"/".join(dir_))
                if not os.path.exists(dest_):
                    os.makedirs(dest_)
                cv2.imwrite(dest_,"/"+fname,crop_img_1)
                print "not found", p_1

        if p_2 not in check_set:
            if p_2 in path_dict.keys():
                # _2 not in check_set:
                x1 = int(path_dict[p_2][0])
                y1 = int(path_dict[p_2][1])
                x2 = int(path_dict[p_2][2])
                y2 = int(path_dict[p_2][3])
                crop_img_2 = p_2_img[y1:y2, x1:x2]
                check_set.add(p_2)
                fname = p_2.split("/")[-1]
                dir_ = p_2.split("/")[:-1]

                dest_ = os.path.join(dest_dir,"/".join(dir_))

                if not os.path.exists(dest_):
                        os.makedirs(dest_)
                cv2.imwrite(dest_+ "/"+fname,crop_img_2)
                #print "cropped", p_1

                
                #print "cropped", p_2
            else:
                fname = p_2.split("/")[-1]
                dir_ = p_2.split("/")[:-1]
                dest_ = os.path.join(dest_dir,"/".join(dir_))
                if not os.path.exists(dest_):
                    os.makedirs(dest_)
                cv2.imwrite(dest_,"/"+fname,crop_img_2)
                print "not found p_2", p_2

       # print i

        


"""
for path, bbox in path_dict.items():
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    path_list = path.split('/')
    key = '/'.join(path_list[:-1])
    if shop_bbox[key] < 1 or consumer_bbox[key] < 1:
        continue

    img_file = os.path.join(data_dir, path)
    # print img_file
    img = cv2.imread(img_file)
    crop_img = img[y1:y2, x1:x2]

    dest = '/'.join(path_list)
    dest = os.path.join(dest_dir, dest)
    #print dest
    dest_list = dest.split('/')
    dest_list = '/'.join(dest_list[:-1])
    dest_list += '/'
    #print path_list
    if not os.path.exists(dest_list):
        os.makedirs(dest_list)

    cv2.imwrite(dest, crop_img)
"""

# img = cv2.imread(img_file)
# cv2.imshow("original", img)
# crop_img = img[y1:y2, x1:x2]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)

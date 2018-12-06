import cv2
import os
from collections import defaultdict
from shutil import copyfile

data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
dest_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/train/'
anno_file = data_dir + 'meta/Anno/list_bbox_consumer2shop.txt'
category = 'DRESSES'

data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/'
dest_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/train/'
anno_file = data_dir + 'meta/Anno/list_bbox_consumer2shop.txt'
category = 'DRESSES'

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
        if path[1] != category:
            continue

        img_id = path[3]
        path_dict[splitLine[0]] = bbox
        key = '/'.join(path[:-1])
        if "shop" in path[-1]:
            shop_bbox[key] += 1
        else:
            consumer_bbox[key] += 1

# path is not a list here
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

    dest = '/'.join(path_list[2:])
    dest = os.path.join(dest_dir, dest)

    dest_list = dest.split('/')
    dest_list = '/'.join(dest_list[:-1])
    dest_list += '/'
    if not os.path.exists(dest_list):
        os.makedirs(dest_list)

    cv2.imwrite(dest, crop_img)


# img = cv2.imread(img_file)
# cv2.imshow("original", img)
# crop_img = img[y1:y2, x1:x2]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
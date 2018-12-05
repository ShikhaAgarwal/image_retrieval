import cv2
from collections import defaultdict

data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

anno_file = data_dir + 'meta/Anno/list_bbox_consumer2shop.txt'
category = 'DRESSES'
img_file = data_dir + 'train/1/00000396.jpg'

i=0
shop_bbox = defaultdict(defaultdict)
consumer_bbox = defaultdict(defaultdict)
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
        sub_category = path[2]
        img_id = path[3]
        print path
        if "shop" in path[-1]:
            shop_bbox[sub_category][img_id] = bbox
        else:
            consumer_bbox[sub_category][img_id] = bbox
            # print consumer_bbox

x1 = 035
y1 = 014
x2 = 185
y2 = 121

# img = cv2.imread(img_file)
# cv2.imshow("original", img)
# crop_img = img[y1:y2, x1:x2]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
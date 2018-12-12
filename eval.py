import csv

#data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/'
data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/'
output_file = data_dir + "output_processed.csv"

hit = 0
miss = 0
test_length = 0
with open(output_file,'rb') as f_in:
    csv_reader = csv.reader(f_in)
    result = dict(csv_reader)
    result_dict = {k:v.split(',') for k,v in result.items()}  
    test_length = len(result_dict.keys())
    for test_image, shop_images in result_dict.items():
        shop_ids = [x.split("/")[:-1] for x in shop_images]
        if test_image.split("/")[:-1] in shop_ids:
            hit += 1
        else:
            miss += 1

assert(hit+miss == test_length)
print hit, miss

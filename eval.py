
data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
# data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/copied_data/DRESSES/Dress/'
output_file = data_dir + "output"

test_images = []
matched_image = []
with open(output_file,'rb') as f_in:
    for line in f_in:
        splitLine = line.strip().split(',\t')
        test_images.append(splitLine[0])
        matched_image.append(splitLine[1])

hit = 0
miss = 0
for test, match in zip(test_images, matched_image):
	if test == match:
		hit += 1
	else:
		miss += 1

assert(hit+miss == len(test_images))
print hit, miss
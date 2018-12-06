import h5py
from sklearn import cluster
from scipy.spatial import distance
from collections import defaultdict
import numpy as np

data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
# data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/copied_data/DRESSES/Dress/'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
num_cluster = 2
clusters_out_file_name = data_dir + "clusters/skirt_cluster_name"
clusters_out_file_data = data_dir + "clusters/skirt_cluster_data"
output_file = data_dir + "output"

def read_data(file_name, dataset_name, image_dataset_name):
    with h5py.File(file_name, 'r') as hf:
        data = hf[dataset_name][:]
        img_name = hf[image_dataset_name][:]
    return data, img_name

phase = TRAIN 
dataset_name = "shop_feature"
image_dataset_name = "shop_feature_image"
file_name = data_dir + phase + '_feature.h5'
train_data, train_img_name = read_data(file_name, dataset_name, image_dataset_name)

phase = TEST
dataset_name = "consumer_feature"
image_dataset_name = "consumer_feature_image"
file_name = data_dir + phase + '_feature.h5'
test_data, test_img_name = read_data(file_name, dataset_name, image_dataset_name)

# ----- KMeans Cluster -----
print "performing kmeans"
kmeans_cluster = cluster.KMeans(n_clusters=num_cluster)
kmeans_cluster.fit(train_data)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

print cluster_centers.shape
print cluster_labels

# ----- storing cluster_label:image_name mapping in a dict and file -----
label_2_img_name = defaultdict(list)
label_2_img_data = defaultdict(list)
for i in range(len(cluster_labels)):
    img_name = train_img_name[i]
    label_2_img_name[cluster_labels[i]].append(img_name)
    label_2_img_data[cluster_labels[i]].append(train_data[i])

sorted_label_img_name = sorted(label_2_img_name.items(),key = lambda x:x[0])
sorted_label_img_data = sorted(label_2_img_data.items(),key = lambda x:x[0])
with open(clusters_out_file_name,'w') as f_out:
    for label, images in sorted_label_img_name:
        for img in images:
            f_out.write(str(label)+'\t'+img+'\n')

# with open(clusters_out_file_data,'w') as f_out:
#     for label, images in sorted_label_img_data:
#         for img in images:
#             f_out.write(str(label)+'\t'+img+'\n')

# ----- Finding the closest matched shop image for each consumer test image -----
closest_centers = kmeans_cluster.predict(test_data)
result = defaultdict(list)
for i, data in enumerate(test_data):
    min_distance = np.inf
    cluster_imgs_data = label_2_img_data[closest_centers[i]]
    cluster_imgs_name = label_2_img_name[closest_centers[i]]
    for j, img in enumerate(cluster_imgs_data):
        dist = distance.cosine(data, img)
        if dist < min_distance:
            min_distance = dist
            matched_image = cluster_imgs_name[j]
    result[test_img_name[i]] = matched_image

print result
with open(output_file, 'w') as f_out:
    for consumer, shop in result.items():
        f_out.write(consumer+',\t'+shop+'\n')
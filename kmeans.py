import h5py
from sklearn import cluster
from scipy.spatial import distance
from collections import defaultdict
import numpy as np
import bottleneck as bn 
import csv

#data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
num_cluster = 1
top_k = 1
clusters_out_file_name = data_dir + "clusters/skirt_cluster_name"
clusters_out_file_data = data_dir + "clusters/skirt_cluster_data"
output_file = data_dir + "output.csv"
mode = "shop"

data_types = [VAL]

def read_data(file_name, dataset_name, image_dataset_name):
    with h5py.File(file_name, 'r') as hf:
        data = hf[dataset_name][:]
        img_name = hf[image_dataset_name][:]
    return data, img_name

phase = TRAIN 
dataset_name = data_types[0] +  "_features"
image_dataset_name = data_types[0] + "_image_names"
file_name = data_dir + data_types[0] + "_" + mode +'_feature.h5'
train_data, train_img_name = read_data(file_name, dataset_name, image_dataset_name)
print train_img_name
print train_data.shape
#phase = TEST
#dataset_name = "consumer_feature"
#image_dataset_name = "consumer_feature_image"
mode = "comsumer"
file_name = data_dir + data_types[0] + "_" +mode + '_feature.h5'
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
print train_img_name
for i in range(len(cluster_labels)):
    img_name = train_img_name[i]
    label_2_img_name[cluster_labels[i]].append(img_name)
    label_2_img_data[cluster_labels[i]].append(train_data[i])

sorted_label_img_name = sorted(label_2_img_name.items(),key = lambda x:x[0])
sorted_label_img_data = sorted(label_2_img_data.items(),key = lambda x:x[0])
#with open(clusters_out_file_name,'w') as f_out:
 #   for label, images in sorted_label_img_name:
  #      for img in images:
   #         f_out.write(str(label)+'\t'+img+'\n')

# with open(clusters_out_file_data,'w') as f_out:
#     for label, images in sorted_label_img_data:
#         for img in images:
#             f_out.write(str(label)+'\t'+img+'\n')

# ----- Finding the closest matched shop image for each consumer test image -----
closest_centers = kmeans_cluster.predict(test_data)
result = defaultdict(list)
for i, data in enumerate(test_data):
    cluster_imgs_data = label_2_img_data[closest_centers[i]]
    cluster_imgs_name = label_2_img_name[closest_centers[i]]
    distance_data = np.zeros(len(cluster_imgs_data))    
    for j, img in enumerate(cluster_imgs_data):
        dist = distance.euclidean(data, img)
        distance_data[j] = dist
        # if dist < min_distance:
        #     min_distance = dist
        #     matched_image = cluster_imgs_name[j]
    top_k_indices = bn.argpartition(distance_data,kth=top_k)
    for idx,k in enumerate(top_k_indices):
        result[test_img_name[i]].append(cluster_imgs_name[k])
        if idx >= top_k-1:
            break
    #result[test_img_name[i]] = cluster_imgs_name[top_k_indices][:top_k]


with open(output_file, 'w') as f_out:
    writer = csv.writer(f_out)
    for consumer, shop in result.items():
        writer.writerow([consumer,",".join(shop)])

  
        #f_out.write(consumer+',\t'+shop+'\n')

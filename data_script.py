from collections import defaultdict

# data_input = "/Users/shikha/Downloads/dataset/img/DRESSES/Skirt/"
# data_output = "/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/"
# suffix_input = "_01.jpg"
# files = ["shop", "comsumer"]

# for root_, dirs, files_ in os.walk(data_input):
#     for d in dirs:
#         path_input = os.path.join(data_input, d)
#         ids = d.split('_')[1]
#         for f in files:
#             src = os.path.join(path_input, f+suffix_input)
#             dest = os.path.join(data_output, f)
#             dest = os.path.join(dest, ids+".jpg")
#             print src
#             print dest
#             copyfile(src, dest)

data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
sub_dir = 'meta/Eval/'
filename = 'list_eval_partition'
input_file = data_dir + sub_dir + filename + '.txt'
train_file = data_dir + filename + '_train.txt'
val_file = data_dir + filename + '_val.txt'
test_file = data_dir + filename + '_test.txt'

i=0
id_2_data_train = defaultdict(list)
id_2_data_val = defaultdict(list)
id_2_data_test = defaultdict(list)
with open(input_file,'rb') as f_in:
    for line in f_in:
        if i < 2:
            i += 1
            continue
        splitLine = line.strip().split()
        mode = splitLine[-1]
        idx = splitLine[-2]
        if mode == 'train':
            id_2_data_train[idx].append(splitLine[:-1])
        elif mode == 'val':
            id_2_data_val[idx].append(splitLine[:-1])
        else:
            id_2_data_test[idx].append(splitLine[:-1])

with open(train_file, 'w') as f_out:
    for data in id_2_data_train.values():
        for value in data:
            s = '\t'.join(value)
            f_out.write('{}\n'.format(s))

with open(val_file, 'w') as f_out:
    for data in id_2_data_val.values():
        for value in data:
            s = '\t'.join(value)
            f_out.write('{}\n'.format(s))

with open(test_file, 'w') as f_out:
    for data in id_2_data_test.values():
        for value in data:
            s = '\t'.join(value)
            f_out.write('{}\n'.format(s))

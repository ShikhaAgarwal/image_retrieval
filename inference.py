import model
from data_loader import data_loader

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

path = ''
data_dir = ''
batch_size = 16
data_types = [VAL]
out_features_size = 4096
dataset_name = data_types[0]

vgg_model = model.FeatureExtractor()
vgg_model.load_state_dict(torch.load(path))

dataloaders, num_samples = data_loader(data_dir, data_types, batch_size)
result = np.zeros((num_samples, out_features_size))
i = 0
for data in dataloaders[phase]:
    anchor, _, _, target, path = data
    if use_gpu:
        anchor = Variable(anchor.cuda())
    else:
        anchor = Variable(anchor)
    anchor_output, _, _ = vgg_features(anchor, pos_sample, neg_sample)
    start = i
    end = i+batch_size
    result[start:end, :] = outputs
    i += batch_size

file_name = data_dir + data_types[0] + '_feature.h5'
with h5py.File(file_name, 'w') as hf:
    hf.create_dataset(dataset_name, data=result)
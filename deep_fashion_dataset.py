import os
import torch.utils.data as data
import numpy as np
from PIL import Image

class DeepFashionDataset(data.Dataset):
    """
    root: path to the partition file in txt
    data_dir: path before /img/class/
    mode: shop, comsumer or all
    mode == all: For each sample (anchor) randomly chooses a negative sample
    """

    def __init__(self, data_dir, root, transform=None, mode='all'):      
        samples = self.make_dataset(root)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.data_dir = data_dir
        self.root = root
        self.samples = samples

        self.transform = transform
        self.mode = mode

    def loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # def find_classes(self, id_2_data):
    #     classes = set()
    #     sub_classes = set()
    #     for data in id_2_data.values():
    #         path = data[0][0]
    #         path = path.strip().split('/')
    #         classes.add(path[1])
    #         sub_classes.add(path[2])

    #     classes.sort()
    #     sub_classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}
    #     subclass_to_idx = {sub_classes[i]: i for i in range(len(sub_classes))}
    #     return classes, sub_classes, class_to_idx, subclass_to_idx

    def make_dataset(self, root):
        images = []

        with open(root,'rb') as f_in:
            for line in f_in:
                splitLine = line.strip().split()
                path = splitLine[0].strip.split('/')
                classes = path[1]
                sub_classes = path[2]
                data_tuple = (splitLine, classes, sub_classes)
                images.append(data_tuple)

        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, classes, sub_classes = self.samples[index]
        target = (classes, sub_classes)
        idx = path[-1]
        anchor = os.path.join(self.data_dir, path[0])
        pos_sample = os.path.join(self.data_dir, path[1])

        pos_sample = self.loader(pos_sample)
        if self.transform is not None:
            pos_sample = self.transform(pos_sample)
        if self.mode == 'shop':
            return pos_sample, None, None, target, idx

        anchor = self.loader(anchor)
        if self.transform is not None:
            anchor = self.transform(anchor)
        if self.mode == 'comsumer':
            return anchor, None, None, target, idx

        neg_id = np.random.choice(len(self.samples))
        while (neg_id == index):
            neg_id = np.random.choice(len(self.samples))
        neg_path = self.samples[neg_id][0][1]
        neg_sample = os.path.join(self.data_dir, neg_path)
        neg_sample = self.loader(neg_sample)

        if self.transform is not None:
            neg_sample = self.transform(neg_sample)

        return anchor, pos_sample, neg_sample, target, idx

    def __len__(self):
        return len(self.samples)


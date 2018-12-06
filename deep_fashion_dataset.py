import os
import torch.utils.data as data
import numpy as np
from PIL import Image

class DeepFashionDataset(data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, root, transform=None, mode='train'):
        classes, class_to_idx = self.find_classes(root)
        samples = self.make_dataset(root, class_to_idx)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.mode = mode

    def loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, dir_, fnames in sorted(os.walk(d)):
                for fname in sorted(dir_):
                    # if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        shop = []
        consumer = []
        for root_, dir_, files_ in os.walk(path):
            for fname in files_:
                if "shop" in fname:
                    shop.append(fname)
                else:
                    consumer.append(fname)

        anchor = os.path.join(path, consumer[np.random.choice(len(consumer))])
        if self.mode == 'test':
            return anchor, None, None, target, path

        pos_sample = os.path.join(path, shop[np.random.choice(len(shop))])
        if self.mode == 'val':
            return pos_sample, None, None, target, path

        neg_id = index
        while (neg_id == index):
            neg_id = np.random.choice(len(self.samples))
        neg_path = self.samples[neg_id][0]
        for root_, dir_, files_ in os.walk(neg_path):
            for fname in files_:
                if "shop" in fname:
                    neg_sample = os.path.join(neg_path, fname)
                    break

        anchor = self.loader(anchor)
        pos_sample = self.loader(pos_sample)
        neg_sample = self.loader(neg_sample)

        if self.transform is not None:
            anchor = self.transform(anchor)
            pos_sample = self.transform(pos_sample)
            neg_sample = self.transform(neg_sample)

        return anchor, pos_sample, neg_sample, target, path

    def __len__(self):
        return len(self.samples)


from torch.utils.data import Dataset


class ImageFolder(Dataset):
    def __init__(self,folder_path, transform ):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        #print (self.files)
        self.transform = transform
        

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))

        return img_path, img

    def __len__(self):
        return len(self.files)

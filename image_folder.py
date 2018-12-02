# from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
import glob

class MyImageFolder(ImageFolder):
    # def __init__(self, folder_path, transform ):
    #     super(ImageFolder, self).__init__(root=folder_path, transform=transform)
    #     self.files = sorted(glob.glob('%s/*.*' % folder_path))
    #     #print (self.files)
    #     self.transform = transform

    # def __getitem__(self, index):
    #     img_path = self.files[index % len(self.files)]
    #     # Extract image
    #     img = np.array(Image.open(img_path))

    #     return img_path, imgx

    # def __len__(self):
    #     return len(self.files)

    def __getitem__(self, index):
        data = super(MyImageFolder, self).__getitem__(index)
        result = []
        result.append(data[0])
        result.append(data[1])
        path = self.imgs[index][0]
        result.append(path)
        return result
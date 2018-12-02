from torchvision.datasets.folder import ImageFolder
import glob

class MyImageFolder(ImageFolder):

    def __getitem__(self, index):
        data = super(MyImageFolder, self).__getitem__(index)
        result = []
        result.append(data[0])
        result.append(data[1])
        path = self.imgs[index][0]
        result.append(path)
        return result
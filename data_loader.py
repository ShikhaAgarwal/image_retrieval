from torchvision import datasets, models, transforms
import os 
from deep_fashion_dataset import DeepFashionDataset

def data_loader(data_dir, data_type, batch_size, shuffle=True, mode='train', num_workers=4):
    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        VAL: transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        TEST: transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        x: DeepFashionDataset(
            os.path.join(data_dir, x), 
            transform=data_transforms[x],
            mode=mode
        )
        for x in data_types
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers
        )
        for x in data_types
    }



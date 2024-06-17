from torchvision import datasets, transforms
from base import BaseDataLoader


# class MinistDataset(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.VOCSegmentation(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class PascalVOC(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.VOCSegmentation(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
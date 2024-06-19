from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
import pickle
from model.model import ViT

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

class PascalVOCDataset(Dataset):
  def __init__(self, data_path):
    self.data_path = data_path
    with open(self.data_path, 'rb') as f:
      self.data = pickle.load(f)

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]
  
class PascalVOCDataLoader(DataLoader):
  def __init__(self, data_path, batch_size, shuffle=True, validation_split=0.1, num_workers=1):
    self.data_path = data_path
    self.dataset = PascalVOCDataset(self.data_path)
    super().__init__(self.dataset, batch_size, shuffle, collate_fn=default_collate, num_workers=num_workers)
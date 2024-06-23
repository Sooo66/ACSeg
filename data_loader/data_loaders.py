from torch.utils.data.dataloader import default_collate
from torch.utils.data import  Dataset
from base import BaseDataLoader
import pickle
import torch

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
    # return self.data[idx]
    (data, target) = self.data[idx]
    image, representation, attn = data
    mt, vt = target
    image = torch.from_numpy(image).float()
    image = torch.permute(image, (2, 0, 1)) # (3, 224, 224)
    representation = torch.from_numpy(representation).float()
    attn = torch.from_numpy(attn).float()
    # target = torch.from_numpy(target).to(torch.float32)

    mt = torch.from_numpy(mt)
    mt = mt.to(dtype=torch.uint8)
    vt = torch.from_numpy(vt).float()

    return (image, representation, attn), (mt, vt)
  
class PascalVOCDataLoader(BaseDataLoader):
  def __init__(self, data_path, batch_size, shuffle=True, validation_split=0.1, num_workers=1):
    self.data_path = data_path
    self.dataset = PascalVOCDataset(self.data_path)
    super().__init__(self.dataset, batch_size, shuffle, validation_split=validation_split, collate_fn=default_collate, num_workers=num_workers)
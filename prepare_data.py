import numbers
import random
import torchvision.transforms.functional as F

class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


import torch
from torchvision import datasets, transforms
import pickle
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm
import cv2
import numpy as np
from matplotlib import pyplot as plt

class ViT():
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16', 
                                                           local_files_only=True, 
                                                           image_mean = [0.485, 0.456, 0.406], 
                                                           image_std = [0.229, 0.224, 0.225]
                                                        )
        self.model = ViTModel.from_pretrained('facebook/dino-vits16', attn_implementation='eager', local_files_only=True).to('cuda')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        img = np.array(x).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = img / 255.

        x = transforms.ToTensor()(x)
        x = self.processor(x, return_tensors='pt', do_rescale=False).to('cuda')
        x = self.model(**x, output_attentions=True)
        last_hidden_state = x.last_hidden_state[:, 1:, :]
        attn = x.attentions
        return img, last_hidden_state, attn[-1]

vit_model = ViT()  # 将模型移动到GPU

trsfm = ExtRandomCrop(224, padding=0, pad_if_needed=True)
dataset = datasets.VOCSegmentation(root='data/', year='2012', image_set="train", download=False)

# 遍历dataset, 用vit_model处理每个图片并使用pickle存储结果, 将所有结果存到一个文件
# ((hidden_state, attn), target)
with open('data/processed_dataset.pkl', 'wb') as f:
    data = []
    for i in tqdm(range(len(dataset))):
        img, target = dataset[i]
        img, target = trsfm(img, target)
        image, hidden_state, attn = vit_model.forward(img)

        # target -> mt(metric), vt(visualize)
        mt = np.array(target).astype(np.uint8)
        vt = target.convert('RGB')
        vt = transforms.ToTensor()(vt).detach().cpu().numpy().astype(np.float32)

        if (i == 0):
            print(image.shape)
            print(hidden_state[-1].shape)
            print(attn[-1].shape)
            print(mt.shape)
            print(vt.shape)

        output = ((image, hidden_state[-1].detach().cpu().numpy().astype(np.float32), attn[-1].detach().cpu().numpy().astype(np.float32)), (mt, vt))
        data.append(output)

    pickle.dump(data, f)

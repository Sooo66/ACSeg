import torch
from torchvision import datasets, transforms
import pickle
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm
import cv2
import numpy as np

class ViT():
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        self.model = ViTModel.from_pretrained('facebook/dino-vits16', attn_implementation='eager').to('cuda')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        img = np.array(x)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).detach().cpu().numpy()

        x = self.processor(x, return_tensors='pt').to('cuda')
        x = self.model(**x, output_attentions=True)
        last_hidden_state = x.last_hidden_state[:, 1:, :]
        attn = x.attentions
        return img, last_hidden_state, attn[-1]

vit_model = ViT()  # 将模型移动到GPU

tgt_trsfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

dataset = datasets.VOCSegmentation(root='data/', year='2012', image_set="train", download=False, target_transform=tgt_trsfm)

# 遍历dataset, 用vit_model处理每个图片并使用pickle存储结果, 将所有结果存到一个文件
# ((hidden_state, attn), target)
with open('data/processed_dataset.pkl', 'wb') as f:
    data = []
    for i in tqdm(range(len(dataset))):
        img, target = dataset[i]
        image, hidden_state, attn = vit_model.forward(img)
        output = ((image, hidden_state[-1].detach().cpu().numpy(), attn[-1].detach().cpu().numpy()), target[-1].detach().cpu().numpy())
        data.append(output)

    pickle.dump(data, f)
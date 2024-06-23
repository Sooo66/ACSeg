import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import ViTImageProcessor, ViTModel
from transformers import CLIPProcessor, CLIPModel
from utils import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
  
class CLIP():
  def __init__(self):
    self.processer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True)
    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True)

    self.classes = []
    for cls in VOC_CLASSES:
      self.classes.append("a photo of a " + cls)

  def forward(self, x):
    # 可以处理 Tensorm (B, C, H, W)和(B, H, W, C)均可
    inputs = self.processer(text=self.classes, images=x, return_tensors="pt", padding=True)
    outputs = self.model(**inputs)
    logits_per_image = outputs.logits_per_image
    cls_per_image = torch.argmax(logits_per_image, dim=-1)
    return cls_per_image

class MHA(BaseModel):
  def __init__(self, d_model=384, num_heads=6, dropout=0.):
    super(MHA, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads

    self.wq = nn.Linear(d_model, d_model)
    self.wk = nn.Linear(d_model, d_model)
    self.wv = nn.Linear(d_model, d_model)
    self.wo = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v):
    '''
    if cross attn: q: (num_prototypes, d_model), k: (B, tokens, d_model), v: (B, tokens, d_model)
    if self attn: q: (B, tokens, d_model), k: (B, tokens, d_model), v: (B, tokens, d_model)
    '''
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
    
    if (q.dim() == 2):
      q = q.unsqueeze(0) # (1, num_protypes, d_model)
      q = torch.stack([q] * k.shape[0], dim=0) # (B, num_protypes, d_model)

    bsz = k.shape[0]

    # (B, num_protoypes, d_model) -> (B, num_heads, num_prototypes, d_model // num_heads)
    # (B, tokens, d_model) -> (B, num_heads, tokens, d_model // num_heads)
    q = q.view(bsz, -1, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)
    k = k.view(bsz, -1, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)
    v = v.view(bsz, -1, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)

    attn = (q @ k.transpose(-2, -1)) / (self.d_model ** 0.5) # (B, num_heads, num_prototypes, tokens)
    attn = F.softmax(attn, dim=-1) # (B, num_heads, num_prototypes, tokens)
    attn = self.dropout(attn)

    x = (attn @ v).transpose(1, 2).contiguous().view(bsz, -1, self.d_model) # (B, num_prototypes, d_model)

    q = q.transpose(1, 2).contiguous().view(bsz, -1, self.d_model) # shortcut

    x = q + self.wo(x) # (B, 5, d_model)
    return x
  
class FeedForward(BaseModel):
  def __init__(self, d_model, d_ff, dropout=0.):
    super(FeedForward, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, d_ff),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(d_ff, d_model),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    x = self.net(x)
    return x   

class ACGBlock(BaseModel):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.):
    super(ACGBlock, self).__init__()
    self.cross_mha = MHA(d_model, num_heads, dropout)
    self.self_mha = MHA(d_model, num_heads, dropout)
    self.ffn = FeedForward(d_model, d_ff, dropout)

    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.layernorm3 = nn.LayerNorm(d_model)

  def forward(self, prototypes, representation):
    x1 = self.cross_mha(prototypes, representation, representation)
    x1 = self.layernorm1(x1)
    x1 = x1 + prototypes
    x2 = self.self_mha(x1, x1, x1)
    x2 = self.layernorm2(x2)
    x2 = x2 + x1
    x3 = self.ffn(x2)
    x3 = self.layernorm3(x3)
    return x3 + x2
  
class ACG(BaseModel):
  def __init__(self, num_prototypes, num_layers, d_model, num_heads, d_ff, dropout):
    super(ACG, self).__init__()
    self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
    self.layers = nn.ModuleList([ACGBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

  def forward(self, representation):
    outputs = self.prototypes
    for layer in self.layers:
      outputs = layer(outputs, representation)
    return outputs
  
  
class PixelAssignment():
  def __init__(self):
    pass

  def __call__(self, concepts, representation):
    S = cosine_similarity(representation, concepts) # (B, tokens, prototypes)
    S = torch.clamp(S, min=0)

    S_expanded = S.unsqueeze(2)  # Shape: (B, tokens, 1, prototypes)
    S_transposed = S.unsqueeze(1)  # Shape: (B, 1, tokens, prototypes)

    # Element-wise multiplication and max reduction on the prototype dimension
    delta = torch.max(S_expanded * S_transposed, dim=-1)[0]  # Shape: (B, tokens, tokens)
    return S, delta

class AffinityGraph():
  '''
  @ param x: representation of the pixels with shape(Batch, tokens, d_model)
  '''
  def __init__(self):
    pass

  def __call__(self, x):
    self.W = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
    self.M = torch.zeros(x.shape[0])
    bsz = x.shape[0]
    for i in range(bsz):
      A = cosine_similarity(x[i], x[i]) # (tokens, tokens)
      A = torch.clamp(A, min=0)
      K = torch.sum(A, dim=1)
      m = A.sum()
      k_outer = torch.outer(K, K)
      W = A - (k_outer / m)
      self.W[i] = W
      self.M[i] = m
    return self.W, self.M
  
class ACSeg(nn.Module):
  def __init__(self, num_prototypes, num_layers, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.acg = ACG(num_prototypes, num_layers, d_model, num_heads, d_ff, dropout)
    self.affinity_graph = AffinityGraph()
    self.pixel_assignment = PixelAssignment()

  # x: (B, H, W, C) -> (B, tokens, d_model)
  def forward(self, x):
    concepts = self.acg(x)
    W, M = self.affinity_graph(x)
    S, delta = self.pixel_assignment(concepts, x)
    # W: (B, tokens, tokens), M: (B), S: (B, tokens, concpts), delta: (B, tokens, tokens)
    return concepts, W, delta, S, M

  
class Classifier():
  def __init__(self):
    self.clip = CLIP()

  def SplitBackgroud(self, assign, attn):
    # assign: (B, tokens)
    # attn: (B, head, tokens + 1, tokens + 1)
    
    attn, _ = torch.min(attn, dim=1)
    attn = torch.sum(attn, dim=1) # (B, tokens + 1)
    bsz = assign.shape[0]
    res = []
    for i in range(bsz):
        score = [0.0] * 5
        for j in range(assign.shape[1]):
            cls = assign[i][j]
            score[cls] += attn[i][j + 1].item() # exist the CLS token
        score = np.array(score)
        score = score[:, np.newaxis]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        min_center = centers[labels[np.argmin(score)]]

        # 将属于这个类别的都设置为 0, 其余的设置为 1
        re = [0] * 5
        for j in range(5):
          if centers[labels[j]] == min_center:
              re[j] = 0
          else:
              re[j] = 1

        res.append(re)
        # print(re)
    return np.array(res) # (B, 5)
  
  def GetMask(self, S, assign):
    # S: (B, tokens, concepts)
    # assign: (B, concepts), foreground or background
    S = S.view(-1, 14, 14, 5)
    S = S.permute(0, 3, 1, 2) # (B, 5, 14, 14)
    up_img = torch.nn.functional.interpolate(S, size=(224, 224), mode='bilinear')
    up_img = torch.argmax(up_img, dim=1) #(B, 14, 14)
    up_img = up_img + 1
    up_img.to(torch.uint8)
    bsz = up_img.shape[0]
    for i in range(bsz):
      asn = assign[i]
      for j in range(asn.shape[0]):
        if asn[j] == 0:
          up_img[i][up_img[i] == j + 1] = 0
    return up_img # (B, 224, 224) tensor
  
  def GetRegionRepresentation(assign, representation):
    # assign: (B, tokens)
    # representation: (B, tokens, d_model)
    bsz = assign.shape[0]
    n = assign.shape[1]
    d_model = representation.shape[2]
    num_concepts = 5
    
    o = torch.zeros(bsz, num_concepts, d_model, device=representation.device)
    counts = torch.zeros(bsz, num_concepts, device=representation.device)
    
    for i in range(bsz):
        for j in range(n):
            cls = assign[i][j]
            o[i][cls] += representation[i][j]
            counts[i][cls] += 1
        # 避免除0
        counts[i] = torch.max(counts[i], torch.ones_like(counts[i]))
        o[i] /= counts[i].unsqueeze(1)
    return o

  
  def GetBox(self, up_img, images):
      # image: (B, 3 224 224)
      bsz = up_img.shape[0]
      output = []
      for i in range(bsz):
        img = images[i]
        # print("image:", img.shape)
        im = up_img[i]
        o = []
        for j in range(1, 6):
          mask = im == j
          if mask.sum() == 0:
            continue
          x, y = torch.where(mask)
          x_min, x_max = x.min(), x.max()
          y_min, y_max = y.min(), y.max()
          bbx = img[:, x_min:x_max, y_min:y_max].detach().cpu().numpy().astype(np.float32)
          # crop_up_img = im[x_min:x_max, y_min:y_max].detach().cpu().numpy().astype(np.uint8)

          if bbx is None or bbx.size == 0:
            continue
          
          # mm_up_img = crop_up_img != j
          # bbx[mm_up_img] = [0, 0, 0]
          
          bbx = cv2.resize(np.transpose(bbx, (1, 2, 0)), (224, 224))
          # print("bbx:", bbx.shape)
          # bbx = torch.reshape(bbx, (224, 224))
          # normlize bbx (-std/min)
          bbx = (bbx - bbx.min()) / (bbx.max() - bbx.min())
          # mask out the pixel that != j
          bbx[bbx != j] = 0

          
          o.append((bbx, j))
        output.append(o)
      return output # (32, [])
  
  def GetCls(self, bbxs):
    concept2cls = []
    bsz = len(bbxs)
    for i in range(bsz):
      cls = {}
      # imgs = [Image.fromarray(img, 'RGB') for img, cep in bbxs[i]]
      imgs = [torch.from_numpy(img) for img, cep in bbxs[i]]
      # print(imgs[0].shape)
      imgs = torch.stack(imgs, dim=0)
      if imgs == []:
         concept2cls.append(cls)
         continue
      o = self.clip.forward(imgs)
      # o = torch.argmax(logits_per_image, dim=-1)
      for j, (img, cep) in enumerate(bbxs[i]):
        cls[cep] = o[j].item() + 1
      concept2cls.append(cls)
    return concept2cls
  
  def GetOutput(self, concept2cls, mask):
    # concept2cls: (32, {})
    # mask: (B, 224, 224)
    bsz = mask.shape[0]
    output = torch.zeros_like(mask)

    for i in range(bsz):
      cls = concept2cls[i]
      if i == 0:
        print(cls)
      non_zero_mask = mask[i] != 0
      # mapped_values = [cls[val.item()] for val in mask[i][non_zero_mask]]
      try:
        mapped_values = [cls[val.item()] for val in mask[i][non_zero_mask]]
      except KeyError as e:
        print(f"KeyError: {e} in batch index {i} with mask value {cls}")
      output[i][non_zero_mask] = torch.tensor(mapped_values, dtype=output.dtype, device=output.device)
    return output

  def CreateKNN(self, mask, target, cpts):
    # cpts: (B, 5, d_model)
    # mask: (B, 224, 224)
    # target: (B, 224, 224)
    bsz = mask.shape[0]
    k, v = [], []
    for i in range(bsz):
      for j in range(cpts.shape[1]):
        # 如果不存在当前，跳过
        mask_value = mask[i] == j + 1
        if mask_value.sum() == 0:
          continue
        target_value = target[i][mask_value]
        c = torch.bincount(target_value)
        c[0] = 0
        m = torch.argmax(c)
        k.append(cpts[i][j])
        v.append(m)
    # db = torch.stack(db, dim=0)
    k = torch.stack(k, dim=0).detach()
    v = torch.tensor(v).detach()
    return k, v
  
  def KnnRetrivel(self, concepts, key, value, mask):
    # concepts: (B, 5, d_model)
    # key: (N, d_model), value: (N, )
    bsz = concepts.shape[0]
    output = []
    for i in range(bsz):
      cpts2cls = {}
      cpts = concepts[i]
      cos = cosine_similarity(cpts, key) # (5, N)
      for j in range(cpts.shape[0]):
        p = torch.argmax(cos[j], dim=-1)
        cls = value[p]
        cpts2cls[j + 1] = cls
      
      # GetOutput
      o = mask[i].clone()
      for j in range(1, 6):
        o[o == j] = cpts2cls[j]
      output.append(o)
    return output


    

    

        














import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import ViTImageProcessor, ViTModel
from transformers import CLIPProcessor, CLIPModel
from utils import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

# Pascal数据集的所有类别和对应数字
PASCAL_CLASSES = [
  "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
  "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
  "sheep", "sofa", "train", "tvmonitor"
]


class ViT():
  def __init__(self):
    self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
    self.model = ViTModel.from_pretrained('facebook/dino-vits8', attn_implementation='eager')

  def forward(self, x):
    # x = self.processor(x, return_tensors='pt')
    x = self.model(x, output_attentions=True)
    last_hidden_state = x.last_hidden_state[:, 1:, :]
    attn = x.attentions
    return last_hidden_state, attn
  
class CLIP():
  def __init__(self):
    self.precessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    self.classes = []
    for cls in PASCAL_CLASSES:
      self.classes.append("a photo of a " + cls)

  def forward(self, x):
    inputs = self.processor(text=self.classes, images=x, return_tensors=True)
    outputs = self.model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

class MHA(BaseModel):
  def __init__(self, d_model, num_heads=6, dropout=0.):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads

    self.wq = nn.Linear(d_model, d_model)
    self.wk = nn.Linear(d_model, d_model)
    self.wv = nn.Linear(d_model, d_model)
    self.wo = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v):
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
    
    if (q.dim() == 2):
      q = q.unsqueeze(0)
      q = torch.stack([q] * k.shape[0], dim=0)

    bsz = k.shape[0]

    q = q.view(q.shape[0], -1, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)
    k = k.view(bsz, -1, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)
    v = v.view(bsz, -1, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)

    attn = (q @ k.transpose(-2, -1)) / (self.d_model ** 0.5)
    attn = F.softmax(attn, dim=-1)
    attn = self.dropout(attn)

    x = (attn @ v).transpose(1, 2).contiguous().view(bsz, -1, self.d_model) 

    q = q.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)

    x = q + self.wo(x) # (B, 5, d_model)
    return x
  
class FeedForward(BaseModel):
  def __init__(self, d_model, d_ff, dropout=0.):
    super().__init__()
    self.d_model = d_model
    self.d_ff = d_ff
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x   

class ACGBlock(BaseModel):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.):
    super().__init__()
    self.cross_mha = MHA(d_model, num_heads, dropout)
    self.self_mha = MHA(d_model, num_heads, dropout)
    self.ffn = FeedForward(d_model, d_ff, dropout)

    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.layernorm3 = nn.LayerNorm(d_model)

  def forward(self, prototypes, representation):
    x1 = self.cross_mha(prototypes, representation, representation)
    x1 = x1 + prototypes
    x1 = self.layernorm1(x1)
    x2 = self.self_mha(x1, x1, x1)
    x2 = x2 + x1
    x2 = self.layernorm2(x2)
    x3 = self.ffn(x2)
    x3 = x3 + x2
    o = self.layernorm3(x3)
    return o
  
class ACG(BaseModel):
  def __init__(self, num_prototypes, num_layers, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
    self.layers = nn.ModuleList([ACGBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

  def forward(self, representation):
    outputs = self.prototypes
    for layer in self.layers:
      outputs = layer(outputs, representation)
    return outputs
  
class PixelAssignment():
  def __init__(self, concepts, representation):
    self.concepts = concepts
    self.representation = representation

  def GetAssignmentAndDelta(self):
    S = cosine_similarity(self.representation, self.concepts) # (B, tokens, prototypes)
    S = torch.clamp(S, min=0)

    assign = torch.argmax(S, dim=-1)

    S_expanded = S.unsqueeze(2)  # Shape: (B, tokens, 1, prototypes)
    S_transposed = S.unsqueeze(1)  # Shape: (B, 1, tokens, prototypes)

    # Element-wise multiplication and max reduction on the prototype dimension
    delta = torch.max(S_expanded * S_transposed, dim=-1)[0]  # Shape: (B, tokens, tokens)
    return assign, delta
  
class AffinityGraph():
  '''
  @ param x: representation of the pixels with shape(Batch, tokens, d_model)
  '''
  def __init__(self, x):
    self.W = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
    self.M = torch.zeros(x.shape[0])
    bsz = x.shape[0]
    for i in range(bsz):
      A = cosine_similarity(x[i], x[i])
      # A = torch.maximum(A, torch.tensor(0, dtype=torch.float32))
      x = torch.clamp(x, min=0)
      K = torch.sum(A, dim=1)
      m = A.sum()
      k = K @ K.T
      W = k * A / m
      self.W[i] = W
      self.M[i] = m
  
  def GetGraph(self):
    return self.W, self.M
  
class ACSeg(nn.Module):
  def __init__(self, num_prototypes, num_layers, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.acg = ACG(num_prototypes, num_layers, d_model, num_heads, d_ff, dropout)

  # x: (B, H, W, C) -> (B, tokens, d_model)
  def forward(self, x):
    concepts = self.acg(x)
    affinity_graph = AffinityGraph(x)
    W, M = affinity_graph.GetGraph()
    pixel_assignment = PixelAssignment(concepts, x)
    assign, delta = pixel_assignment.GetDelta()

    return W, delta, assign, M



# class PixelAssignment():
#   '''
#   @ param concepts: prototypes with shape(batch, num_prototypes, d_model)
#   @ param representation: representation of the pixels with shape(Batch, tokens, d_model)
#   '''
#   def __init__(self, concepts, representation):
#     self.concepts = concepts
#     self.representation = representation

#   def GetAssignment(self):
#     # S = cosine_similarity(self.representation, self.concepts)
#     # assignment = torch.argmax(S, dim=-1)
#     # return assignment
#     bsz = self.representation.shape[0]
#     num_concepts = self.concepts.shape[1]
#     bsz_assignment = {}
#     for i in range(bsz):
#       assigment = {}
#       S = cosine_similarity(self.representation[i], self.concepts[i])
#       S = torch.argmax(S, dim=-1)
#       for j in range(num_concepts):
#         asgn = []
#         for k in range(S.shape[0]):
#           if S[k] == j:
#             # asgn.append(self.representation[i][k])
#             # asgn[f"token_{k}"] = self.representation[i][k]
#             asgn.append((k, self.representation[i][k]))
#         assigment[f"concept_{j}"] = self.concepts[i][j]
#         assigment[f"pixels_{j}"] = asgn
#       bsz_assignment[f"batch_{i}"] = assigment
#     return bsz_assignment  

def PixelAssignment(concepts, representation):
  # concept: (B, num_prototypes, d_model)
  # representation: (B, tokens, d_model)

  S = cosine_similarity(representation, concepts) # (B, tokens, prototypes)


  
class Classifier():
  # def __init__(self, attentions, assignement):
  #   self.attentions = attentions # (B, num_heads, tokens, tokens)
  #   self.attentions = torch.min(self.attentions, dim=1) # (B, tokens, tokens)
  #   self.assignment = assignement
  def __init__(self):
    pass

  def SplitBackgroud(assign, attn):
    # assign: (B, tokens)
    # attn: (B, head, tokens, tokens)
    
    attn, _ = torch.min(attn, dim=1)
    attn = torch.sum(attn, dim=-1)
    bsz = assign.shape[0]
    res = []
    for i in range(bsz):
        score = [0] * 5
        for j in range(assign.shape[1]):
            cls = assign[i][j]
            score[cls] += attn[i][j]
        
        score = [(i, s) for i, s in enumerate(score)]
        score = sorted(score, key=lambda x: x[1])
        mmax = 0
        idx = 0
        for k in range(4):
            if mmax < score[k + 1][1] - score[k][1]:
                mmax = score[k + 1][1] - score[k][1]
                idx = score[k][0]

        re = [0] * 5
        for i in range(5):
            if i <= idx:
                re[score[i][0]] = 0
            else:
                re[score[i][0]] = 1
        res.append(re)
    return np.array(res)

    

    

        














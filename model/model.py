import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import ViTImageProcessor, ViTModel
from utils import cosine_similarity

class ViT():
  def __init__(self):
    self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
    self.model = ViTModel.from_pretrained('facebook/dino-vits8', , attn_implementation='eager')

  def forward(self, x):
    x = self.processor(x)
    x = self.model(**x)
    x = x[:, 1:, :]
    return x
  
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
  '''
  @ param concepts: prototypes with shape(batch, num_prototypes, d_model)
  @ param representation: representation of the pixels with shape(Batch, tokens, d_model)
  '''
  def __init__(self, concepts, representation):
    self.concepts = concepts
    self.representation = representation

  def GetAssignment(self):
    # S = cosine_similarity(self.representation, self.concepts)
    # assignment = torch.argmax(S, dim=-1)
    # return assignment
    bsz = self.representation.shape[0]
    num_concepts = self.concepts.shape[1]
    bsz_assignment = {}
    for i in range(bsz):
      assigment = {}
      S = cosine_similarity(self.representation[i], self.concepts[i])
      S = torch.argmax(S, dim=-1)
      for j in range(num_concepts):
        asgn = []
        for k in range(S.shape[0]):
          if S[k] == j:
            # asgn.append(self.representation[i][k])
            # asgn[f"token_{k}"] = self.representation[i][k]
            asgn.append((k, self.representation[i][k]))
        assigment[f"concept_{j}"] = self.concepts[i][j]
        assigment[f"pixels_{j}"] = asgn
      bsz_assignment[f"batch_{i}"] = assigment
    return bsz_assignment  

  
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
  
class Classifier():
  def __init__(self, attentions, assignement):
    self.attentions = attentions # (B, num_heads, tokens, tokens)
    self.attentions = torch.min(self.attentions, dim=1) # (B, tokens, tokens)
    self.assignment = assignement

  def SplitBackgroud(self):
    bsz = self.attentions.shape[0]
    num_concepts = 5
    for i in range(bsz):
      assign = self.assignment[f"batch_{i}"]
      attention = self.attentions[i]
      forground_score = []
      for j in range(num_concepts):
        score = 0
        for k in assign[f"pixels_{j}"]:
          score += attention[k[0]].sum()
        forground_score.append(score)







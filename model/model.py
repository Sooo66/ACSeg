import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import ViTImageProcessor, ViTModel

class ViT():
  def __init__(self):
    self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
    self.model = ViTModel.from_pretrained('facebook/dino-vits8')

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
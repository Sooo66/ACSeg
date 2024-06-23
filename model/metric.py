import torch
import numpy as np

def fast_hist(a, b, n):
  k = (a >= 0) & (a < n)
  return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)
  
def mIoU(output, label):
  output = output.detach().cpu().numpy()
  # print(f'max_out: {output.max()}')
  label = label.detach().cpu().numpy()
  # print(output.shape)
  # print(label.shape)
  n = 21
  bsz = output.shape[0]
  hist = np.zeros((n, n))
  for i in range(bsz):
    hist += fast_hist(label[i].flatten(), output[i].flatten(), n)
  return 100 * np.nanmean(per_class_iu(hist))

def mPA(output, label):
  output = output.detach().cpu().numpy()
  label = label.detach().cpu().numpy()
  n = 21
  bsz = output.shape[0]
  hist = np.zeros((n, n))
  for i in range(bsz):
    hist += fast_hist(label[i].flatten(), output[i].flatten(), n)
  return 100 * np.nanmean(per_class_PA_Recall(hist))

def Accuracy(output, label):
  output = output.detach().cpu().numpy()
  label = label.detach().cpu().numpy()
  n = 21
  bsz = output.shape[0]
  hist = np.zeros((n, n))
  for i in range(bsz):
    hist += fast_hist(label[i].flatten(), output[i].flatten(), n)
  return 100 * np.nanmean(per_Accuracy(hist))
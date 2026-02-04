# Source: https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/utils/clip_loss.py
# Author: Oezguen Turgut
# License: MIT

from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

class CLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float,
               lambda_0: float = 0.5) -> None:
    super(CLIPLoss, self).__init__()

    self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1-lambda_0

  def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)

    #logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
    logits = torch.matmul(out0, out1.T) / self.temperature
    labels = torch.arange(len(out0), device=out0.device)
    
    loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
    loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
    loss = loss_0 + loss_1
  
    return loss, logits, labels


class MultiModalCLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.

  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, temperature: float, lambda_ecg_cmr1: float = 0.33, 
              lambda_ecg_cmr2: float = 0.33, lambda_cmr1_cmr2: float = 0.33):
      
      super().__init__()

      self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
      self.temperature = temperature

      total = lambda_ecg_cmr1 + lambda_ecg_cmr2 + lambda_cmr1_cmr2
      self.lambda_ecg_cmr1 = lambda_ecg_cmr1 / total
      self.lambda_ecg_cmr2 = lambda_ecg_cmr2 / total  
      self.lambda_cmr1_cmr2 = lambda_cmr1_cmr2 / total

  def forward(self, ecg_out, cmr1_out, cmr2_out):
      # Normalize embeddings
      ecg_out = F.normalize(ecg_out, dim=1)
      cmr1_out = F.normalize(cmr1_out, dim=1)
      cmr2_out = F.normalize(cmr2_out, dim=1)
      
      labels = torch.arange(len(ecg_out), device=ecg_out.device)
      
      # ECG <-> CMR1 pair
      logits_ecg_cmr1 = torch.matmul(ecg_out, cmr1_out.T) / self.temperature
      loss_ecg_cmr1 = self.lambda_ecg_cmr1 * (
          self.cross_entropy(logits_ecg_cmr1, labels) + 
          self.cross_entropy(logits_ecg_cmr1.T, labels)
      ) / 2
      
      # ECG <-> CMR2 pair  
      logits_ecg_cmr2 = torch.matmul(ecg_out, cmr2_out.T) / self.temperature
      loss_ecg_cmr2 = self.lambda_ecg_cmr2 * (
          self.cross_entropy(logits_ecg_cmr2, labels) + 
          self.cross_entropy(logits_ecg_cmr2.T, labels)
      ) / 2
      
      # CMR1 <-> CMR2 pair
      logits_cmr1_cmr2 = torch.matmul(cmr1_out, cmr2_out.T) / self.temperature
      loss_cmr1_cmr2 = self.lambda_cmr1_cmr2 * (
          self.cross_entropy(logits_cmr1_cmr2, labels) + 
          self.cross_entropy(logits_cmr1_cmr2.T, labels)
      ) / 2
      
      total_loss = loss_ecg_cmr1 + loss_ecg_cmr2 + loss_cmr1_cmr2

      return total_loss, logits_ecg_cmr1, labels  # or return all logits if needed
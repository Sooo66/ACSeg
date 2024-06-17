import torch.nn.functional as F
import torch

class ModularityLoss(torch.nn.Module):
    def __init__(self, m):
        super(ModularityLoss, self).__init__()
        self.m = m

    def forward(self, x, y):
      '''
      @ param x: The result of pixel assignment, (B, tokens, prototypes)
      @ param y: The Affinity Graph used to supervision, (B, tokens, tokens)
      @ param m: A scalar value used in the final loss computation, (B, 1)
      '''
      x = torch.clamp(x, min=0)

      x_expanded = x.unsqueeze(2)  # Shape: (B, tokens, 1, prototypes)
      x_transposed = x.unsqueeze(1)  # Shape: (B, 1, tokens, prototypes)

      # Element-wise multiplication and max reduction on the prototype dimension
      delta = torch.max(x_expanded * x_transposed, dim=-1)[0]  # Shape: (B, tokens, tokens)

      # Compute the loss
      loss = torch.sum(delta * y)
      loss = loss * (-1 / (2 * self.m))
      
      return loss.sum()



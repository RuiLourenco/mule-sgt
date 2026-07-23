import torch
import numpy as np

data = torch.zeros(2, 2, 4, 4, 1)
gradients = torch.zeros(2, 2, 4, 4, 3)

validMask = torch.zeros(2, 2, 4, 4, 1, dtype=torch.bool)
validMask[0, 0, 1:3, 1:3, 0] = True

squeezedMask = ~validMask.squeeze(-1)

# Poison data
data = torch.where(validMask, data, torch.tensor(float('nan')))
# Poison gradients
gradients = torch.where(squeezedMask.unsqueeze(-1), torch.tensor(float('nan')), gradients)

print("data isnan sum:", data.isnan().sum().item())
print("gradients isnan sum:", gradients.isnan().sum().item())


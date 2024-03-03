import torch
import torch.nn as nn
import torch.nn.functional as F

class SwishGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(SwishGLU, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.fc = nn.Linear(input_dim, hidden_dim * 2)
        
    def forward(self, x):
        x = self.fc(x)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return x1 * torch.sigmoid(x2)

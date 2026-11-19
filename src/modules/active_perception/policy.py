import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadActionPolicy(nn.Module):
    def __init__(self, state_dim=28, action_dim=9, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.head_q = nn.Linear(hidden_dim, 1)
        self.head_c = nn.Linear(hidden_dim, 1)
        self.head_i = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action_one_hot):
        """
        state: (B, 28)
        action_one_hot: (B, 9)
        """
        x = torch.cat([state, action_one_hot], dim=1)
        feat = self.shared(x)
        
        dq_hat = self.head_q(feat).squeeze(1)
        dc_hat = self.head_c(feat).squeeze(1)
        i_logit = self.head_i(feat).squeeze(1)
        
        return dq_hat, dc_hat, i_logit

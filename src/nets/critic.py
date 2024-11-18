from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch 


class GNNCritic(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + 1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCriticTD3(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, layer_norm=False):
        super().__init__()
        self.act_dim = act_dim
        self.layer_norm = layer_norm

        self.conv1 = GCNConv(in_channels, in_channels)

        self.lin1 = nn.Linear(in_channels + 1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

        if self.layer_norm:
            self.lin1_norm = nn.LayerNorm(hidden_size)
            self.lin2_norm = nn.LayerNorm(hidden_size)

        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        if not self.layer_norm:
            x = F.relu(self.lin1(concat))
            x = F.relu(self.lin2(x))  # (B, N, H)
            x = torch.sum(x, dim=1)  # (B, H)
            x = self.lin3(x).squeeze(-1)  # (B)
        else:
            x = F.relu(self.lin1_norm(self.lin1(concat)))
            x = F.relu(self.lin2_norm(self.lin2(x)))
            x = torch.sum(x, dim=1)
            x = self.lin3(x)
            x = x.squeeze(-1)
            
        return x

class GNNCriticLSTM(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lstm = nn.LSTM(in_channels + 1, hidden_size, dropout=0.3)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        x, _ = self.lstm(concat)
        x = F.relu(self.lin1(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin2(x).squeeze(-1)  # (B)
        return x
    

class GNNValue(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x 
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
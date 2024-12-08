import torch
import torch.nn as nn
import torch.nn.functional as F

class MTFNN(nn.Module):

    def __init__(self, server_num, user_num, hidden_dim=256, layer_num=5):
        super(MTFNN, self).__init__()
        self.server_num = server_num
        self.user_num = user_num
        self.out_dim = server_num * user_num * 3     # categorical prob + continuous ratio
        self.input_size = server_num * user_num * 6  # edge_index + 5-dim attrs

        self.input_layer = nn.Linear(self.input_size, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out_layer = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x_i = F.relu(layer(x))
            x = x + x_i
        x = self.out_layer(x)

        x1, x2 = x[:, :self.server_num * self.user_num * 2], x[:, self.server_num * self.user_num * 2:]
        x1 = x1.reshape(batch_size, -1, 2)
        x1 = F.softmax(x1, dim=-1)
        x2 = torch.sigmoid(x2)
        return x1, x2


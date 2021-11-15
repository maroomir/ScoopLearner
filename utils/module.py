import torch
from torch import tensor
from torch.nn import Module


class LSTM(Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_layer: int,
                 dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, output_size, batch_first=True, dropout=dropout)
        self.layer_num = num_layer

    def forward(self, x: tensor):
        batch, time, feature = x.size()
        if time <= 1:
            return x
        if self.layer_num != 0:
            odds = []
            evens = []
            for i in range(time // 2):
                # Reduce the time resolution by half
                odds.append(2 * i)
                evens.append(2 * i + 1)
            x = torch.cat((x[:, odds, :], x[:, evens, :]), dim=-1)
        x, hidden = self.lstm(x)
        return x


class StateContainer(Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_layer: int,
                 dropout=0.0):
        super(StateContainer, self).__init__()
        modules = []
        for i in range(num_layer):
            if i == 0:
                modules.append(LSTM(input_size, output_size, i, dropout))
            else:
                modules.append(LSTM(output_size * 2, output_size, i, dropout))
        self.network = torch.nn.Sequential(*modules)

    def forward(self, x: tensor):
        return self.network(x)

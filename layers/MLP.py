import torch
import torch.nn as nn
from layers.Embed import DataEmbedding


class Projector(nn.Module):

    def __init__(self, enc_in, seq_len, hidden_dims=[16, 16], hidden_layers=2, output_dim=1, kernel_size=3):
        super(Projector, self).__init__()

        d_model = 128
        self.emd = DataEmbedding(enc_in, d_model)

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        # layers = [nn.Linear(enc_in * seq_len, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU()]
        layers = [nn.Linear(d_model, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.BatchNorm1d(hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False), nn.Sigmoid()]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.emd(x, None)
        x = self.series_conv(x)  # B x 1 x E
        x = x.view(batch_size, -1)  # B x E
        y = self.backbone(x)    # B x O
        y = y.unsqueeze(2)      # B x O x 1

        return y

import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    """Fully connected networks of the Figure 2 experiment (2, 4 or 6 layers).

    The input image (3 x img_size x img_size) is flattened; hidden widths follow the
    configurations used in the paper: 256 (2 layers), 512-512-256 (4 layers) and
    512-512-512-512-256 (6 layers).
    """

    WIDTHS = {2: [256], 4: [512, 512, 256], 6: [512, 512, 512, 512, 256]}

    def __init__(self, num_classes=10, n_layers=4, img_size=128):
        super().__init__()
        if n_layers not in self.WIDTHS:
            raise ValueError(f"n_layers must be one of {sorted(self.WIDTHS)}")
        dims = [3 * img_size * img_size] + self.WIDTHS[n_layers] + [num_classes]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

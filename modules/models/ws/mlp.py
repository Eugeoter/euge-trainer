import torch.nn as nn


class MLP4(nn.Module):
    def __init__(self, input_size=768, batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048) if batch_norm else nn.Identity(),
            nn.Dropout(0.3),

            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512) if batch_norm else nn.Identity(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if batch_norm else nn.Identity(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.Dropout(0.1),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class MLP5(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.GELU(),
            nn.Dropout(0.5),

            ResidualBlock(2048, 512, dropout_rate=0.4),
            ResidualBlock(512, 256, dropout_rate=0.3),
            ResidualBlock(256, 128, dropout_rate=0.2),

            nn.Linear(128, 32),
            nn.GELU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.linear(x) + self.shortcut(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256, dropout_prob=0.2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(feature_dim) for _ in range(3)]
        )

    def forward(self, x):
        return self.res_blocks(self.downsample(x))




def compute_correlations_stack(encoded_feats):
    """
    encoded_feats: Tensor of shape [K=6, D, H, W]
    Returns: List of 5 tensors of shape [(H*W, H*W)]
    """
    K, D, H, W = encoded_feats.shape
    output = []
    for i in range(K - 1):
        f1 = encoded_feats[i].view(D, H * W).T  # [H*W, D]
        f2 = encoded_feats[i + 1].view(D, H * W)  # [D, H*W]
        cor = f1 @ f2  # [H*W, H*W]
        cor = cor.view(-1, H, W)  # Ensure shape is [H*W, H*W]
        output.append(cor)
    # make the outputs in a one tensor
    output = torch.stack(output, dim=0)  # [K-1, H*W, H*W]
    
    return output  # List of 5 tensors, each (H*W x H*W)


class CompressionEncoding(nn.Module):
    def __init__(self, in_channels, compressed_channels=512, dropout_prob=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(1024, compressed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(compressed_channels, compressed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(compressed_channels, compressed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(compressed_channels, compressed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x):
        x = torch.cat(list(x), dim=0)
        x = x.unsqueeze(0)
        return self.encoder(x)



class MotionFeatureEncoder(nn.Module):
    def __init__(self, in_channels=512, out_dim=2048, num_blocks=9, dropout_prob=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            *[ResidualBlock(in_channels) for _ in range(num_blocks)],
            nn.Dropout2d(p=dropout_prob)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout_fc(x)
        return self.fc(x)



class IMFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder(in_channels=3, feature_dim=256)
        self.compression_encoding = CompressionEncoding(in_channels=5*784, compressed_channels=512)
        self.motion_feature_encoder = MotionFeatureEncoder(in_channels=512, out_dim=2048)

    def forward(self, x):  # x: [6, 3, 256, 344]
        x = self.feature_encoder(x)  # [6, 256, 32, 43]
        x = compute_correlations_stack(x)  # List of 5 tensors, each (H*W x H*W)
        x = self.compression_encoding(x)
        x = self.motion_feature_encoder(x)
        return x.squeeze(0)  # [2048]

class BatchedIMFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = IMFE()

    def forward(self, x):  # x: [B, K, C, H, W]
        result = []
        for i in range(x.shape[0]):
            result.append(self.model(x[i]))
        return torch.stack(result, dim=0)
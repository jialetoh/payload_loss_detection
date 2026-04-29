import torch
import torch.nn as nn
import torchvision.models as models


class SiameseLossDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Load MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.encoder = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze encoder for deployment
        for param in self.encoder.parameters():
            param.requires_grad = False

        # MLP (Concat |f1-f2| and f1*f2)
        self.mlp = nn.Sequential(
            nn.Linear(1152, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        """Passes a single image through the encoder to get the 576-dim embedding."""
        x = self.encoder(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward_mlp(self, f1, f2):
        """Takes two pre-computed embeddings and runs the comparison math."""
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        combined = torch.cat((diff, prod), dim=1) # 576 + 576 = 1152
        return self.mlp(combined)

    def forward(self, ref, curr):
        """Used during training when caching isn't required."""
        f1 = self.forward_one(ref)
        f2 = self.forward_one(curr)
        return self.forward_mlp(f1, f2)
        
import torch
import torch.nn as nn




class ConvEncoder(nn.Module):
    """
    Convolutional encoder for image data.

    Architecture:
    - Conv2d(1, 16, kernel=5, stride=1, padding=2) + ReLU
    - Conv2d(16, 16, kernel=5, stride=1, padding=2) + ReLU
    - MaxPool2d(kernel=2, stride=2)
    - Conv2d(16, 32, kernel=3, stride=1, padding=1) + ReLU
    - Conv2d(32, 32, kernel=3, stride=1, padding=1) + ReLU
    - MaxPool2d(kernel=2, stride=2)
    - Conv2d(32, 64, kernel=3, stride=1, padding=1) + ReLU
    - Conv2d(64, 64, kernel=3, stride=1, padding=1) + ReLU
    - MaxPool2d(kernel=2, stride=2)
    - (Optional) Flatten

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space
    """

    def __init__(self, latent_dim=32):
        super(ConvEncoder, self).__init__()

        self.net = nn.Sequential(
            # Input: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # Output: (batch_size, 16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2), # Output: (batch_size, 16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch_size, 16, 14, 14)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # Output: (batch_size, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # Output: (batch_size, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch_size, 32, 7, 7)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (batch_size, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # Output: (batch_size, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch_size, 64, 3, 3)

            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=0), # Output: (batch_size, 64, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        h = self.net(x) # Output shape: (batch_size, latent_dim, 1, 1)
        h = h.view(h.size(0), -1) # Flatten to (batch_size, latent_dim)
        h = self.head(h)
        return h



class PixelShuffleDecoder(nn.Module):
    """
    Decoder using PixelShuffle (sub-pixel convolution) for upsampling.

    Projects latent vector to a small spatial map, then progressively
    upsamples using Conv2d + manual pixel shuffle (view+permute).
    Avoids ConvTranspose2d (OOM in auto_LiRPA) and large Linear layers.

    Architecture (~23K params vs ~215K for HybridDecoder):
        Linear(latent_dim, 8*7*7) → reshape (8, 7, 7)
        → Conv2d(8, 64, 3) + BN + ReLU → pixel_shuffle(2) → (16, 14, 14)
        → Conv2d(16, 32, 3) + BN + ReLU → pixel_shuffle(2) → (8, 28, 28)
        → Conv2d(8, 8, 3) + BN + ReLU  (refinement)
        → Conv2d(8, 1, 3) + Sigmoid → (1, 28, 28)

    Uses manual view+permute instead of nn.PixelShuffle to avoid
    onnx::DepthToSpace which auto_LiRPA may not support.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space
    """

    def __init__(self, latent_dim=32):
        super(PixelShuffleDecoder, self).__init__()

        # Project to small spatial map: (8, 7, 7)
        self.fc = nn.Linear(latent_dim, 8 * 7 * 7)

        # Stage 1: (8, 7, 7) → (64, 7, 7) → pixel_shuffle → (16, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Stage 2: (16, 14, 14) → (32, 14, 14) → pixel_shuffle → (8, 28, 28)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Refinement at full resolution: (8, 28, 28) → (8, 28, 28)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        # Output: (8, 28, 28) → (1, 28, 28)
        self.conv_out = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _pixel_shuffle(x, r=2):
        """Manual pixel shuffle via view+permute (auto_LiRPA compatible)."""
        b, c, h, w = x.shape
        x = x.view(b, c // (r * r), r, r, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // (r * r), h * r, w * r)
        return x

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = x.view(-1, 8, 7, 7)

        x = self.conv1(x)           # (b, 64, 7, 7)
        x = self._pixel_shuffle(x)  # (b, 16, 14, 14)

        x = self.conv2(x)           # (b, 32, 14, 14)
        x = self._pixel_shuffle(x)  # (b, 8, 28, 28)

        x = self.conv_refine(x)     # (b, 8, 28, 28)
        return self.conv_out(x)     # (b, 1, 28, 28)



class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for image data. Combines ConvEncoder and ConvDecoder.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space
    """

    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = PixelShuffleDecoder(latent_dim=latent_dim)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28)

        Returns
        -------
        torch.Tensor
            Latent tensor of shape (batch_size, latent_dim)
        torch.Tensor
            Reconstructed tensor of shape (batch_size, 1, 28, 28)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

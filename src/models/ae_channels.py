import torch
import torch.nn as nn





class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Input: (1, 28, 28)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # (16, 14, 14)
            # nn.BatchNorm2d(16),
            nn.GroupNorm(4, 16), # (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2), # (16, 14, 14)
            # nn.BatchNorm2d(16),
            nn.GroupNorm(4, 16), # (16, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (16, 7, 7)

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (32, 4, 4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # (32, 4, 4)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64, 2, 2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
        )

        self.fc_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
    
    def forward(self, x):
        x = self.encoder(x) # (batch_size, 64, 1, 1)
        x = x.view(x.size(0), -1) # (batch_size, 64)
        x = self.head(x) # (batch_size, 128)
        mu = self.fc_mu(x) # (batch_size, latent_dim)
        logvar = self.fc_logvar(x) # (batch_size, latent_dim)
        return mu, logvar

            



class PixelShuffleDecoder(nn.Module):
    """
    Decoder using PixelShuffle (sub-pixel convolution) for upsampling.

    Takes a flat latent vector and upsamples to (1, 28, 28) using
    Linear projection + Conv2d + manual pixel shuffle (view+permute).
    Avoids ConvTranspose2d (OOM in auto_LiRPA).

    Architecture:
        Input: (latent_dim,)
        → Linear(latent_dim, 16*7*7) + ReLU → reshape (16, 7, 7)
        → Conv2d(16, 64, 3) + BN + ReLU → pixel_shuffle(2) → (16, 14, 14)
        → Conv2d(16, 64, 3) + BN + ReLU → pixel_shuffle(2) → (16, 28, 28)
        → Conv2d(16, 8, 3) + BN + ReLU  (refinement)
        → Conv2d(8, 1, 3) + Sigmoid → (1, 28, 28)

    Uses manual view+permute instead of nn.PixelShuffle to avoid
    onnx::DepthToSpace which auto_LiRPA may not support.
    """

    def __init__(self, latent_dim=32):
        super(PixelShuffleDecoder, self).__init__()

        # Project flat latent → spatial: (latent_dim,) → (16, 7, 7)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16 * 7 * 7),
            nn.ReLU(),
        )

        # Stage 1: (16, 7, 7) → (64, 7, 7) → pixel_shuffle → (16, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Stage 2: (16, 14, 14) → (64, 14, 14) → pixel_shuffle → (16, 28, 28)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Refinement at full resolution: (16, 28, 28) → (8, 28, 28)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
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

    def forward(self, z):
        x = self.fc(z)              # (b, 16*7*7)
        x = x.view(-1, 16, 7, 7)   # (b, 16, 7, 7)

        x = self.conv1(x)           # (b, 64, 7, 7)
        x = self._pixel_shuffle(x)  # (b, 16, 14, 14)

        x = self.conv2(x)           # (b, 64, 14, 14)
        x = self._pixel_shuffle(x)  # (b, 16, 28, 28)

        x = self.conv_refine(x)     # (b, 8, 28, 28)
        return self.conv_out(x)     # (b, 1, 28, 28)



class ConvAutoencoder(nn.Module):
    """
    Convolutional VAE for image data. Combines ConvEncoder (with mu/logvar heads)
    and PixelShuffleDecoder.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space
    """

    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = PixelShuffleDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + eps * std using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28)

        Returns
        -------
        mu : torch.Tensor
            Mean of shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Log-variance of shape (batch_size, latent_dim)
        reconstructed : torch.Tensor
            Reconstructed tensor of shape (batch_size, 1, 28, 28)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return mu, logvar, reconstructed

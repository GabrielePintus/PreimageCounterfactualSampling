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

    def forward(self, x): return self.net(x).view(x.size(0), -1) # Flatten to (batch_size, latent_dim)



class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvDecoder, self).__init__()
        self.latent_dim = latent_dim

        # Note: no nn.Unflatten — we reshape manually in forward() to avoid
        # onnx::Mod operation that auto_LiRPA doesn't support
        self.net = nn.Sequential(
            # 1. (latent_dim, 1, 1) -> (32, 3, 3)
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            # 2. (32, 3, 3) -> (16, 7, 7)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            # 3. (16, 7, 7) -> (8, 14, 14)
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),

            # 4. (8, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x): return self.net(x.view(-1, self.latent_dim, 1, 1))



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
        self.decoder = ConvDecoder(latent_dim=latent_dim)

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

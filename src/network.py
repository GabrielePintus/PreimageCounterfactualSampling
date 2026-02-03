import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(28*28, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 10)
        # )
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x): return self.net(x)


class SimpleCNNClassifier(nn.Module):
    
    def __init__(self):
        super(SimpleCNNClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            
            nn.ReLU(),
            nn.Linear(32 *2 *2, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x


import torch


class TestDownSampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(6),
            torch.nn.Conv2d(6, 12, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.Conv2d(12, 24, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(24),
            torch.nn.Conv2d(24, 48, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.Conv2d(48, 96, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.Conv2d(96, 192, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(192),
            torch.nn.Conv2d(192, 384, (3, 3)),
            torch.nn.ReLU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        y = self.layers(img)
        return y


class TestUpSampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(384, 192, (3, 3), (2, 2), (0, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(192),
            torch.nn.ConvTranspose2d(192, 96, (3, 3), (2, 2), (1, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.ConvTranspose2d(96, 48, (3, 3), (2, 2), (1, 1), (0, 0)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.ConvTranspose2d(48, 24, (3, 3), (2, 2), (1, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(24),
            torch.nn.ConvTranspose2d(24, 12, (3, 3), (2, 2), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.ConvTranspose2d(12, 6, (3, 3), (2, 2), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(6),
            torch.nn.ConvTranspose2d(6, 6, (3, 3), (2, 2), (1, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(6),
            torch.nn.ConvTranspose2d(6, 3, (3, 3), (2, 2), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.layers(img)


class TestDownSamplerExpand(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 12, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.Conv2d(12, 48, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.Conv2d(48, 192, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(192),
            torch.nn.Conv2d(192, 768, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(768),
            torch.nn.Conv2d(768, 3072, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3072),
            torch.nn.Conv2d(3072, 3072, (3, 3), (2, 2), 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3072),
            torch.nn.Conv2d(3072, 3072, (3, 3)),
            torch.nn.ReLU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        y = self.layers(img)
        return y


class TestUpSamplerExpand(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3072, 3072, (3, 3), (2, 2), (0, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3072),
            torch.nn.ConvTranspose2d(3072, 3072, (3, 3), (2, 2), (1, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3072),
            torch.nn.ConvTranspose2d(3072, 768, (3, 3), (2, 2), (1, 1), (0, 0)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(768),
            torch.nn.ConvTranspose2d(768, 192, (3, 3), (2, 2), (1, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(192),
            torch.nn.ConvTranspose2d(192, 48, (3, 3), (2, 2), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.ConvTranspose2d(48, 12, (3, 3), (2, 2), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.ConvTranspose2d(12, 12, (3, 3), (2, 2), (1, 1), (0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.ConvTranspose2d(12, 3, (3, 3), (2, 2), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.layers(img)


class TestFullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = TestDownSampler()
        self.up = TestUpSampler()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        y = self.down(img)
        y = self.up(y)
        return y


class TestFullModelExpand(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = TestDownSamplerExpand()
        self.up = TestUpSamplerExpand()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        y = self.down(img)
        y = self.up(y)
        return y

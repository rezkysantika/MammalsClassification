import torch
import torch.nn as nn

class ConvBlock2D(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, kernel_size = 3, padding = 1, stride = 1) -> None:
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(output_channels, momentum = 0.9),
            nn.LeakyReLU(inplace = False)
        )
    
    def forward(self, x):
        return self.block_1(x)


class ConvPoolBlock(nn.Module):
    def __init__(self, input_channels : int, output_channels : int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock2D(input_channels,   output_channels, kernel_size = 3, padding = "same"),
            ConvBlock2D(output_channels,  output_channels, kernel_size = 1, padding = "same"),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
    
    def forward(self, x) :
        return self.block(x)

class ExtendedSimpleCNN2D(nn.Module):
    def __init__(self, input_channels : int, output_classes : int) -> None:
        super().__init__()

        self.feet = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size = 5, stride = 3, padding = 1)
        )

        self.body = nn.Sequential(
            ConvPoolBlock(16,   32),

            ConvPoolBlock(32,   64),

            ConvPoolBlock(64,  128),

            ConvPoolBlock(128, 256),
        )

        self.neck = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.feet(x)

        x = self.body(x)

        x = self.neck(x)
        x = torch.squeeze(x, dim = (2, 3))

        x = self.head(x)

        return x

if __name__ == "__main__":
    print("Net-9")

    cux = torch.device('cpu')

    mod = ExtendedSimpleCNN2D(3, 2).to(cux)
    import datetime

    with torch.no_grad():
        t = torch.rand(1, 3, 128, 128).to(cux)
        y = mod(t)

    import torchinfo
    torchinfo.summary(mod, input_data = t)

    start_t = datetime.datetime.now()
    for _ in range(10):
        with torch.no_grad():
            t = torch.rand(1, 3, 128, 128).to(cux)
            y = mod(t)
    stop_t   = datetime.datetime.now()
    exc_time = (stop_t- start_t).total_seconds()
    print("Total Time :", exc_time / 10)
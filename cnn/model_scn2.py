import torch
import torch.nn as nn
import torch.onnx
import datetime
import torchinfo

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
    def __init__(self, in_channels, out_channels):
        super(ConvPoolBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Add batch normalization here
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # And here
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class ExtendedSimpleCNN2D(nn.Module):
    def __init__(self, input_channels : int, output_classes : int) -> None:
        super().__init__()

        self.feet = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size = 5, stride = 2, padding = 1)
        )

        self.body = nn.Sequential(
            ConvPoolBlock(16,   32),

            ConvPoolBlock(32,   64),

            ConvPoolBlock(64,  128),

            ConvPoolBlock(128, 256),

            # ConvPoolBlock(256, 256)
        )

        self.neck = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.6), #add drop out
            nn.Linear(64, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.feet(x)

        x = self.body(x)

        x = self.neck(x)
        # x = torch.squeeze(x, dim = (2, 3))
        x = torch.flatten(x, 1)

        x = self.head(x)

        return x

if __name__ == "__main__":
    print("Net-9")

    cux = torch.device('cpu')

    # Model definition
    mod = ExtendedSimpleCNN2D(3, 2).to(cux)

    with torch.no_grad():
        t = torch.rand(1, 3, 128, 128).to(cux)
        y = mod(t)

    torchinfo.summary(mod, input_data = t)

    start_t = datetime.datetime.now()
    for _ in range(10):
        with torch.no_grad():
            t = torch.rand(1, 3, 128, 128).to(cux)
            y = mod(t)
    stop_t   = datetime.datetime.now()
    exc_time = (stop_t- start_t).total_seconds()
    print("Total Time :", exc_time / 10)

    mod.eval()  # Set model to evaluation mode

    # Dummy input for size [batch_size, channels, height, width] -> [1, 3, 177, 177]
    dummy_input = torch.rand(1, 3, 177, 177).to(cux)

    # Perform inference to verify the model
    with torch.no_grad():
        output = mod(dummy_input)

    print("Model output:", output)

    # Export model to ONNX
    torch.onnx.export(
        mod,                         # Model being exported
        dummy_input,                 # Input tensor
        "model_scn2.onnx",           # Output file name
        export_params=True,          # Store the trained parameter weights
        opset_version=11,            # ONNX opset version
        do_constant_folding=True,    # Perform constant folding optimization
        input_names=['input'],       # Input tensor names
        output_names=['output'],     # Output tensor names
        dynamic_axes={               # Dynamic axis for batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("Model successfully exported to ONNX as 'model_scn2.onnx'!")
    
# if __name__ == "__main__":
#     print("Net-9")

#     cux = torch.device('cpu')

#     mod = ExtendedSimpleCNN2D(3, 2).to(cux)
#     import datetime

#     with torch.no_grad():
#         t = torch.rand(1, 3, 128, 128).to(cux)
#         y = mod(t)

#     import torchinfo
#     torchinfo.summary(mod, input_data = t)

#     start_t = datetime.datetime.now()
#     for _ in range(10):
#         with torch.no_grad():
#             t = torch.rand(1, 3, 128, 128).to(cux)
#             y = mod(t)
#     stop_t   = datetime.datetime.now()
#     exc_time = (stop_t- start_t).total_seconds()
#     print("Total Time :", exc_time / 10)
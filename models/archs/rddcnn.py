import paddle
import paddle.nn as nn
from models.archs.deform_conv_v2 import DeformConv2d  # Ensure this is correctly implemented

class DnCNN(nn.Layer):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        layers = []

        # Initial deformable convolution layer
        layers.append(DeformConv2d(inc=image_channels, outc=n_channels, kernel_size=kernel_size, padding=1, bias=False, modulation=True))
        layers.append(nn.ReLU())

        # Middle layers
        for i in range(depth - 2):
            if i == 11:
                layers.append(nn.Conv2D(n_channels, n_channels, kernel_size, padding=2, bias_attr=False, dilation=2))
            else:
                layers.append(nn.Conv2D(n_channels, n_channels, kernel_size, padding=1, bias_attr=False))
            layers.append(nn.BatchNorm2D(n_channels, epsilon=0.0001, momentum=0.95))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Conv2D(n_channels, image_channels, kernel_size, padding=1, bias_attr=False))
        self.dncnn = nn.Sequential(*layers)
        self.initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.Orthogonal()(m.weight)
                if m.bias is not None:
                    nn.initializer.Constant(value=0.0)(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                nn.initializer.Constant(value=1.0)(m.weight)
                nn.initializer.Constant(value=0.0)(m.bias)
        print('Init weights finished')


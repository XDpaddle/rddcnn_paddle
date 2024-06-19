import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Conv_BN_Relu_first(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, bias_attr):
        super(Conv_BN_Relu_first, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=1, bias_attr_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class Conv_BN_Relu_other(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, bias_attr):
        super(Conv_BN_Relu_other, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=1, bias_attr_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class Conv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, bias_attr):
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=1, bias_attr_attr=False)

    def forward(self, x):
        return self.conv(x)

class Self_Attn(nn.Layer):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2D(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2D(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2D(in_dim, in_dim, 1)
        self.gamma = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0))
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        proj_query = self.query_conv(x).reshape([batch_size, -1, width * height]).transpose([0, 2, 1])
        proj_key = self.key_conv(x).reshape([batch_size, -1, width * height])
        energy = paddle.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).reshape([batch_size, -1, width * height])
        out = paddle.bmm(proj_value, attention.transpose([0, 2, 1]))
        out = out.reshape([batch_size, channels, height, width])
        return self.gamma * out + x, attention

class ADNet(nn.Layer):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(nn.Conv2D(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_4 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_5 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_6 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_7 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_8 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_9 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_10 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_11 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_12 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_13 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_14 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_15 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_16 = nn.Conv2D(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False)
        self.conv3 = nn.Conv2D(in_channels=6,out_channels=3,kernel_size=1,stride=1,padding=0,groups=1,bias_attr=True)
        self.ReLU = nn.ReLU()
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        def _initialize_weights(self):
            for m in self.sublayers(include_self=False):
                if isinstance(m, nn.Conv2D):
                    fan = nn.initializer._calculate_fan_in_and_fan_out(m.weight)[0]
                    std = (2.0 / fan) ** 0.5
                    nn.initializer.Normal(0.0, std)(m.weight)
                if isinstance(m, nn.BatchNorm2D):
                    std = (2.0 / (9.0 * 64)) ** 0.5  # Adjusted from the original calculation
                    nn.initializer.Normal(0.0, std)(m.weight)
                    clip_b = 0.025
                    weight = m.weight.numpy()  # Get the numpy array of the weight
                    weight = paddle.clip(weight, min=-clip_b, max=clip_b)  # Apply clipping
                    m.weight.set_value(weight)  # Set the modified weights back
                    m._mean.set_value(paddle.full_like(m._mean, 0.01))  # Initialize running mean
    def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias_attr=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias_attr=bias_attr))
        return nn.Sequential(*layers)
    def forward(self, x):
        input = x 
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)   
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = paddle.concat([x,x1],1)
        out= self.Tanh(out)
        out = self.conv3(out)
        out = out*x1
        out2 = x - out
        return out2

# Note: Adjust the initialization and usage of parameters and buffers appropriately.

import torch.nn as nn



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f_x = self.conv1(x)
        f_x = self.bn1(f_x)
        f_x = self.relu(f_x)

        return f_x



class _Vgg16(nn.Module):
    def __init__(self, block, block_cnts, dilation=1):
        super(_Vgg16, self).__init__()

        self.layer1 = self._layer(block, block_cnts[0], in_channels=3, out_channels=64, stride=1)
        self.layer2 = self._layer(block, block_cnts[1], in_channels=64, out_channels=128, stride=1)
        self.layer3 = self._layer(block, block_cnts[2], in_channels=128, out_channels=256, stride=1)
        self.layer4 = self._layer(block, block_cnts[3], in_channels=256, out_channels=512, stride=1)
        self.layer5 = self._layer(block, block_cnts[4], in_channels=512, out_channels=512, stride=1)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classify1 = self._classify_layer(in_features=512, out_features=4096)
        self.classify2 = self._classify_layer(in_features=4096, out_features=4096)

        self.dense = nn.Linear(in_features=4096, out_features=10, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.flatern = nn.Flatten(start_dim=1)

    def _classify_layer(self, in_features, out_features):
        blocks = [
            nn.Linear(in_features=in_features, out_features=out_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        ]

        return nn.Sequential(*blocks)

    def _layer(self, block, block_cnt, in_channels, out_channels, stride):
        blocks = []
        blocks.append(
            block(in_channels, out_channels)
        )

        for cnt in range(1, block_cnt):
            b = block(out_channels, out_channels)
            blocks.append(b)

        blocks.append(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avg(x)

        x = self.flatern(x)
        x = self.classify1(x)
        x = self.classify2(x)

        # x = self.flatern(x)
        x = self.dense(x)
        # x = self.softmax(x)
        return x
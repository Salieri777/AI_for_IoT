
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

#################################
# DEEP CONVOLUTIONAL NEURAL NET #
#################################

# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ESC50Model(nn.Module):
    """
    自定义 CNN，用于 ESC-50 数据集的音频分类
    参数保持与原版兼容：
      - input_shape: 传入但内部不再使用，保留兼容性
      - batch_size: 同上，仅占位
      - num_cat: 分类数量，默认为 50
    结构：
      1. 4 个 Conv–BN–ReLU–Pool 模块，每模块后接 Dropout
      2. 全局平均池化将特征图降为通道向量
      3. 两层全连接，最后输出 num_cat 个 logits
    """

    def __init__(self, num_cat: int = 50):
        super().__init__()
        # （兼容旧调用）：input_shape, batch_size 参数不再用于计算，仅保留签名

        # —— 模块 1：32 通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        # —— 模块 2：64 通道
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        # —— 模块 3：128 通道
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(128)
        # —— 模块 4：256 通道
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7   = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8   = nn.BatchNorm2d(256)

        # 公用：最大池化 + 随机失活
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # 全局平均池化，将任意 H×W 特征图汇聚为 1×1×C
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 两层全连接：256→256→num_cat
        self.fc1   = nn.Linear(256, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2   = nn.Linear(256, num_cat)

    def forward(self, x):
        # 模块 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 模块 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 模块 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 模块 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 全局平均池化，得到 [batch, 256, 1, 1]
        x = self.global_avg_pool(x)
        # 拉平为 [batch, 256]
        x = x.view(x.size(0), -1)

        # 全连接 + 激活 + Dropout
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)

        # 最后一层全连接：输出各类别 logits
        x = self.fc2(x)
        return x
    
##############
# RESNET_34  #
##############   

class RES:
    """
    使用预训练的 ResNet34 作为特征提取器，修改输入通道和输出类别数。
    """

    def __init__(self):
        self.model = resnet34(pretrained=True)

    def gen_resnet(self):
        if torch.cuda.is_available(): 
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        model = self.model
        model.fc = nn.Linear(512, 50)  # 修改全连接层输出为 50 类
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = model.to(device)

        return model

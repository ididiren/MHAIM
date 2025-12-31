import torch
import torch.nn as nn
from triplet_attention import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def squash2(x,dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    u = (1-1/torch.exp(squared_norm.sqrt()))*x / (squared_norm+1e-8).sqrt()
    return u

class PrimaryCaps2(nn.Module):

    def __init__(self, out_channels):
        super(PrimaryCaps2, self).__init__()
        self.out_channels = out_channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = squash2(x.contiguous().view(batch_size, -1, self.out_channels), dim=-1)
        return x

class DigitCaps2(nn.Module):

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, D):

        super(DigitCaps2, self).__init__()
        # self.D = D
        self.D = float(D)
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.device = device
        self.attention_coef = 1/torch.sqrt(torch.tensor([self.D])).to(self.device)
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)
        self.B = nn.Parameter(0.01 * torch.randn(num_caps, 1, in_caps),requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1)
        a = self.attention_coef*torch.matmul(u_hat,u_hat.transpose(2,3)).to(self.device)
        c = a.sum(dim=-2,keepdim=True).softmax(dim=1)
        s = torch.matmul((c + self.B), u_hat).squeeze(-2)
        v = squash2(s)
        return v

class CapsuleLoss(nn.Module):

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda

    def forward(self, labels, logits):
        left = (self.upper - logits).relu() ** 2
        right = (logits - self.lower).relu() ** 2
        margin_loss = torch.sum((labels * left + self.lmda * (1 - labels) * right),dim=-1)
        margin_loss = torch.mean(margin_loss)

        return margin_loss

class Cnn_M1(nn.Module):

    def __init__(self, channels):
        super(Cnn_M1,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.Dropout(0.3),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.Dropout(0.3)
        )
        self.conv1_adjust = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=(1, 1))  # 调整通道数
        self.triplet_attention = TripletAttention(64, 16)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):
        residual = x
        out = self.conv_layers(x)

        # out = self.pool(out)
        out = self.triplet_attention(out)
        out = self.conv1_adjust(out)
        out += residual
        out = out.view(out.size(0), -1)  # 展平特征图
        return out

class Cnn_M2(nn.Module):

    def __init__(self, channels):
        super(Cnn_M2,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.Dropout(0.3),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.Dropout(0.3)
        )
        self.conv1_adjust = nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=(1, 1))  # 调整通道数
        self.triplet_attention = TripletAttention(32, 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)


    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        out = self.triplet_attention(out)
        out = self.conv1_adjust(out)
        out += residual
        out = out.view(out.size(0), -1)  # 展平特征图
        return out

class Cnn_M3(nn.Module):

    def __init__(self, channels):
        super(Cnn_M3,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.conv1_adjust = nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=(1, 1))  # 调整通道数
        self.triplet_attention = TripletAttention(32, 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)


    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        out = self.triplet_attention(out)
        out = self.conv1_adjust(out)
        out += residual
        out = out.view(out.size(0), -1)  # 展平特征图
        return out


class Mymodel(nn.Module):

    def __init__(self, dropout=0.2):
        super(Mymodel, self).__init__()
        self.cnn1 = Cnn_M1(channels=8)
        self.cnn2 = Cnn_M2(channels=1)
        self.cnn3 = Cnn_M3(channels=1)
        self.pri = PrimaryCaps2(out_channels=8)
        self.dig = DigitCaps2(in_dim=8,
                             in_caps=16,
                             num_caps=2,
                             dim_caps=2,
                             D = 128)
        self.linear1 = nn.Sequential(
            nn.Linear(80000, 256),  # herg_braga的80为51200,CYP,non_tox等97时为73728, 85时为56448，52时为21632,
            # （64）40时为25600，(32)40时为12800，（16）40时为6400,(64)51时为43264, (64)55和54时为46656,56时为50176,97时为147456
            # (64)45时为30976，(64)50时为40000，(64)64时为65536，
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256,64),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(10000, 256),  # herg_braga的80为51200,CYP,non_tox等97时为73728, 85时为56448，52时为21632,
            # （64）40时为25600，(32)40时为12800，（16）40时为6400,(64)51时为43264, (64)55和54时为46656,56时为50176,97时为147456
            # (64)45时为30976，(64)50时为40000，(64)64时为65536，
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 48),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(10000, 256),  # herg_braga的80为51200,CYP,non_tox等97时为73728, 85时为56448，52时为21632,
            # （64）40时为25600，(32)40时为12800，（16）40时为6400,(64)51时为43264, (64)55和54时为46656,56时为50176,97时为147456
            # (64)45时为30976，(64)50时为40000，(64)64时为65536，
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 16),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = x[:, :8, :, :]  # 前8个通道
        x2 = x[:, 8:9, :, :]  # 后2个通道
        x3 = x[:, 9:10, :, :]  # 后2个通道
        out1 = self.cnn1(x1)
        out2 = self.cnn2(x2)
        out3 = self.cnn3(x3)
        out1 = self.linear1(out1)
        out2 = self.linear2(out2)
        out3 = self.linear3(out3)
        out = torch.cat((out1, out2, out3), dim=-1)
        # print(out.shape)
        # out = self.linear3(out)
        out = self.pri(out)
        out = self.dig(out)
        # print(out.shape)

        logits = (out ** 2).sum(dim=-1)
        logits = (logits + 1e-8).sqrt()
        # print(logits.shape)

        return logits
# model = Mymodel()
# input_data = torch.randn(32, 11, 100, 100)  # batch size = 32, channel = 11, height = width = 56
# output = model(input_data)
#
# # 检查输出维度
# print("Output shape:", output.shape)  # 应为 (32, 2)，表示32个样本的二分类输出

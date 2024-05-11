import torch
import torch.nn as nn
from torch.nn import init

class conv_block_group(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_group, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_ch=1):  
        super(Encoder, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.Conv1 = conv_block_group(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block_group(ch_in=32, ch_out=64)
        self.Conv3 = conv_block_group(ch_in=64, ch_out=128)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.Maxpool(x1)
        x1 = self.Conv2(x1)
        x1 = self.Maxpool(x1)
        x1 = self.Conv3(x1)
        return x1


class Classifier(nn.Module):
    def __init__(self, num_classes=4, fc_num=1):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_fc = nn.Sequential(
            nn.Linear(128*fc_num, 256*fc_num),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256*fc_num, 256*fc_num),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256*fc_num, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x2 = self.avgpool(x2)
        x3 = torch.cat([x,x2],1)
        x3 = torch.flatten(x3, 1)       
        x3 = self.classifier_fc(x3)
        return x3


class UnetNN(nn.Module):
    def __init__(self, img_ch=1, num_classes=4, fc_num=1, encoder=Encoder, classifier=Classifier, init_weights: bool = True):
        super(UnetNN, self).__init__()
        self.img_ch = img_ch
        self.encoder_class = encoder(img_ch=img_ch)
        self.encoder_class2 = encoder(img_ch=img_ch)
        self.classifier = classifier(num_classes=num_classes, fc_num=fc_num)
        if init_weights:
            self._initialize_weights()

    def forward(self, a):
        B,_,L,W = a.shape
        x = a[:, 0].unsqueeze(1)
        x2 = a[:, 1].unsqueeze(1)
        x = self.encoder_class(x)
        x2 = self.encoder_class2(x2)
        d = self.classifier(x,x2)
        return d

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)









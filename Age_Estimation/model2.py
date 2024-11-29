import torch
import torch.nn as nn


class MultipleOutputCNN(nn.Module):
    def __init__(self, min_age=1, max_age=5):
        super(MultipleOutputCNN, self).__init__()
        self.min_age = min_age
        self.max_age = max_age

        # 입력 이미지 (60,60,3)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=20, alpha=0.0001, beta=0.75, k=1.0),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=40, alpha=0.0001, beta=0.75, k=1.0),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=80, alpha=0.0001, beta=0.75, k=1.0)
        )

        self.FC = nn.Linear(in_features=80, out_features=80)
        self.Flatten = nn.Flatten()
        
        # Linear layers for each age category
        self.fc_layers = nn.ModuleList([nn.Linear(80, 2) for _ in range(min_age, max_age + 1)])
        self.softmax = nn.Softmax(dim=2)

        self._init_parameters()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.Flatten(x)
        feature = self.FC(x)

        # 각 연령에 대한 예측
        out = self.softmax(self.fc_layers[0](feature).unsqueeze(1))
        for i in range(1, len(self.fc_layers)):
            temp = self.softmax(self.fc_layers[i](feature).unsqueeze(1))
            out = torch.cat((out, temp), 1)

        return out

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(6, 16, kernel_size=5))
        # layers.append(nn.Dropout(p=0.5))  # add a dropout layer
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(16, 120, kernel_size=5))
        # layers.append(nn.Dropout(p=0.5))  # add a dropout layer
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(84, num_classes))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()

class VGG11(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64,  # [B,64,224,224]
                      kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,64,112,112]
            nn.Conv2d(in_channels=64, out_channels=128,  # [B,128,112,112]
                      kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,128,56,56]
            # [B,256,56,56]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            # [B,256,56,56]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,256,28,28]
            # [B,512,56,56]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            # [B,512,56,56]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,512,14,14]
            # [B,512,28,28]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            # [B,512,56,56]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # [B,512,7,7]
        )

        self.fc_layers=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        fc_out=self.fc_layers(conv_out.view(conv_out.size(0),-1))
        return fc_out



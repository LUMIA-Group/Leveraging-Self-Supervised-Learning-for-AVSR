import torch
import torch.nn as nn
import torchvision.models as models


class MoCoVisualFrontend(nn.Module):
    def __init__(self, dModel, nClasses, frameLen, Mocofile, vidfeaturedim):
        super(MoCoVisualFrontend, self).__init__()
        self.dModel = dModel
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.vidfeaturedim = vidfeaturedim
        # Conv3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # moco0
        MoCoModel = models.__dict__['resnet50']()
        MoCoModel.fc = nn.Identity()
        MoCoModel.conv1 = nn.Identity()
        MoCoModel.bn1 = nn.Identity()
        MoCoModel.relu = nn.Identity()
        MoCoModel.maxpool = nn.Identity()

        checkpoint = torch.load(Mocofile, map_location="cpu")
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = MoCoModel.load_state_dict(state_dict, strict=False)

        self.MoCoModel = MoCoModel

        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(self.vidfeaturedim, self.dModel, (5,), (2,), (0,), bias=False),
            nn.BatchNorm1d(self.dModel),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(self.dModel, self.dModel, (5,), (2,), (0,), bias=False),
            nn.BatchNorm1d(self.dModel),
            nn.ReLU(True),
        )

        self.backend_conv2 = nn.Sequential(
            nn.Linear(self.dModel, self.dModel // 4),
            nn.BatchNorm1d(self.dModel // 4),
            nn.ReLU(True),
            nn.Linear(self.dModel // 4, self.nClasses)
        )

    def forward(self, x):
        x = x.reshape([-1, self.frameLen] + list(x.shape[1:]))
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.reshape([-1] + list(x.shape[2:]))
        x = self.MoCoModel(x).reshape(-1, self.frameLen, self.vidfeaturedim)

        x = x.transpose(1, 2)
        x = self.backend_conv1(x)
        x = torch.mean(x, 2)
        x = self.backend_conv2(x)
        return x

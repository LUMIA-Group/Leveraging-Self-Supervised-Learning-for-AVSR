import torch
import torch.nn as nn
import torchvision.models as models
from config import args


class MoCoVisualFrontend(nn.Module):
    def __init__(self, dModel=args["FRONTEND_DMODEL"], nClasses=args["WORD_NUM_CLASSES"], frameLen=args["FRAME_LENGTH"],
                 vidfeaturedim=args["VIDEO_FEATURE_SIZE"]):
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
        # moco
        MoCoModel = models.__dict__['resnet50']()
        MoCoModel.fc = nn.Identity()
        MoCoModel.conv1 = nn.Identity()
        MoCoModel.bn1 = nn.Identity()
        MoCoModel.relu = nn.Identity()
        MoCoModel.maxpool = nn.Identity()
        self.MoCoModel = MoCoModel

    def forward(self, x, x_len):
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        mask = torch.zeros(x.shape[:2], device=x.device)
        mask[(torch.arange(mask.shape[0], device=x.device), x_len - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        x = x[~mask]
        x = self.MoCoModel(x)
        return x

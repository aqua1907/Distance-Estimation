import torch
import torch.nn as nn
from model.mixnet import MixNet
import torchvision


class DistanceEstimation(nn.Module):
    def __init__(self, criterion=None):
        super(DistanceEstimation, self).__init__()
        self.criterion = criterion
        self.backbone = MixNet(arch="s")
        self.roi_pool = torchvision.ops.RoIPool(output_size=(3, 3), spatial_scale=30)
        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.softplus = nn.Softplus()

    def forward(self, images, bboxes, targets=None):
        features = self.backbone(images)
        roi_pooled = self.roi_pool(features, bboxes)
        roi_pooled = torch.flatten(roi_pooled, 1)

        fc_out = self.fc1(roi_pooled)
        fc_out = self.fc2(fc_out)
        fc_out = self.fc3(fc_out)

        preds = self.softplus(fc_out)

        if self.training:
            loss = self.criterion(preds, targets)
            return loss
        else:
            return preds


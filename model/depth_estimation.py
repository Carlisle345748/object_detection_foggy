import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec


def conv3x3(in_channels, out_channels, stride=1, padding=0):
    """
    Convolutional network with kernel_size = 3, default padding = 0, default stride = 1
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    """
    Convolutional network with kernel_size = 1, padding = 0, default stride = 1
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class DEB(nn.Module):
    """
    Depth Estimation Block: generate depth map with input feature map.
    """
    def __init__(self, input_shape: ShapeSpec):
        super(DEB, self).__init__()
        self.deb = nn.Sequential(
            conv1x1(input_shape.channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            conv3x3(64, 1),
            nn.Sigmoid()
        )
        self.deb.apply(self.init_weights)
        self.loss = nn.MSELoss()

    @classmethod
    @torch.no_grad()
    def init_weights(cls, model):
        if type(model) == nn.Conv2d:
            nn.init.kaiming_normal_(model.weight.data)

    def forward(self, x, gt_depth_map):
        depth_map = self.deb(x)
        _, _, h1, w1 = depth_map.size()
        _, _, h, w = gt_depth_map.size()
        if h1 != h or w1 != w:
            gt_depth_map = torch.nn.Upsample(size=(h1, w1), mode='bilinear')(gt_depth_map)
        return {"depth_loss": self.loss(depth_map, gt_depth_map)}


if __name__ == '__main__':
    test_input = torch.rand((4, 2048, 32, 32))
    input_shape = ShapeSpec(channels=2048)
    test_gt = torch.rand((4, 1, 32, 32))

    test_deb = DEB(input_shape)
    loss = test_deb(test_input, test_gt)
    print(loss["depth_loss"])

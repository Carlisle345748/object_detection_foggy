import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec


class DEB(nn.Module):
    """
    Depth Estimation Block: generate depth map with input feature map.
    """
    def __init__(self, input_shape: ShapeSpec):
        super(DEB, self).__init__()
        self.deb = nn.Sequential(
            nn.Conv2d(input_shape.channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.deb.apply(self.init_weights)
        self.loss = nn.MSELoss()

    @classmethod
    @torch.no_grad()
    def init_weights(cls, model):
        if type(model) == nn.Conv2d:
            nn.init.kaiming_normal_(model.weight.data)

    def forward(self, x, gt_depth_map=None):
        depth_map = self.deb(x)

        _, _, h1, w1 = depth_map.size()
        _, _, h, w = gt_depth_map.size()
        if h1 != h or w1 != w:
            gt_depth_map = torch.nn.Upsample(size=(h1, w1), mode='bilinear')(gt_depth_map)

        if gt_depth_map is None:
            return None, {"depth": depth_map}

        depth_loss = self.loss(depth_map, gt_depth_map)
        return {"depth_loss": depth_loss}, {"depth": depth_map}


if __name__ == '__main__':
    test_input = torch.rand((4, 2048, 32, 32))
    input_shape = ShapeSpec(channels=2048)
    test_gt = torch.rand((4, 1, 32, 32))

    test_deb = DEB(input_shape)
    loss, d_map = test_deb(test_input, test_gt)
    print(d_map["depth"].shape, test_gt.shape)

import torch
import torch.nn as nn
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, ResNet, build_backbone
from typing import List, Dict, Tuple

from detectron2.structures import ImageList

from model.depth_estimation import DEB


@META_ARCH_REGISTRY.register()
class ResnetDEB(nn.Module):
    @configurable
    def __init__(
        self, 
        backbone: ResNet, 
        backbone_out_feature: str, 
        depth_estimation: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_out_feature = backbone_out_feature
        self.depth_estimation = depth_estimation

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_out_feature = cfg.MODEL.RESNETS.OUT_FEATURES[0]
        feature_shape = backbone.output_shape()[backbone_out_feature]
        depth_estimation = DEB(feature_shape)

        return {
            "backbone": backbone,
            "backbone_out_feature": backbone_out_feature,
            "depth_estimation": depth_estimation,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_inputs(batched_inputs, "image")
        features = self.backbone(images.tensor)
        gt_depth = self.preprocess_inputs(batched_inputs, "depth")
        losses, _ = self.depth_estimation(features[self.backbone_out_feature], gt_depth)
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]],
    ):
        images = self.preprocess_inputs(batched_inputs, "image")
        features = self.backbone(images.tensor)
        _, depth_map = self.depth_estimation(features[self.backbone_out_feature])
        return depth_map

    def preprocess_inputs(self, batched_inputs: List[Dict[str, torch.Tensor]], attr_name: str):
        """
        Normalize, pad and batch the input images.
        attr_name: "image" for preprocess images, "depth" for preprocessing depth maps.
        """
        images = [x[attr_name].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

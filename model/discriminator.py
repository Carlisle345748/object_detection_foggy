from typing import Any

import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec


class Discriminator(nn.Module):
    """
    Domain discriminator for adverse learning. The default loss function is binary cross entropy loss.
    The discriminator has built-in gradient reverse layer
    """

    def __init__(self, input_shape: ShapeSpec, loss: str = "bce", alpha: float = 1.0):
        super().__init__()
        self.grl = GradReverseLayer(alpha=alpha)

        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape.channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=256),
            nn.LeakyReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=128),
            nn.LeakyReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=128),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Flatten()
        )

        if loss == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"{loss} loss hasn't been implemented")

        self.block1.apply(self.init_weights)
        self.block2.apply(self.init_weights)
        self.block3.apply(self.init_weights)

    @classmethod
    @torch.no_grad()
    def init_weights(cls, model):
        if type(model) == nn.Conv2d:
            nn.init.kaiming_normal_(model.weight.data)

    def forward(self, x: torch.Tensor, y: int):
        x = self.grl(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)

        labels = (torch.ones_like(x) * y).to(x.device)
        return {"discriminator_loss": self.loss(x, labels)}


class GradReverseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> torch.Tensor:
        return grad_outputs.neg() * ctx.alpha, None


class GradReverseLayer(nn.Module):
    """
    Gradient reverse layer. The alpha is a scale factor after negating the gradient
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverseFunc.apply(x, self.alpha)

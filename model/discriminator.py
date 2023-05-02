from typing import Any

import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec


class Discriminator(nn.Module):
    """
    Domain discriminator for adverse learning. The default loss function is binary cross entropy loss
    """
    def __init__(self, input_shape: ShapeSpec, loss: str = "bce"):
        super().__init__()

        self.conv1 = nn.Conv2d(input_shape.channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        if loss == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"{loss} loss hasn't been implemented")

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        scores = self.classifier(x)
        loss = self.loss(x, y)
        return scores, loss


class GradReverseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> torch.Tensor:
        return grad_outputs.neg() * ctx.alpha


class GradReverseLayer(nn.Module):
    """
    Gradient reverse layer. The alpha is a scale factor after negating the gradient
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverseFunc.apply(x, self.alpha)

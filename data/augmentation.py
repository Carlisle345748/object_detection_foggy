import os

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from detectron2.config import get_cfg

from trainer.config import add_teacher_student_config


def build_strong_augmentation():
    return transforms.Compose(
        [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 7), sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),
        ]
    )


def test_strong_augmentation():
    """
    Test the DatasetMapper on depth maps

    Usage:
        python -m data.mapper
    """
    cwd = os.getcwd().removesuffix("/data")

    cfg = get_cfg()
    add_teacher_student_config(cfg)
    cfg.merge_from_file(os.path.join(cwd, "config", "RCNN-C4-50.yaml"))

    img = Image.open(os.path.join(cwd, "datasets", "cityscapes", "leftImg8bit",
                                  "train", "bremen", "bremen_000000_000019_leftImg8bit.png"))
    aug = build_strong_augmentation()
    aug_img = aug(F.to_tensor(img))

    img.show()
    F.to_pil_image(aug_img).show()


if __name__ == "__main__":
    test_strong_augmentation()

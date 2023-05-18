import copy
import logging
import os

import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg, configurable
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.utils.file_io import PathManager

from data.augmentation import build_strong_augmentation
from trainer.config import add_teacher_student_config

logger = logging.getLogger(__name__)
logger.parent = logging.getLogger('detectron2')


class DepthDatasetMapper(DatasetMapper):
    """
    Extend the Default DatasetMapper to load the depth map and apply augmentation to it
    """

    def _transform_image_and_depth(self, dataset_dict):
        # Load image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Image augmentation
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        # Load depth map
        if "depth_file" not in dataset_dict:
            return image, None, transforms

        depth = self.read_depth_map(dataset_dict["depth_file"])
        utils.check_image_size(dataset_dict, depth)

        # Normalize depth
        depth_mean, depth_std = np.mean(depth), np.std(depth)
        depth = (depth - depth_mean) / depth_std
        dataset_dict["depth_mean"] = depth_mean
        dataset_dict["depth_std"] = depth_std

        # Apply augmentation to depth map
        depth = transforms.apply_image(depth)

        return image, depth, transforms

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image, depth, _ = self._transform_image_and_depth(dataset_dict)

        # Add depth map into data
        if depth is not None:
            dataset_dict["depth"] = torch.unsqueeze(torch.as_tensor(np.ascontiguousarray(depth)), 0)
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("depth", None)
            return dataset_dict

        return dataset_dict

    @classmethod
    def read_depth_map(cls, file_name):
        with PathManager.open(file_name, "rb") as f:
            image = Image.open(f)
            image = image.convert("L")

            # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
            image = _apply_exif_orientation(image)
            return convert_PIL_to_numpy(image, "L")


class TeacherStudentDepthDatasetMapper(DepthDatasetMapper):
    """
    Extend the DepthDatasetMapper to transform object detection annotations
    """

    @configurable
    def __init__(self, strong_augmentation: bool, **kwargs):
        super().__init__(**kwargs)
        self.strong_augmentation = build_strong_augmentation() if strong_augmentation else None
        if strong_augmentation:
            logger.info("Strong augmentations used in training: " + str(self.strong_augmentation))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train=is_train)
        ret["strong_augmentation"] = cfg.INPUT.STRONG_AUG
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image, depth, transforms = self._transform_image_and_depth(dataset_dict)

        # Add depth map into datadict
        if depth is not None:
            dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth))

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # Use strong augmentation
        if self.strong_augmentation and self.is_train:
            dataset_dict["image_strong_aug"] = self.strong_augmentation(dataset_dict["image"])

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("depth", None)
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class BaselineDatasetMapper(DatasetMapper):
    @configurable
    def __init__(self, strong_augmentation: bool, **kwargs):
        super().__init__(**kwargs)
        self.strong_augmentation = build_strong_augmentation() if strong_augmentation else None
        if strong_augmentation:
            logger.info("Strong augmentations used in training: " + str(self.strong_augmentation))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train=is_train)
        ret["strong_augmentation"] = cfg.INPUT.STRONG_AUG
        return ret

    def __call__(self, dataset_dict):
        super().__call__(dataset_dict)
        if self.strong_augmentation and self.is_train:
            dataset_dict["image"] = self.strong_augmentation(dataset_dict["image"])


def test_augmentation():
    """
    Test the DatasetMapper on depth maps

    Usage:
        python -m data.mapper
    """
    cwd = os.getcwd().removesuffix("/data")

    cfg = get_cfg()
    add_teacher_student_config(cfg)
    cfg.merge_from_file(os.path.join(cwd, "config", "RCNN-C4-50.yaml"))

    img = Image.open(os.path.join(cwd, "datasets", "cityscapes", "disparity",
                                  "train", "bremen", "bremen_000084_000019_disparity.png"))
    image = np.asarray(img).astype(np.float32)
    augs = utils.build_augmentation(cfg, True)
    augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augmentations = T.AugmentationList(augs)

    aug_input = T.AugInput(image)
    augmentations(aug_input)
    aug_img = Image.fromarray(aug_input.image)

    img.show()
    aug_img.show()


def test_normalize():
    cwd = os.getcwd().removesuffix("/data")
    img = Image.open(os.path.join(cwd, "datasets", "cityscapes", "disparity",
                                  "train", "bremen", "bremen_000084_000019_disparity.png"))
    img.show()
    image = img.convert("L")
    image = np.asarray(image)
    grayscale = Image.fromarray(image)
    grayscale.show()


if __name__ == "__main__":
    test_normalize()

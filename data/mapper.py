import copy
import os

import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from trainer.config import add_teacher_student_config


class DetectionWithDepthDatasetMapper(DatasetMapper):
    """
    Extend the Default DatasetMapper to load the depth map and apply augmentation to it
    """

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Read depth map
        depth = utils.read_image(dataset_dict["depth_file"]).astype(np.float32)
        utils.check_image_size(dataset_dict, depth)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # Apply augmentation to depth map
        depth = transforms.apply_image(depth)

        # Add depth map into data
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth))

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


if __name__ == "__main__":
    """
    Test the DatasetMapper on depth maps

    Usage:
        python -m data.mapper 
    """
    cwd = os.getcwd().removesuffix("/data")
    dataset_dir = os.path.join(os.getcwd(), "datasets")

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
    transforms = augmentations(aug_input)
    aug_img = Image.fromarray(aug_input.image)

    img.show()
    aug_img.show()

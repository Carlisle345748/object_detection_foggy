import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from load_cityscapes_foggy import load_cityscapes_instances

DATASETS = {
    "train": ("cityscapes_foggy/leftImg8bit/train", "cityscapes_foggy/gtFine/train"),
    "val": ("cityscapes_foggy/leftImg8bit/val", "cityscapes_foggy/gtFine/val")
}


def register_dataset():
    dataset_dir = os.path.join(os.getcwd(), "datasets")
    for key, (image_dir, gt_dir) in DATASETS.items():
        dataset_name = "cityscapes_foggy_" + key
        image_dir = os.path.join(dataset_dir, image_dir)
        gt_dir = os.path.join(dataset_dir, gt_dir)
        meta = _get_builtin_metadata("cityscapes")
        DatasetCatalog.register(dataset_name,
                                lambda img=image_dir, gt=gt_dir: load_cityscapes_instances(img, gt, False, False))
        MetadataCatalog.get(dataset_name).set(**meta, image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco")



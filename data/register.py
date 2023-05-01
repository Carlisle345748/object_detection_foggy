import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .load_cityscapes import load_cityscapes_instances, load_cityscapes_depth

DOMAIN_ADAPTATION_DATASETS = {
    "cityscapes_train": ("cityscapes/leftImg8bit/train",
                         "cityscapes/gtFine/train",
                         "cityscapes/disparity/train"),
    "cityscapes_val": ("cityscapes/leftImg8bit/val",
                       "cityscapes/gtFine/val",
                       "cityscapes/disparity/val"),
    "cityscapes_foggy_train": ("cityscapes_foggy/leftImg8bit/train", "cityscapes_foggy/gtFine/train", None),
    "cityscapes_foggy_val": ("cityscapes_foggy/leftImg8bit/val", "cityscapes_foggy/gtFine/val", None)
}

DEPTH_DATASETS = {
    "cityscapes_depth_train": ("cityscapes/leftImg8bit/train", "cityscapes/disparity/train"),
    "cityscapes_depth_val": ("cityscapes/leftImg8bit/val", "cityscapes/disparity/val"),
}


def register_domain_adaptation_dataset():
    """
    Register cityscapes and cityscapes_foggy dataset that built for semi-supervised learning
    Add dataset "cityscapes_train", "cityscapes_val", "cityscapes_foggy_train", and "cityscapes_foggy_val"
    """
    dataset_dir = os.path.join(os.getcwd(), "datasets")
    for dataset_name, dirs in DOMAIN_ADAPTATION_DATASETS.items():
        data_dir = []
        for d in dirs:
            data_dir.append(os.path.join(dataset_dir, d) if d else None)

        foggy = dataset_name.startswith("cityscapes_foggy")
        image_dir, gt_dir, depth_dir = data_dir[0], data_dir[1], data_dir[2]

        # Register Dataset
        DatasetCatalog.register(dataset_name,
                                lambda img=image_dir, gt=gt_dir, depth=depth_dir, fog=foggy:
                                load_cityscapes_instances(img, gt, depth, False, False, fog))

        # Register Dataset Metadata
        MetadataCatalog.get(dataset_name).set(
            **_get_builtin_metadata("cityscapes"),
            image_dir=image_dir,
            gt_dir=gt_dir,
            depth_dir=depth_dir,
            evaluator_type="coco"
        )


def register_depth_dataset():
    dataset_dir = os.path.join(os.getcwd(), "datasets")
    for dataset_name, dirs in DEPTH_DATASETS.items():
        data_dir = []
        for d in dirs:
            data_dir.append(os.path.join(dataset_dir, d) if d else None)

        image_dir, depth_dir = data_dir[0], data_dir[1]

        # Register Dataset
        DatasetCatalog.register(dataset_name, lambda img=image_dir, depth=depth_dir: load_cityscapes_depth(img, depth))

        # Register Dataset Metadata
        MetadataCatalog.get(dataset_name).set(
            image_dir=image_dir,
            depth_dir=depth_dir,
            evaluator_type="coco"  # TODO Add depth model evaluator
        )


if __name__ == "__main__":
    print(DatasetCatalog)
    print("*" * 50)
    print(MetadataCatalog.get("cityscapes_train"))
    print("*" * 50)
    print(MetadataCatalog.get("cityscapes_val"))
    print("*" * 50)
    print(MetadataCatalog.get("cityscapes_foggy_train"))
    print("*" * 50)
    print(MetadataCatalog.get("cityscapes_foggy_val"))

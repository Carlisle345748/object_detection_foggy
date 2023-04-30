import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .load_cityscapes import load_cityscapes_instances

DATASETS = {
    "cityscapes_train": ("cityscapes/leftImg8bit/train",
                         "cityscapes/gtFine/train",
                         "cityscapes/disparity/train"),
    "cityscapes_val": ("cityscapes/leftImg8bit/val",
                       "cityscapes/gtFine/val",
                       "cityscapes/disparity/val"),
    "cityscapes_foggy_train": ("cityscapes_foggy/leftImg8bit/train", "cityscapes_foggy/gtFine/train", None),
    "cityscapes_foggy_val": ("cityscapes_foggy/leftImg8bit/val", "cityscapes_foggy/gtFine/val", None)
}


def register_dataset():
    dataset_dir = os.path.join(os.getcwd(), "datasets")
    for dataset_name, dirs in DATASETS.items():
        data_dir = []
        for d in dirs:
            data_dir.append(os.path.join(dataset_dir, d) if d else None)

        image_dir, gt_dir, depth_dir = data_dir[0], data_dir[1], data_dir[2]
        foggy = dataset_name.startswith("cityscapes_foggy")

        DatasetCatalog.register(dataset_name,
                                lambda img=image_dir, gt=gt_dir, depth=depth_dir:
                                load_cityscapes_instances(img, gt, depth, False, False, foggy))

        MetadataCatalog.get(dataset_name).set(
            **_get_builtin_metadata("cityscapes"),
            image_dir=image_dir,
            gt_dir=gt_dir,
            depth_dir=depth_dir,
            evaluator_type="coco"
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

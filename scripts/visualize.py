import os

import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer
from model.teacher_student import TeacherStudentRCNN  # Import for registering model
from model.resnet_deb import ResnetDEB  # Import for registering model

from trainer.config import add_teacher_student_config


class CityscapesVisualizer:
    def __init__(self, model_path: str):
        model_name = os.path.basename(model_path)
        config_file = os.path.join(model_path, "config.yaml")
        model_weight = os.path.join(model_path, "model_final.pth")
        output = os.path.join("/Users/carlisle/Developer/cs231n/output/final/visualization/inference", model_name)

        cfg = get_cfg()
        add_teacher_student_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(cfg)

        os.makedirs(output, exist_ok=True)
        self.output = output

    def visualize(self, dataset: str):
        PathManager.isdir(dataset)
        for image in PathManager.ls(dataset):
            image = os.path.join(dataset, image)
            if PathManager.isfile(image) and not image.endswith(".DS_Store") and\
                    not PathManager.exists(self.get_prediction_name(image)):
                self.visualize_and_save(image)
                print(f"visualize {image}")

    def visualize_and_save(self, image: str):
        try:
            img = np.asarray(Image.open(image))
            outputs = self.predictor(img)
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("cityscapes_fine_instance_seg_val"))
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            img = Image.fromarray(out.get_image()[:, :, ::-1])
            img.save(self.get_prediction_name(image))
        except Exception as e:
            print(e)

    def get_prediction_name(self, image: str):
        basename = os.path.basename(image).removesuffix(".png")
        return os.path.join(self.output, f"{basename}_pred.png")


if __name__ == "__main__":
    model_path = "/Users/carlisle/Developer/cs231n/output/final/models/bce/cityscapes_ts_bce_no_deb"
    vis = CityscapesVisualizer(model_path)
    vis.visualize("/Users/carlisle/Developer/cs231n/output/final/visualization/images/")

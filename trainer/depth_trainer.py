import os
from abc import ABC

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from data.dataset_mapper import DepthDatasetMapper

import model.resnet_deb # Import for side effect


class DepthTrainer(DefaultTrainer, ABC):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return [COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))]

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DepthDatasetMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

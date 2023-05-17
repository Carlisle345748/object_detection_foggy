import os
from abc import ABC

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from data.dataset_mapper import BaselineDatasetMapper


class BaselineTrainer(DefaultTrainer, ABC):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return [COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))]

    def build_train_loader(cls, cfg):
        mapper = BaselineDatasetMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

from abc import ABC

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer

from data.dataset_mapper import DepthDatasetMapper
from trainer.depth_evaluator import DepthEvaluator


class DepthTrainer(DefaultTrainer, ABC):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return [DepthEvaluator()]

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DepthDatasetMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

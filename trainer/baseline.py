import os
from abc import ABC

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class BaselineTrainer(DefaultTrainer, ABC):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return [COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))]

import os
from abc import ABC

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from data.domain_adaptation_dataloader import build_domain_adaptation_train_loader


class TeacherStudentTrainer(DefaultTrainer, ABC):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return [COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))]

    @classmethod
    def build_train_loader(cls, cfg):
        return build_domain_adaptation_train_loader(cfg)


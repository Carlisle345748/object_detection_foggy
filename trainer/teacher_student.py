import logging
import os
import weakref
from abc import ABC

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, TrainerBase, create_ddp_model, SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from model.EnsembleModel import EnsembleModel


class TeacherStudentTrainer(DefaultTrainer, ABC):

    def __init__(self, cfg):
        TrainerBase.__init__(self)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        student_model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, student_model)
        data_loader = self.build_train_loader(cfg)

        student_model = create_ddp_model(student_model, broadcast_buffers=False)

        # ***********************************************************************************

        # Create teacher model
        teacher_model = self.build_model(cfg)
        self.teacher_model = teacher_model

        # Ensemble model allow both teacher and student model to be saved and loaded together
        ensemble_model = EnsembleModel(teacher_model, student_model)

        # ***********************************************************************************

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            student_model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            ensemble_model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return [COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))]


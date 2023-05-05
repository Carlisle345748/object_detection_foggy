import logging
import os
import weakref
from abc import ABC

from detectron2.engine import DefaultTrainer, create_ddp_model, AMPTrainer, SimpleTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from data.domain_adaptation_dataloader import build_domain_adaptation_train_loader
from trainer.checkpointer import TeacherStudentCheckpointer
from trainer.grad_monitor import GradMonitor


class TeacherStudentTrainer(DefaultTrainer, ABC):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self._trainer.register_hooks([GradMonitor(model)])

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = TeacherStudentCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            cfg=cfg,
            model=model,
            save_dir=cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return [COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))]

    @classmethod
    def build_train_loader(cls, cfg):
        return build_domain_adaptation_train_loader(cfg)

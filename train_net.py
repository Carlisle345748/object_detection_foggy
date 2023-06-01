import os

import detectron2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch, default_setup

import data  # Import for side-effect

from trainer.baseline import BaselineTrainer
from trainer.checkpointer import TeacherStudentCheckpointer
from trainer.config import add_teacher_student_config
from trainer.depth_trainer import DepthTrainer
from trainer.teacher_student import TeacherStudentTrainer


def show_version():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)


def setup_config(args):
    cfg = get_cfg()
    add_teacher_student_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = TeacherStudentTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


def train_depth(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DepthTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


def test_dataloader(cfg):
    ts_dataloader = TeacherStudentTrainer.build_train_loader(cfg)
    ts_dataloader_iter = iter(ts_dataloader)
    print(next(ts_dataloader_iter))

    depth_dataloader = DepthTrainer.build_train_loader(cfg)
    depth_dataloader_it = iter(depth_dataloader)
    print(next(depth_dataloader_it))


def evaluation(cfg):
    model = TeacherStudentTrainer.build_model(cfg)
    TeacherStudentCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS), resume=False
    )
    res = TeacherStudentTrainer.test(cfg, model)
    return res


def main(args):
    cfg = setup_config(args)
    if args.eval_only:
        evaluation(cfg)
    elif args.test_dataloader:
        test_dataloader(cfg)
    elif args.depth:
        train_depth(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--test_dataloader", action="store_true")
    parser.add_argument("--depth", action="store_true")
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

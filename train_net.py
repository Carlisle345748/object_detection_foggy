import os

import detectron2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch, default_setup

import data  # Import for side-effect

from trainer.baseline import BaselineTrainer


def show_version():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)


def setup_config(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = BaselineTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


def evaluation(cfg):
    model = BaselineTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS), resume=False
    )
    res = BaselineTrainer.test(cfg, model)
    return res


def main(args):
    cfg = setup_config(args)
    if args.eval_only:
        evaluation(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


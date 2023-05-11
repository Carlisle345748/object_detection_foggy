
from detectron2.config import CfgNode as CN


def add_teacher_student_config(cfg):

    _C = cfg

    _C.SOLVER.IMS_PER_BATCH_SOURCE = 1
    _C.SOLVER.IMS_PER_BATCH_TARGET = 1

    _C.DATASETS.TRAIN_SOURCE = ("coco_2017_train",)
    _C.DATASETS.TRAIN_TARGET = ("coco_2017_train",)

    _C.MODEL.TEACHER_STUDENT = CN()
    _C.MODEL.TEACHER_STUDENT.BASE_ARCH = "GeneralizedRCNN"
    _C.MODEL.TEACHER_STUDENT.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    _C.MODEL.TEACHER_STUDENT.DIS_WEIGHT = 0.1
    _C.MODEL.TEACHER_STUDENT.SOURCE_WEIGHT = 1
    _C.MODEL.TEACHER_STUDENT.TARGET_WEIGHT = 1

    _C.MODEL.TEACHER_STUDENT.DEB = False

    _C.MODEL.TEACHER_STUDENT.FOCAL = CN()
    _C.MODEL.TEACHER_STUDENT.FOCAL.ENABLE = False
    _C.MODEL.TEACHER_STUDENT.FOCAL.ALPHA = 0.25
    _C.MODEL.TEACHER_STUDENT.FOCAL.GAMMA = 2.0


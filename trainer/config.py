

def add_teacher_student_config(cfg):

    _C = cfg

    _C.SOLVER.IMS_PER_BATCH_SOURCE = 1
    _C.SOLVER.IMS_PER_BATCH_TARGET = 1

    _C.DATASETS.TRAIN_SOURCE = ("coco_2017_train",)
    _C.DATASETS.TRAIN_TARGET = ("coco_2017_train",)

    _C.MODEL.TEACHER_STUDENT.BASE_ARCH = "GeneralizedRCNN"

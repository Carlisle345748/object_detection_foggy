
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


class EnsembleModel(nn.Module):
    def __init__(self, teacher: nn.Module, student: nn.Module):
        super().__init__()

        if isinstance(teacher, (DistributedDataParallel, DataParallel)):
            teacher = teacher.module

        if isinstance(student, (DistributedDataParallel, DataParallel)):
            student = student.module

        self.teacher = teacher
        self.student = student

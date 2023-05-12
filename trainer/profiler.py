import torch
from detectron2.config import get_cfg
from torch.profiler import profile, record_function, ProfilerActivity

from trainer.config import add_teacher_student_config
from trainer.teacher_student import TeacherStudentTrainer
from detectron2.utils.events import EventStorage

# Define your model
cfg = get_cfg()
add_teacher_student_config(cfg)
cfg.merge_from_file("config/RCNN-C4-50-TS-Test.yaml")

model = TeacherStudentTrainer.build_model(cfg)

# Define your input
dataloder = TeacherStudentTrainer.build_train_loader(cfg)
input = next(iter(dataloder))

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# Run the profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_training"):
        # Run your model
        with EventStorage(0):
            output = model(input)

# Print the profiler results
print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
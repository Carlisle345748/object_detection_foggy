from typing import Dict, Any

from detectron2.checkpoint import DetectionCheckpointer

from model.teacher_student import TeacherStudentRCNN


class TeacherStudentCheckpointer(DetectionCheckpointer):
    def __init__(self, cfg, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model=model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables)

        self.base_model_weight = cfg.MODEL.TEACHER_STUDENT.WEIGHTS

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.init_model(path)

    def init_model(self, meta_model_weight):
        if meta_model_weight:
            return self.load(meta_model_weight, checkpointables=[])

        assert isinstance(self.model, TeacherStudentRCNN), "Model must be TeacherStudentRCNN"
        assert self.base_model_weight, "Base model weight is not provided"

        # Temporary replace the model
        meta_model = self.model

        # Load student model
        self.model = meta_model.student
        self.load(self.base_model_weight, checkpointables=[])

        # Load teacher model
        self.model = meta_model.teacher
        ret = self.load(self.base_model_weight, checkpointables=[])

        # Resume the model
        self.model = meta_model

        return ret

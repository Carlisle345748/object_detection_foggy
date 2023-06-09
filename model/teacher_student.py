import logging
from collections import OrderedDict
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from detectron2.config import configurable
from detectron2.layers import move_device_like
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY, detector_postprocess
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.roi_heads import Res5ROIHeads
from detectron2.structures import Instances, ImageList
from detectron2.utils.events import get_event_storage

import model.roi_head  # Import for side-effect
from model.depth_estimation import DEB
from model.discriminator import Discriminator
from model.resnet_deb import ResnetDEB


@META_ARCH_REGISTRY.register()
class TeacherStudentRCNN(nn.Module):

    @configurable
    def __init__(
            self,
            *,
            teacher: GeneralizedRCNN,
            student: GeneralizedRCNN,
            discriminator: Discriminator,
            depth_estimation: nn.Module,
            backbone_out_feature: str,
            teacher_update_step=1,
            confident_thresh=0.7,
            source_losses_weight=1,
            target_losses_weight=1,
            discriminator_losses_weight=0.1,
            depth_losses_weight=0.1,
            strong_augmentation=False,
            vis_period=0,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.discriminator = discriminator
        self.depth_estimation = depth_estimation
        self.backbone_out_feature = backbone_out_feature
        self.teacher_update_step = teacher_update_step
        self.vis_period = vis_period

        self.SOURCE_LABEL = 0
        self.TARGET_LABEL = 1

        self.iter = 0
        self.confident_thresh = confident_thresh
        self.source_losses_weight = source_losses_weight
        self.target_losses_weight = target_losses_weight
        self.discriminator_losses_weight = discriminator_losses_weight
        self.depth_losses_weight = depth_losses_weight
        self.strong_augmentation = strong_augmentation

    @classmethod
    def from_config(cls, cfg):
        base_arch = cfg.MODEL.TEACHER_STUDENT.BASE_ARCH
        teacher_model = META_ARCH_REGISTRY.get(base_arch)(cfg)
        student_model = META_ARCH_REGISTRY.get(base_arch)(cfg)

        assert len(cfg.MODEL.RESNETS.OUT_FEATURES) == 1, "feature map produced by backbone should has one layer"
        backbone_out_feature = cfg.MODEL.RESNETS.OUT_FEATURES[0]

        feature_shape = student_model.backbone.output_shape()[backbone_out_feature]

        discriminator = Discriminator(
            input_shape=feature_shape,
            loss="focal" if cfg.MODEL.TEACHER_STUDENT.DIS.FOCAL.ENABLE else "bce",
            alpha=cfg.MODEL.TEACHER_STUDENT.DIS.FOCAL.ALPHA,
            gamma=cfg.MODEL.TEACHER_STUDENT.DIS.FOCAL.GAMMA,
        ) if cfg.MODEL.TEACHER_STUDENT.DIS.ENABLE else None

        depth_estimation = DEB(feature_shape) if cfg.MODEL.TEACHER_STUDENT.DEB.ENABLE else None

        return {
            "teacher": teacher_model,
            "student": student_model,
            "discriminator": discriminator,
            "backbone_out_feature": backbone_out_feature,
            "depth_estimation": depth_estimation,
            "source_losses_weight": cfg.MODEL.TEACHER_STUDENT.SOURCE_WEIGHT,
            "target_losses_weight": cfg.MODEL.TEACHER_STUDENT.TARGET_WEIGHT,
            "discriminator_losses_weight": cfg.MODEL.TEACHER_STUDENT.DIS.LOSS_WEIGHT,
            "depth_losses_weight": cfg.MODEL.TEACHER_STUDENT.DEPTH_WEIGHT,
            "teacher_update_step": cfg.MODEL.TEACHER_STUDENT.TEACHER_UPDATE_STEP,
            "strong_augmentation": cfg.INPUT.STRONG_AUG,
            "vis_period": cfg.VIS_PERIOD,
        }

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        if self.iter > 0 and self.iter % self.teacher_update_step == 0:
            self.update_teacher()

        source_inputs, target_inputs = batched_inputs

        pseudo_labels = self.get_pseudo_label(target_inputs)
        target_inputs = self.add_pseudo_labels(target_inputs, pseudo_labels["roi"])

        source_losses = self.student_forward(source_inputs, self.SOURCE_LABEL)
        target_losses = self.student_forward(target_inputs, self.TARGET_LABEL)

        losses = self.weight_losses(source_losses, target_losses)
        self.iter += 1

        return losses

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        return self.student.inference(batched_inputs, detected_instances, do_postprocess)

    @torch.no_grad()
    def get_pseudo_label(self, batched_inputs):
        self.teacher.eval()

        images = self.teacher.preprocess_image(batched_inputs)
        features = self.teacher.backbone(images.tensor)

        if self.teacher.proposal_generator is not None:
            proposals, _ = self.teacher.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.teacher.device) for x in batched_inputs]

        roi_results, _ = self.teacher.roi_heads(images, features, proposals, None)

        # Filter pseudo labels
        proposals = self.filter_proposals(proposals)
        roi_results = self.filter_roi(roi_results)

        return {
            "proposals": proposals,
            "roi": roi_results
        }

    def student_forward(self, batched_inputs, discriminator_label):
        assert self.student.training
        storage = get_event_storage()

        image_key = "image_strong_aug" if self.strong_augmentation else "image"
        images = self.preprocess_image(batched_inputs, image_key=image_key)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.student.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.student.backbone(images.tensor)

        discriminator_losses = {}
        if self.discriminator is not None:
            discriminator_losses = self.discriminator(features[self.backbone_out_feature], discriminator_label)

        deb_losses = {}
        if self.depth_estimation is not None and "depth" in batched_inputs[0]:
            gt_depth = self.preprocess_depth(batched_inputs)
            deb_losses, depth_map = self.depth_estimation(features[self.backbone_out_feature], gt_depth.tensor)
            if self.vis_period > 0 and storage.iter % self.vis_period == 0:
                ResnetDEB.visualize_training(batched_inputs, depth_map)

        if self.student.proposal_generator is not None:
            proposals, proposal_losses = self.student.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.student.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.student.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0 and storage.iter % self.vis_period == 0 and discriminator_label == self.TARGET_LABEL:
            self.student.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(discriminator_losses)
        losses.update(deb_losses)

        return losses

    def filter_proposals(self, proposals: List[Instances]):
        for i in range(len(proposals)):
            instance = proposals[i]
            proposals[i] = instance[instance.objectness_logits > self.confident_thresh]
        return proposals

    def filter_roi(self, proposals: List[Instances]):
        for i in range(len(proposals)):
            instance = proposals[i]
            proposals[i] = instance[instance.scores > self.confident_thresh]
        return proposals

    @classmethod
    def add_pseudo_labels(cls, batch_inputs, pseudo_labels):
        for data, instance in zip(batch_inputs, pseudo_labels):
            instance = instance.to("cpu")
            gt_instances = Instances(instance.image_size)
            gt_instances.gt_boxes = instance.pred_boxes
            gt_instances.gt_classes = instance.pred_classes
            data["instances"] = gt_instances
        return batch_inputs

    def weight_losses(self, source_losses: Dict[str, torch.Tensor], target_losses: Dict[str, torch.Tensor]):
        losses = {}
        for key, loss in source_losses.items():
            losses[f"source_{key}"] = loss * self._get_loss_weight(key, source=True)

        for key, loss in target_losses.items():
            if key.endswith("box_reg") or key.endswith("rpn_loc"):
                continue
            losses[f"target_{key}"] = loss * self._get_loss_weight(key, source=False)

        return losses

    def _get_loss_weight(self, loss_name, source):
        if loss_name.startswith("discriminator"):
            return self.discriminator_losses_weight
        elif loss_name.startswith("depth"):
            return self.depth_losses_weight
        else:
            return self.source_losses_weight if source else self.target_losses_weight

    def update_teacher(self, keep=0.9996):
        student_dict = self.student.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.teacher.state_dict().items():
            if key in student_dict.keys():
                new_teacher_dict[key] = student_dict[key] * (1 - keep) + value * keep
            else:
                raise Exception(f"{key} is not found in student model")

        self.teacher.load_state_dict(new_teacher_dict)

    def preprocess_depth(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Pad and batch the input depth map.
        """
        gt_depth = [x["depth"].to(self.student.device) for x in batched_inputs]
        gt_depth = ImageList.from_tensors(
            gt_depth,
            self.student.backbone.size_divisibility,
            padding_constraints=self.student.backbone.padding_constraints
        )
        return gt_depth

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def maybe_compile(self):
        logger = logging.getLogger(__name__)
        logger.parent = logging.getLogger('detectron2')
        if torch.version.__version__.startswith("2"):
            self.student = self._compile_base_model(self.student)
            self.teacher = self._compile_base_model(self.teacher)
            self.discriminator = torch.compile(self.discriminator)
            logger.info("Compile model success")
        else:
            logger.info(f"Torch compile only support version > 2. Current version f{torch.version.__version__}")

    @classmethod
    def _compile_base_model(cls, model: GeneralizedRCNN):
        model.backbone = torch.compile(model.backbone)
        assert isinstance(model.proposal_generator, RPN)
        model.proposal_generator.rpn_head = torch.compile(model.proposal_generator.rpn_head)
        model.proposal_generator.anchor_generator = torch.compile(model.proposal_generator.anchor_generator)
        assert isinstance(model.roi_heads, Res5ROIHeads)
        model.roi_heads.pooler = torch.compile(model.roi_heads.pooler)
        model.roi_heads.res5 = torch.compile(model.roi_heads.res5)
        model.roi_heads.box_predictor = torch.compile(model.roi_heads.box_predictor)
        return model

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], image_key: str = "image"):
        """
        Normalize, pad and batch the input images.
        """
        images = [move_device_like(x[image_key], self.student.pixel_mean) for x in batched_inputs]
        images = [(x - self.student.pixel_mean) / self.student.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.student.backbone.size_divisibility,
            padding_constraints=self.student.backbone.padding_constraints,
        )
        return images

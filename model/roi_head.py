import logging

import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.modeling import FastRCNNOutputLayers, ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from fvcore.nn import sigmoid_focal_loss_jit
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logger.parent = logging.getLogger('detectron2')


@ROI_HEADS_REGISTRY.register()
class TeacherStudentROIHead(Res5ROIHeads):
    @classmethod
    def from_config(cls, cfg, input_shape: ShapeSpec):
        ret = super().from_config(cfg, input_shape=input_shape)

        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        ret["box_predictor"] = TeacherStudentOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )
        return ret


class TeacherStudentOutputLayers(FastRCNNOutputLayers):

    @configurable()
    def __init__(self, *, use_focal, focal_loss_alpha, focal_loss_gamma, **kwargs):
        super().__init__(**kwargs)
        if use_focal:
            logger.info("ROI Head use focal loss")

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.use_focal = use_focal

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["use_focal"] = cfg.MODEL.TEACHER_STUDENT.ROI_FOCAL_LOSS.ENABLE
        ret['focal_loss_alpha'] = cfg.MODEL.TEACHER_STUDENT.ROI_FOCAL_LOSS.ALPHA
        ret['focal_loss_gamma'] = cfg.MODEL.TEACHER_STUDENT.ROI_FOCAL_LOSS.GAMMA
        return ret

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )

        prefix = "target" if len(gt_classes) > 0 and hasattr(proposals[0], "pseudo") else "source"
        _log_classification_stats(scores, gt_classes, prefix=prefix)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_focal:
            loss_cls = self.sigmoid_focal_loss(scores, gt_classes)
        elif self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def sigmoid_focal_loss(self, pred_class_logits, gt_classes):
        """
        Args:
            pred_class_logits: shape (N, K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
        """
        valid_mask = gt_classes >= 0

        gt_labels_target = F.one_hot(gt_classes[valid_mask], num_classes=self.num_classes + 1)[:, :-1]
        pred_class_logits = pred_class_logits[valid_mask, :-1]

        return sigmoid_focal_loss_jit(
            pred_class_logits,
            gt_labels_target.to(pred_class_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="mean",
        )


if __name__ == "__main__":
    x = [torch.randn(2, 5, 4), torch.randn(2, 5, 4)]
    y = cat(x, dim=1)
    print(y.shape)

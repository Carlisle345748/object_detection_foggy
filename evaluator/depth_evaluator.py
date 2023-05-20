from detectron2.evaluation import DatasetEvaluator
import torch.nn.functional as F
import torch


class DepthEvaluator(DatasetEvaluator):
    """
    DepthEvaluator inherits from DatasetEvaluator. It is used to evaluate the performance of a depth estimation model. The class has the following methods:
    """
    def __init__(self):
        self.mse = 0
        self.mae = 0
        self.count = 0
        self.silog = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0

    def reset(self):
        self.mse = 0
        self.mae = 0
        self.count = 0

    def process(self, inputs, outputs):
        for data, pred in zip(inputs, outputs):
            gt_depth_map = data["depth"]
            pred_depth_map = pred["depth"].to(gt_depth_map.device)
            if "depth_mean" in data:
                gt_depth_map = gt_depth_map * data["depth_std"] + data["depth_mean"]
                pred_depth_map = pred_depth_map * data["depth_std"] + data["depth_mean"]
            if "depth_max" in data:
                gt_depth_map = gt_depth_map * (data["depth_max"] - data["depth_min"]) + data["depth_min"]
                pred_depth_map = pred_depth_map * (data["depth_max"] - data["depth_min"]) + data["depth_min"]

            pred_depth_map = pred_depth_map[gt_depth_map != 0]
            gt_depth_map = gt_depth_map[gt_depth_map != 0]

            self.mse += F.mse_loss(pred_depth_map, gt_depth_map).item()
            self.mae += torch.mean(torch.abs(pred_depth_map - gt_depth_map)).item()
            self.silog += self.silog_loss(pred_depth_map, gt_depth_map).item()
            self.delta1 += self.delta_threshold(gt_depth_map, pred_depth_map, 1.25).item()
            self.delta2 += self.delta_threshold(gt_depth_map, pred_depth_map, 1.25 ** 2).item()
            self.delta3 += self.delta_threshold(gt_depth_map, pred_depth_map, 1.25 ** 3).item()
            
            self.count += 1

    def evaluate(self):
        return {
            "Depth MSE": self.mse / self.count,
            "Depth MAE": self.mae / self.count,
            "Depth SILog Error": self.silog / self.count,
            "Depth Delta1": self.delta1 / self.count,
            "Depth Delta2": self.delta2 / self.count,
            "Depth Delta3": self.delta3 / self.count,
        }

    @classmethod
    def silog_loss(cls, pred_depth: torch.Tensor, gt_depth: torch.Tensor):
        """
        Computes the Scale-Invariant Logarithmic Error (SILog) between the predicted and ground truth depth maps.
        """
        gt_depth_log = torch.log(torch.clamp(gt_depth, min=1e-6))
        pred_depth_log = torch.log(torch.clamp(pred_depth, min=1e-6))

        log_diff = gt_depth_log - pred_depth_log
        N = gt_depth.numel()  # Number of pixels

        squared_error = torch.sum(log_diff ** 2) / N
        squared_mean_error = torch.sum(log_diff) ** 2 / (N ** 2)

        silog_loss = squared_error - squared_mean_error

        return silog_loss

    @classmethod
    def delta_threshold(cls, y_true: torch.Tensor, y_pred: torch.Tensor, delta: float):
        ratio = torch.maximum(y_true / y_pred, y_pred / y_true)
        within_threshold = (ratio < delta).float()
        return torch.mean(within_threshold)

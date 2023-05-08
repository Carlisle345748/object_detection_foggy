from detectron2.evaluation import DatasetEvaluator
import torch.nn.functional as F
import torch


class DepthEvaluator(DatasetEvaluator):
    def __init__(self):
        self.mse = 0
        self.mae = 0
        self.count = 0

    def reset(self):
        self.mse = 0
        self.mae = 0
        self.count = 0

    def process(self, inputs, outputs):
        for data, pred in zip(inputs, outputs):
            gt_depth_map = data["depth"]
            pred_depth_map = pred["depth"]
            self.mse = F.mse_loss(pred_depth_map, gt_depth_map).to("cup")
            self.mae = torch.mean(torch.abs(pred_depth_map - gt_depth_map)).to("cup")

            self.count += 1

    def evaluate(self):
        return {
            "Depth MSE": self.mse / self.count,
        }

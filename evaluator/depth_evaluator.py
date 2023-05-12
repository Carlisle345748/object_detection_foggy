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
    
    def silog_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor):
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

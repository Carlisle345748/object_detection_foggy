import unittest
import torch
from depth_evaluator import DepthEvaluator

class TestDepthEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = DepthEvaluator()

    def test_silog_loss(self):
        pred_depth = torch.tensor([1, 2, 3, 4])
        gt_depth = torch.tensor([2, 3, 4, 5])
        silog = self.evaluator.silog_loss(pred_depth, gt_depth)
        expected_silog = (torch.mean((torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6)) ** 2) -
                          (torch.mean(torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6))) ** 2)
        self.assertAlmostEqual(silog.item(), expected_silog.item(), places=5)

    def test_silog_loss_with_zeros(self):
        pred_depth = torch.tensor([1, 2, 3, 4])
        gt_depth = torch.tensor([0, 3, 4, 5])
        silog = self.evaluator.silog_loss(pred_depth, gt_depth)
        expected_silog = (torch.mean((torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6)) ** 2) -
                          (torch.mean(torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6))) ** 2)
        self.assertAlmostEqual(silog.item(), expected_silog.item(), places=5)

    def test_silog_loss_with_large_values(self):
        pred_depth = torch.tensor([1000, 2000, 3000, 4000])
        gt_depth = torch.tensor([2000, 3000, 4000, 5000])
        silog = self.evaluator.silog_loss(pred_depth, gt_depth)
        expected_silog = (torch.mean((torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6)) ** 2) -
                          (torch.mean(torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6))) ** 2)
        self.assertAlmostEqual(silog.item(), expected_silog.item(), places=5)

    def test_silog_loss_with_2d_input(self):
        pred_depth = torch.tensor([[1, 2], [3, 4]])
        gt_depth = torch.tensor([[0, 3], [4, 5]])
        silog = self.evaluator.silog_loss(pred_depth, gt_depth)
        expected_silog = (torch.mean((torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6)) ** 2) -
                        (torch.mean(torch.log(gt_depth + 1e-6) - torch.log(pred_depth + 1e-6))) ** 2)
        self.assertAlmostEqual(silog.item(), expected_silog.item(), places=5)


if __name__ == '__main__':
    unittest.main()
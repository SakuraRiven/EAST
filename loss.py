import torch
import torch.nn as nn


class Loss(nn.Module):
	def __init__(self):
		super(Loss, self).__init__()
	
	def forward(self, gt_score, pred_score, gt_geo, pred_geo):
		inter = torch.sum(gt_score * pred_score)
		union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
		dice_loss = 1. - (2 * inter / union)

		d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
		d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
		area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
		area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
		w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
		h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
		area_intersect = w_union * h_union
		area_union = area_gt + area_pred - area_intersect
		iou_loss = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
		angle_loss = 1 - torch.cos(angle_pred - angle_gt)
		geo_loss = iou_loss + 10 * angle_loss
		print('dice loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(dice_loss, \
                     torch.mean(angle_loss*gt_score), torch.mean(iou_loss*gt_score)))
		return 100 * torch.mean(geo_loss * gt_score) + dice_loss

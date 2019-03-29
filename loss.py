import torch
import torch.nn as nn


def get_ce_loss(gt_score, pred_score):
	mask = (gt_score.int() != 0).float()
	num_positive = torch.sum(mask)
	num_negative = mask.numel() - num_positive
	mask[gt_score.int() != 0] = num_negative / (num_positive + num_negative)
	mask[gt_score.int() == 0] = num_positive / (num_positive + num_negative)
	return nn.functional.binary_cross_entropy_with_logits(pred_score, gt_score, weight=mask)


def get_dice_loss(gt_score, pred_score):
	inter = torch.sum(gt_score * pred_score)
	union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
	return 1. - (2 * inter / union)
	 

def get_geo_loss(gt_geo, pred_geo):
	d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
	d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
	area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
	area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
	w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
	h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
	area_intersect = w_union * h_union
	area_union = area_gt + area_pred - area_intersect
	iou_loss_map = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
	angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
	return iou_loss_map, angle_loss_map


class Loss(nn.Module):
	def __init__(self):
		super(Loss, self).__init__()

	def forward(self, gt_score, pred_score, gt_geo, pred_geo):
		classify_loss = get_dice_loss(gt_score, pred_score)
		#classify_loss = 1000 * get_ce_loss(gt_score, pred_score)
		iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

		angle_loss = torch.sum(angle_loss_map*gt_score) / torch.sum(gt_score)
		iou_loss = torch.sum(iou_loss_map*gt_score) / torch.sum(gt_score)
		geo_loss = 10 * angle_loss + iou_loss
		print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
		return geo_loss + classify_loss

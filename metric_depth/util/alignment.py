# Author: Bingxin Ke
# Last modified: 2024-01-11

import numpy as np
import torch
import torch.nn.functional as F  

def align_depth_least_square_torch(  
    gt_arr: torch.Tensor,  
    pred_arr: torch.Tensor,  
    valid_mask_arr: torch.Tensor,  
    return_scale_shift=True,  
    max_resolution=None, 
 ):  
    # 获取原始形状  
    ori_shape = pred_arr.shape  
  
    # # 压缩多余的维度（如果有的话）  
    # gt = gt_arr.squeeze()  # [H, W]
    gt = gt_arr.detach()
    # pred = pred_arr.squeeze()  
    pred = pred_arr
    valid_mask = valid_mask_arr.bool()  
  
    # Downsample  
    if max_resolution is not None:  
        scale_factor = min(max_resolution / torch.tensor(ori_shape[-2:], dtype=torch.float32)).item()  
        if scale_factor < 1:  
            downscaler = F.interpolate(input=torch.unsqueeze(torch.zeros_like(gt), 0), scale_factor=scale_factor, mode='nearest', align_corners=False).shape[2:]  
            gt = F.interpolate(input=gt.unsqueeze(0).unsqueeze(0), size=downscaler, mode='nearest', align_corners=False).squeeze(0).squeeze(0)  
            pred = F.interpolate(input=pred.unsqueeze(0).unsqueeze(0), size=downscaler, mode='nearest', align_corners=False).squeeze(0).squeeze(0)  
            valid_mask = F.interpolate(input=valid_mask.unsqueeze(0).unsqueeze(0).float(), size=downscaler, mode='nearest', align_corners=False).squeeze(0).squeeze(0).bool()  
  
    assert (  
        gt.shape == pred.shape == valid_mask.shape  
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"  
  
    # 应用掩码并准备数据  
    gt_masked = gt[valid_mask].reshape(-1, 1)  
    pred_masked = pred[valid_mask].reshape(-1, 1)  
    _ones = torch.ones_like(pred_masked)  
  
    # 构造A矩阵并求解最小二乘问题  
    A = torch.cat([pred_masked, _ones], dim=-1)  
    X, *_ = torch.linalg.lstsq(A, gt_masked)  
    scale, shift = X[:, 0]  
  
    # # 应用缩放和平移  
    # aligned_pred = pred_arr * scale + shift  
  
    # # 恢复维度  
    # aligned_pred = aligned_pred.reshape(ori_shape)  
  
    if return_scale_shift:  
        return scale.item(), shift.item()  
 

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


# ******************** disparity space ********************
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)

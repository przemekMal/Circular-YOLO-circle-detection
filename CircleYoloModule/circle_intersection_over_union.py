# -*- coding: utf-8 -*-
"""circle_intersection_over_union.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t33QRVZ9pTNmv6Jua1c2-XPQxQzTFN7o
"""

import torch

def intersection_over_union(predictions, labels):
    """
    Args:
        - predictions (tensor): Tensor containing predicted circles in the format (N, 3), where N is the number of
          predictions and each circle is represented by its center coordinates (x, y) and radius (r).
        - labels (tensor): Tensor containing ground truth circles in the same format as predictions.

    Returns:
        - float: IoU value for all examples in the input tensors.

    This function computes the Intersection over Union (IoU) metric for a set of predicted and ground truth circles
    represented by their center coordinates (x, y) and radius (r). In contrast to the standard YOLO function, where
    IoU is calculated for bounding boxes defined by their width and height, here the IoU is computed for circles.

    """
    # Max and min for radius and cords
    maxes_r = torch.max(predictions[..., 2:3], labels[..., 2:3])
    mins_r = torch.min(predictions[..., 2:3], labels[..., 2:3])

    # Distance between circle centers
    d = torch.sqrt((predictions[..., 0:1] - labels[..., 0:1])**2 + (predictions[..., 1:2] - labels[..., 1:2])**2)

    #What type of area lenses or concave lenses
    sp = torch.gt((d - maxes_r ), 0) #If < 0 then concave lens

    # Cheking conditions for tensors
    base_condition = torch.logical_and(torch.logical_and(maxes_r > 0, mins_r > 0), d < (maxes_r + mins_r))

    part_condition = torch.logical_and((maxes_r - mins_r) < d, base_condition)
    
    condition_concave_lens = torch.logical_and(part_condition, torch.logical_not(sp))

    condition_lens = torch.logical_and(part_condition, sp)

    condition_inside =  torch.logical_and(base_condition, d + mins_r <= maxes_r)

    # Calculate intersection area for condition
    d_pow_2, mins_r_pow_2, maxes_r_pow_2  = d**2, mins_r**2, maxes_r**2, 

    theta = torch.where(condition_concave_lens, 2*torch.arccos((maxes_r-d)/mins_r), torch.zeros_like(d))

    AuB = torch.where(condition_concave_lens,
                       # Circle Area - segment area
                       (torch.pi * mins_r_pow_2) -
                       # segment area = 0.5 * radius**2 * (angle_radians - sin(angle_radians))
                       0.5 * mins_r_pow_2 * (theta - torch.sin(theta))
                       , torch.zeros_like(d)) # Default values for non condition

    AuB = torch.where(condition_lens,
                       mins_r_pow_2 * torch.acos((d_pow_2 + mins_r_pow_2 - maxes_r_pow_2) / (2 * d * mins_r)) +
                       maxes_r_pow_2 * torch.acos((d_pow_2 + maxes_r_pow_2 - mins_r_pow_2) / (2 * d * maxes_r)) -
                       0.5 * torch.sqrt(4*d_pow_2*maxes_r_pow_2-(d_pow_2-mins_r_pow_2+maxes_r_pow_2)**2),
                      AuB)

    AuB = torch.where(condition_inside,
                      (torch.pi * mins_r_pow_2),
                      AuB)

    #AuB = torch.where(torch.isnan(AuB), torch.tensor(0), AuB)

    # Calculate output IoU
    #A = maxes_r**2 *PI
    #B = mins_r**2  *PI
    #                 PI   *      (A       +       B)
    return (AuB / (torch.pi*(maxes_r_pow_2 + mins_r_pow_2) - AuB))
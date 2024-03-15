# -*- coding: utf-8 -*-
"""mAP_utilities.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19_dSfRL208qFytyw_n3M02X2wE3QpVmM
"""
from CircleYolov1Module.circle_intersection_over_union import intersection_over_union

def non_max_suppression(apples, iou_threshold=0.5, threshold = 0.5):
    """
    Performs Non-Maximum Suppression (NMS) on a list of detected objects.

    Args:
        detections (list): A list of detected objects, where each object is represented as a tuple.
                           Each tuple consists of class index, confidence score, and bounding box coordinates.
        iou_threshold (float): Intersection over Union (IoU) threshold for considering two bounding boxes as the same object.
                               Default is 0.5.
        threshold (float): Confidence score threshold for filtering detections. Default is 0.5.

    Returns:
        list: List of selected detections after Non-Maximum Suppression.
    """
    assert type(apples) == list

    # Filter detections based on confidence threshold
    apples = [apple for apple in apples if apple[1]>threshold]

    # Sort detections by confidence score in descending order
    apples = sorted(apples, key=lambda x: x[1], reverse=True)
    apples_to_return = []
    while apples:
        chosen_apple = apples.pop(0)
        apples = [
            apple
            for apple in apples
            if apple[0] != chosen_apple[0] # Different class
            or intersection_over_union(
                torch.tensor(chosen_apple[2:]),
                torch.tensor(apple[2:])
            )
            < iou_threshold
        ]
        apples_to_return.append(chosen_apple)
    return apples_to_return


def convert_grid_boxes(predictions, S=7, C=1):
    """
    Convert grid-based bounding box predictions to entire image ratio.

    Args:
        predictions (torch.Tensor): Tensor of bounding box predictions, with shape [batch_size, S, S, -1].
                                     Each prediction consists of class probabilities, bounding box coordinates (x, y, r), where r represent radius.
        S (int): Size of the grid. Default is 7.
        C (int): Number of classes. Default is 1.

    Returns:
        torch.Tensor: Converted bounding box predictions with shape [batch_size, S, S, -1], where each prediction is represented as (class_index, confidence, x, y, r).
    """
    predictions = predictions.to('cpu') # Move predictions to CPU
    batch_size = predictions.shape[0] # Get batch size
    predictions = predictions.reshape(batch_size, S, S, -1) # Reshape predictions to [batch_size, S, S, -1]

    # Determine the number of bounding box predictions per grid cell
    shape_size_loop = int((predictions.shape[-1] - C)/4)

    # Iterate over each bounding box prediction
    for bbox in range(1, shape_size_loop, 1):
        # Create a mask to select bounding box predictions with higher confidence scores
        mask = predictions[..., C:(C+1)] < predictions[..., (C+4*bbox):(C+1+4*bbox)]
        # Create a slice to select the current bounding box predictions
        bbox_slice = slice(C + 4 * bbox, C + 4 * bbox + 4)
        # Update bounding box predictions with higher confidence scores
        predictions[..., C:(C+4)] = torch.where(mask, predictions[..., bbox_slice], predictions[..., C:(C+4)])

    # Extract class probabilities, bounding box coordinates, and radius from predictions
    prob = (predictions[...,C:(C+1)])
    cell_indicies = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1/S * (predictions[...,(C+1):(C+2)] + cell_indicies)
    y = 1/S * (predictions[...,(C+2):(C+3)] + cell_indicies.permute(0, 2, 1, 3))
    r = predictions[..., (C+3):(C+4)]

    # Determine class indices based on the highest probability
    if C < 2:
        class_idx = torch.zeros_like(r)
    else:
        class_idx = predictions[..., 0:C].argmax(-1).unsqueeze(-1)

    # Concatenate class index, confidence, x, y, and radius to form converted bounding box predictions
    converted_boxes = torch.cat((class_idx, prob, x, y, r), dim=-1)
    return converted_boxes

def grid_boxes_to_boxes(grid_boxes, S: int = 7, C: int = 1):
    """
    Convert grid-based bounding box predictions to individual bounding boxes.

    Args:
        grid_boxes (torch.Tensor): Tensor of grid-based bounding box predictions with shape [batch_size, 7, 7, 4].
                                   Each prediction consists of class probabilities, bounding box coordinates (x, y, r), where r represents radius.
        S (int): Size of the grid. Default is 7.
        C (int): Number of classes. Default is 1.

    Returns:
        List: A list of individual bounding boxes for each image in the batch.
              Each bounding box is represented as [class_index, confidence, x, y, r].
    """
    # Convert grid-based bounding box predictions to entire image ratio
    converted_boxes = convert_grid_boxes(grid_boxes, S=S, C=C).reshape(grid_boxes.shape[0], S*S, -1)
    converted_boxes[..., 0] = converted_boxes[..., 0].long()
    all_boxes = []
    # Iterate over each image in the batch
    for batch_idx in range(grid_boxes.shape[0]):
        boxes = []
        # Iterate over each grid cell
        for box_idx in range(S*S):
            boxes.append([x.item() for x in converted_boxes[batch_idx, box_idx, :]])
        all_boxes.append(boxes)
    return all_boxes

def get_bboxes(model, loader, device, IoU_threshold: float = 0.5, threshold: float = 0.5, S: int = 7, C: int = 1):
    """
    Extracts bounding boxes from model predictions and true labels.

    Args:
        model (torch.nn.Module): Trained model for predicting bounding boxes.
        loader (torch.utils.data.DataLoader): DataLoader containing images and corresponding labels.
        IoU_threshold (float): Intersection over Union threshold for non-maximum suppression. Default is 0.5.
        threshold (float): Threshold for filtering out low confidence predictions. Default is 0.4.
        S (int): Size of the grid used in the model. Default is 7.
        C (int): Number of classes. Default is 1.

    Returns:
        Tuple: A tuple containing lists of predicted and true bounding boxes respectively.
               Each bounding box is represented as [image_index, class_index, confidence, x, y, r].
    """
    all_pred_box = [] # List to store predicted bounding boxes
    all_true_box = [] # List to store true bounding boxe
    train_idx = 0 # Index for tracking the image
    model.eval() # Set model to evaluation mode

    for batch_idx, (x, labels) in enumerate(loader):
        x, labels = x.to(device), labels.to(device) # Move data to device
        with torch.no_grad():
            predictions = model(x) # Forward pass

        # Convert predictions and labels to individual bounding boxes
        bboxes = grid_boxes_to_boxes(predictions, S=S, C=C)
        true_labels = grid_boxes_to_boxes(labels, S=S, C=C)
        batch_size = x.shape[0]

        # Iterate over each image in the batch
        for idx in range(batch_size):
            # Apply non-maximum suppression to predictions
            nms_predictions = non_max_suppression(bboxes[idx], IoU_threshold, threshold)

            # Append predicted bounding boxes to the list
            for pred in nms_predictions:
                all_pred_box.append([train_idx] + pred)

            # Append true bounding boxes to the list
            for true_leb in true_labels[idx]:
                if true_leb[1] > threshold:
                    all_true_box.append([train_idx] + true_leb)

            train_idx += 1 # Increment image index

    model.train() # Set model back to training mode
    return all_pred_box, all_true_box # Return lists of predicted and true bounding boxes


import torch
from collections import Counter

#mAP@0.5:0.1:0.95
def mean_average_precision(
    pred_boxes,               # List of predicted bounding boxes in format (img_idx, class, probability, x, y, r)
    true_boxes,               # List of true bounding boxes in format (img_idx, class, probability, x, y, r)
    threshold_mAP=0.5,        # Initial IoU threshold for mAP calculations. Default: 0.5
    step_threshold=0.05,      # Step size for increasing the threshold in mAP calculations. Default: 0.05
    stop_threshold_mAP=0.95,  # Threshold at which to stop mAP calculations. Default: 0.95k
    C=1,                      # Number of classes. Default: 1
    epsilon=1e-12             # Small additional value in the denominator to avoid division by zero. Default: 1e-12
    ):
    """
    Calculates the mean average precision for a set of predicted and true bounding boxes.

    Args:
        pred_boxes (list): List of predicted bounding boxes.
        true_boxes (list): List of true bounding boxes.
        threshold_mAP (float): Initial IoU threshold for mAP calculations. Default: 0.5.
        step_threshold (float): Step size for increasing the threshold in mAP calculations. Default: 0.05.
        stop_threshold_mAP (float): Threshold at which to stop mAP calculations. Default: 0.95.
        C (int): Number of classes. Default: 1.
        epsilon (float): Small additional value in the denominator to avoid division by zero. Default: 1e-12.

    Returns:
        float: Mean average precision (mAP).
    """
    mean_average_precision = [] # Table of values mAP for each step

    # Iterating calulate mAP while threshold is lower than end case
    while threshold_mAP < stop_threshold_mAP:
        average_precision = [] # List of Average Precision for current threshold step
        # Calculate Average Precision for each class
        for c in range(C):
            detection_list = []
            ground_truhts = []
            for detection in pred_boxes:
                if detection[1] == c:
                    detection_list.append(detection)
            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truhts.append(true_box)
            #Making dic, wich reprezented how many bbox, imgs have
            #amount_boxes = {0:3, 1:5, 2:3}
            amount_bboxes = Counter([gt[0] for gt in ground_truhts])

            #changing dir to had val = zero tensors, with shape reprezented num of imgs
            #amount_boxes = {0:torch.zeros([0,0,0]), 1:torch.zeros([0,0,0,0,0]), 2:torch.zeros([0,0,0])}
            for key, val in amount_bboxes.items():
                amount_bboxes[key]= torch.zeros(val)
            #sort over probability
            detection_list.sort(key= lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detection_list)))
            FP = torch.zeros((len(detection_list)))
            total_true_boxes = len(ground_truhts)

            for detection_idx, detection in enumerate(detection_list):
              #Taking only bboxs for correct img
                ground_truth_img = [ bbox for bbox in ground_truhts if bbox[0] == detection[0] ]
              #number of target bbox
                num_gts = len(ground_truth_img)
                best_iou = 0
                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(torch.tensor(gt[3:]), torch.tensor(detection[3:]))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                if best_iou > threshold_mAP:
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            #cumulative sum for recall [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_boxes+epsilon)
            precision = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            precision = torch.cat((torch.tensor([1]), precision))
            recalls = torch.cat((torch.tensor([0]), recalls))
            average_precision.append(torch.trapz(precision, recalls))

        mean_average_precision.append(sum(average_precision)/len(average_precision))
        threshold_mAP += step_threshold

    return sum(mean_average_precision)/len(mean_average_precision)
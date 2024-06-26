# -*- coding: utf-8 -*-
"""yolo_training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DY8HUT9vtwFmn4l-VN2X4u8cskcMKMSI
"""

import torch
import torch.cuda.amp

class ModelCheckpoint:
    """
    Callback class to save the best model weights based on a monitored metric.

    Args:
        filepath (str): Path to save the best model weights.
        monitor (str): Metric to monitor for saving the best weights. Default is 'loss'.
        mode (str): Mode for comparison, either 'min' or 'max'. Default is 'min'.
        threshold (float): Threshold to trigger saving the weights. Default is 0.0.

    Attributes:
        filepath (str): Path to save the best model weights.
        monitor (str): Metric being monitored.
        mode (str): Mode for comparison, either 'min' or 'max'.
        threshold (float): Threshold to trigger saving the weights.
        best_value (float): Current best value of the monitored metric.
        is_better (function): Function to compare the current value with the best value.
        best_weights (dict): Best model weights saved so far.

    Methods:
        __call__(self, current_value, model): Method to call the callback. Checks if the current value
            is better than the best value, updates the best value and saves the model weights accordingly.
    """
    def __init__(self, filepath, monitor='loss', mode='min', threshold=0.0):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.threshold = threshold
        self.best_value = None
        self.is_better = None
        self._init_is_better(mode, threshold)
        self.best_weights = None

    def _init_is_better(self, mode, threshold):
        if mode == 'min':
            self.is_better = lambda a, best: a < best - threshold
            self.best_value = float('inf')
        else:
            self.is_better = lambda a, best: a > best + threshold
            self.best_value = float('-inf')

    def __call__(self, current_value, model):
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_weights = model.state_dict()
            torch.save(model.state_dict(), self.filepath)
            print(f"Model weights saved with {self.monitor} of {current_value}")

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               yolo_loss_fn: torch.nn.Module,
               scaler: torch.cuda.amp.GradScaler,
               mean_average_precision = None,
               optimizer: torch.optim.Optimizer = None,
               checkpoint = None,
               device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Function to perform one training step on the YOLOv1 model.

    Args:
        model (torch.nn.Module): YOLOv1 model to be trained.
        data_loader (DataLoader): DataLoader containing the training data.
        yolo_loss_fn (torch.nn.Module): Loss function for YOLOv1 model.
        mean_average_precision (callable, optional): Mean Average Precision calculation function. Default is None.
        optimizer (torch.optim.Optimizer, optional): Optimizer for the model parameters. Default is None.
        checkpoint (callable, optional): Checkpoint callback function to save model weights. Default is None.
        device (torch.device, optional): Device on which to perform training. Default is CUDA if available, else CPU.

    Returns:
        float: Average training loss for the epoch.
    """
    mean_train_loss = []

    model.train()
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)
        
        if scaler:
            # Automated mixed precision policy
            with torch.cuda.amp.autocast(): #Automated mixed policy
                # 1. Forward pass
                y_pred = model(X)
        
                # 2. Calculate loss
                train_loss = yolo_loss_fn(y_pred, y)
                
            # 3. Loss backward
            scaler.scale(train_loss).backward()
    
            # 4. Optimizer step
            #scaler.unscale_(optimizer)
            scaler.step(optimizer)
        
            # 5. Scaler update
            scaler.update()
        
            # 6. Optimizer zero grad
            optimizer.zero_grad()
        else:
            # Zero your gradients for every batch!
            optimizer.zero_grad()
    
            # Make predictions for this batch
            y_pred = model(X)
    
            # Compute the loss and its gradients
            train_loss = yolo_loss_fn(y_pred, y)
            train_loss.backward()
    
            # Adjust learning weights
            optimizer.step()

        
        mean_train_loss.append(train_loss.item())

    avg_loss = sum(mean_train_loss)/(len(mean_train_loss))
    # Save checkpoint
    if checkpoint:
        checkpoint(avg_loss, model)
    print(f"Train loss: {avg_loss:.3f}")
    return avg_loss

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               yolo_loss_fn: torch.nn.Module,
               checkpoint = None,
               device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Function to perform one testing step on the YOLOv1 model.

    Args:
        model (torch.nn.Module): YOLOv1 model to be tested.
        data_loader (DataLoader): DataLoader containing the testing data.
        yolo_loss_fn (torch.nn.Module): Loss function for YOLOv1 model.
        checkpoint (callable, optional): Checkpoint callback function to save model weights. Default is None.
        device (torch.device, optional): Device on which to perform testing. Default is CUDA if available, else CPU.

    Returns:
        float: Average testing loss for the epoch.
    """
    mean_test_loss = []

    model.eval()
    for batch, (X_test, y_test) in enumerate(data_loader):
        # Send data to GPU
        X_test, y_test = X_test.to(device), y_test.to(device)

        # 1. Prediction test
        with torch.no_grad():
            y_pred_test = model(X_test)

        # 2. Calculate loss
        test_loss = yolo_loss_fn(y_pred_test, y_test)
        mean_test_loss.append(test_loss.item())


    avg_loss_test = sum(mean_test_loss)/(len(mean_test_loss))
    # Save checkpoint
    if checkpoint:
        checkpoint(avg_loss_test, model)
    print(f"Test loss: {avg_loss_test:.3f}")

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds\n")
    return total_time
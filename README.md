## Usage

# YOLOv1-Circle Training Script

This project implements a YOLOv1 model using circular bounding boxes for object detection. The model is trained on custom datasets and leverages PyTorch for deep learning.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- CUDA (optional, for GPU acceleration)
- Colab environment (for the given paths)

### Installation

1. Install dependencies by running the following command:

   ```bash
   pip install torch torchvision
   ```

2. Clone the repository and place your datasets in the specified directories or update the paths accordingly.

## Training Instructions

1. **Update Paths**:
   - Modify the following paths in the script to point to your own directories:
     - `PATH_TO_SAVE_MODEL`: Path to save the trained YOLO model.
     - `PATH_TO_SAVE_CHECKPOINT`: Path to save model checkpoints.
     - `DST_PATH_END_CSV`: Path to save the CSV file containing training loss and AP (Average Precision).
     - `ARH_PATH_0`, `ARH_PATH_1`, `ARH_PATH_2`: Paths to your dataset archives. These archives will be unpacked during training.

2. **Training Configuration**:
   - The model hyperparameters (such as batch size, learning rate, and epochs) can be adjusted in the script. The default settings are:
     - `BATCH_SIZE = 64`
     - `LEARNING_RATE = 1e-5`
     - `EPOCHS = 250`
   - Other important settings like image size, number of classes, and stride are also configurable in the script.

3. **Run the Script**:
   - Ensure that the necessary datasets are available, and the paths to the datasets are correctly specified.
   - To start the training, execute the following command:

     ```bash
     python train_yolov1_circle.py
     ```

4. **Model Checkpoints**:
   - The model will save checkpoints every 5 epochs (or as specified). If `LOAD_MODEL = True`, the script will attempt to load the last checkpoint to resume training.

5. **CSV Logging**:
   - During training, the loss and AP for each epoch will be logged into the CSV file specified by `DST_PATH_END_CSV`.

## Example Paths

```python
PATH_TO_SAVE_MODEL = '/content/drive/MyDrive/Models/yoloV1_circle.pth'
PATH_TO_SAVE_CHECKPOINT = '/content/drive/MyDrive/Models/checkpoint_yoloV1_circle.pth'
DST_PATH_END_CSV = '/content/drive/MyDrive/Results/yoloV1_loss_ap.csv'
ARH_PATH_0 = '/content/drive/MyDrive/Datasets/dataset1.zip'
ARH_PATH_1 = '/content/drive/MyDrive/Datasets/dataset2.zip'
ARH_PATH_2 = '/content/drive/MyDrive/Datasets/dataset3.zip'
```

Make sure to modify the paths above to match the location of your files and directories.

## License

This repository is licensed under the [MIT License](LICENSE).

## Additional Information

This repository presents a variation of the YOLOv1 model where instead of height and width in bounding boxes, radius is used. All functions are based on detecting circles instead of rectangles. Additionally, a model has been trained for detecting apples, achieving a final mAP (mean Average Precision) of 25% over the range from 0.5 to 0.95.

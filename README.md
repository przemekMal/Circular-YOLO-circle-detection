# Machine Learning Repository

This repository hosts a collection of machine learning projects implemented in PyTorch, covering object detection, classification, and more.

## Object Detection

The object detection projects in this repository are inspired by the work of Aladdin Persson. You can find the original source code in the following GitHub repository:

- Repository: [Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection)
- Author: Aladdin Persson

I have adapted the object detection implementations from this repository as a foundation for our projects.

## Usage

...

## License

This repository is licensed under the [MIT License](LICENSE).

## Training

### YOLOv1 Model Training with Circular Bounding Boxes

This project uses YOLOv1 with circular bounding boxes to detect objects in a custom dataset. Below is a guide on how to set up and train the model, including the paths you need to edit based on your environment.

1. **Hyperparameters:**
   - **Batch size:** 64
   - **Image size:** 448x448 pixels
   - **Learning rate:** 1e-5
   - **Weight decay:** 1e-4
   - **Epochs:** 250
   - **Number of classes:** 1
   - **Stride:** 7 (used for segmentation)
   - **Bounding boxes per grid:** 1
   - **Lambda values:** `lambda_noobj = 0.5`, `lambda_coord = 5` (used in loss function)
   - **Model saving path:** `PATH_TO_SAVE_MODEL` (see below for details)

2. **Paths to Edit:**
   - **Model Save Path:**
     ```python
     PATH_TO_SAVE_MODEL = '/content/drive/MyDrive/!MojeYolov1/Modele_magisterka/model_yolo1-1_circle_1BBox_noobj-0,5_coord-5.pth'
     ```
     This is the path where the trained model will be saved. Edit this to the directory where you want to store your trained model.
   
   - **Dataset Paths:**
     ```python
     ARH_PATH_0 = '/content/drive/MyDrive/!MojeYolov1/archiwumJabłkaParts.zip'  # Dataset 1
     ARH_PATH_1 = '/content/drive/MyDrive/!MojeYolov1/archiwumMinnerApple.zip'  # Dataset 2
     ARH_PATH_2 = '/content/drive/MyDrive/!MojeYolov1/archiwumACFR_multifruit-2016(apples_only).zip'  # Dataset 3
     ```
     These paths point to the archived datasets. You need to update them to reflect the actual locations of your datasets in your Google Drive or local environment.

   - **Script Path:**
     Ensure that your Python script can find the required custom modules by adding their paths:
     ```python
     sys.path.append('/content/drive/MyDrive/Colab Notebooks/YOLO')
     ```
     Update this path to where your `YOLO` module and scripts (`CircleYoloModule`, `utilities`, etc.) are stored.

3. **Data Preparation:**
   - The dataset is split into training, validation, and testing sets (90%, 5%, and 5%, respectively). The images are transformed (resized, color jittered) and loaded using PyTorch’s `DataLoader` for efficient parallel loading.
   - To modify the dataset unpacking and transformations:
     ```python
     dataset_train, dataset_val, dataset_test = dataset_utilities.unpackDatasets(
       '/content',  # Update this to your working directory
       transforms,  # Data augmentation and normalization
       "V1",        # YOLOv1 specific flag
       None,        # No anchor boxes for YOLOv1
       (0.9, 0.05, 0.05),  # Split ratios for train/val/test
       ARH_PATH_0, ARH_PATH_1, ARH_PATH_2  # Update paths to your datasets
     )
     ```

4. **Training Process:**
   - The model is trained on a GPU (if available). The training loop processes images in batches, computes the loss, and optimizes using the Adam optimizer.
   - Every 5th epoch, mAP is calculated using the validation set to monitor model performance. Model checkpoints are saved when a higher mAP is achieved.

5. **Saving Results:**
   - The final results, including the epoch number, training loss, and validation mAP, are saved in a CSV file:
     ```python
     path_to_CSV = '/content/epoch_loss_ap.csv'  # Update this to your CSV file path
     move('/content/epoch_loss_ap.csv', DST_PATH_END_CSV)
     ```
     Update the CSV file paths based on your directory structure.

6. **Recommendations:**
   - **Google Colab** is highly recommended for training this model due to its access to powerful GPUs and the seamless integration with Google Drive for storing datasets and models.

## Additional Information

This repository presents a variation of the YOLOv1 model where instead of height and width in bounding boxes, radius is used. All functions are based on detecting circles instead of rectangles. Additionally, a model has been trained for detecting apples, achieving a final mAP (mean Average Precision) of 25% over the range from 0.5 to 0.95.

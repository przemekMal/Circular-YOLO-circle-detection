# Machine Learning Repository

This repository hosts a collection of machine learning projects implemented in PyTorch, covering object detection, classification, and more.

## Object Detection

The object detection projects in this repository are inspired by the work of Aladdin Persson. You can find the original source code in the following GitHub repository:

- Repository: [Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection)
- Author: Aladdin Persson

I have adapted the object detection implementations from this repository as a foundation for our projects.

## Usage

YOLO Circle Bounding Box Training

This script trains a custom YOLOv3 model with circular bounding boxes on a specified dataset. The model is built for detecting objects represented by circular bounding boxes instead of traditional rectangular ones, providing enhanced accuracy for circular or round objects.
Requirements:

    Torch for model and tensor operations
    TorchVision for transformations and augmentation
    Utilities provided in CircleYoloModule, including custom loss functions and data utilities

Steps:

    Set Up Hyperparameters:
        You can adjust batch size, learning rate, weight decay, and more in the HIPERPARAMETERS section.
        Ensure the paths to save the model and checkpoints are set according to your file structure.

    Data Preparation:
        Datasets are unpacked and transformed. You need to specify your data archives using the paths provided:
            Example datasets:
                ARH_PATH_0 = /content/drive/MyDrive/!MojeYolov1/archiwumJabłkaParts.zip
                ARH_PATH_1 = /content/drive/MyDrive/!MojeYolov1/archiwumMinnerApple.zip
                ARH_PATH_2 = /content/drive/MyDrive/!MojeYolov1/archiwumACFR_multifruit-2016(apples_only).zip
        This code splits the dataset into training, validation, and testing sets with a ratio of 90%, 5%, and 5% respectively.
        Transforms include resizing, color jittering, and normalization for image augmentation.

    Model Training:
        The model structure is defined in the Yolo_V3 class.
        You can load a pre-trained model by setting LOAD_MODEL to True and providing the path to the model weights.
        During training, the script will periodically save the best-performing model based on the highest mAP (mean Average Precision) and store a CSV file with the epoch loss and mAP values for later analysis.
        Checkpoints are saved every epoch, and you can restart training from the last saved checkpoint.

    Evaluation:
        The evaluation is performed every 10 epochs, using mAP @50 as the metric. If the mAP reaches 95%, the training stops early.

    File Paths to Edit:
        Model Save Path: Adjust PATH_TO_SAVE_MODEL to specify where to save the trained model.
        Checkpoint Save Path: Update PATH_TO_SAVE_CHECKPOINT to specify the location for saving checkpoints.
        CSV Save Path: Set DST_PATH_END_CSV to the path for saving the CSV containing loss and mAP values.
        Dataset Paths: Make sure the archive paths (ARH_PATH_0, ARH_PATH_1, etc.) point to your dataset files.

Recommended Environment:

For this script, I highly recommend using Google Colab, as it provides free access to GPU resources, making training faster and more efficient. The script can be executed seamlessly in Colab by uploading your dataset and setting the appropriate file paths.

## License

This repository is licensed under the [MIT License](LICENSE).

## Additional Information

This repository presents a variation of the YOLOv1 model where instead of height and width in bounding boxes, radius is used. All functions are based on detecting circles instead of rectangles. Additionally, a model has been trained for detecting apples, achieving a final mAP (mean Average Precision) of 25% over the range from 0.5 to 0.95.

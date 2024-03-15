# -*- coding: utf-8 -*-
"""dataset_utilities.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13shGHARrI0TR_SHc-gAKI9lIqDp6pfpj
"""

import os
import shutil

def deleteFiles(folder_path):
    """
    Deletes all files and folders from the specified folder path, and then removes the main folder itself.

    Args:
    - folder_path (str): The path of the folder whose contents are to be deleted.

    Note:
    - The function is irreversible and leads to permanent data loss. Use with caution.
    """
    # Get the list of folder contents
    folder_contents = os.listdir(folder_path)

    # Iterate through each item in the folder contents list
    for item in folder_contents:
        item_path = os.path.join(folder_path, item)
        # If the item is a file, delete it
        if os.path.isfile(item_path):
            os.remove(item_path)
        # If the item is a directory, remove it along with its contents
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    # Also remove the main folder along with its contents
    shutil.rmtree(folder_path)

def concateFolder(dst_folder: str, source_folder_list: list[str]):
    """
    Concatenates multiple source folders into a single destination folder.

    Args:
    - dst_folder (str): The destination folder where the contents of source folders will be concatenated.
    - source_folder_list (list[str]): A list of source folders whose contents will be concatenated.

    Note:
    - The function moves all files from each source folder to the destination folder.
    - After concatenation, each source folder is deleted.

    Example:
    ```python
    concateFolder("combined_folder", ["folder1", "folder2", "folder3"])
    ```
    """
    # Ensure destination folder exists; if not, create it
    if not os.path.exists(dst_folder): os.makedirs(dst_folder)
    # Iterate through each source folder in the list
    for f in source_folder_list:
        # Get the list of files in the current source folder
        file_list = os.listdir(f)
        
        # Move each file from the source folder to the destination folder
        for name in file_list:
            shutil.move(os.path.join(f, name), os.path.join(dst_folder, name))
        
        # Delete the source folder along with its contents
        deleteFiles(f)

def prevent_data_leakage(dir_txt, dir_img, split: tuple = (0.8, 0.1, 0.1), idx_split_name = -7):
    """
    Prevents data leakage by splitting and moving label and image files into separate train, validation, and test sets.

    Args:
    - dir_txt (str): The directory containing label files.
    - dir_img (str): The directory containing image files.
    - split (tuple): A tuple specifying the split ratios for train, validation, and test sets (default is (0.8, 0.1, 0.1)).
    - idx_split_name (int): The index to split label names for grouping (default is -7).

    Note:
    - This function prevents data leakage by splitting label and image files into separate train, validation, and test sets based on the provided split ratios.
    - It creates separate directories for each set: train, validation, and test, for both label and image files.
    """
    # Prevent directory errors by creating train, validation, and test directories if they don't exist
    if not os.path.exists(f"{dir_txt}_train"): os.makedirs(f"{dir_txt}_train")
    if not os.path.exists(f"{dir_txt}_val"): os.makedirs(f"{dir_txt}_val")
    if not os.path.exists(f"{dir_txt}_test"): os.makedirs(f"{dir_txt}_test")

    if not os.path.exists(f"{dir_img}_train"): os.makedirs(f"{dir_img}_train")
    if not os.path.exists(f"{dir_img}_val"): os.makedirs(f"{dir_img}_val")
    if not os.path.exists(f"{dir_img}_test"): os.makedirs(f"{dir_img}_test")

    # Making a list of label files
    labels = os.listdir(dir_txt)
    labels = [f for f in labels if f.endswith('.txt')]
    
    # Grouping labels based on the provided index
    end_table_lable = []
    while labels:
        temp_leb = labels.pop(0)
        lebels_to_remove = [temp_leb]
        for idx, leb in enumerate(labels):
            if temp_leb[:idx_split_name] in leb:
                leb_pop = labels.pop(idx)
                lebels_to_remove.append(leb_pop)
        end_table_lable.append(lebels_to_remove)
        
    # Calculate sizes for train, validation, and test sets
    cnt_file = len(end_table_lable)
    train_split = round(split[0] * cnt_file)
    val_split = round(split[1] * cnt_file)
    end_table_label_train = end_table_lable[:train_split]
    end_table_label_val = end_table_lable[train_split:train_split+val_split]
    end_table_label_test = end_table_lable[train_split+val_split:]
    
    # Move label files to corresponding train, validation, and test directories
    for many_elements in end_table_label_train:
        for element in many_elements:
            path_to_file = os.path.join(dir_txt, element)
            dst_path = os.path.join(f"{dir_txt}_train", element)
            shutil.move(path_to_file, dst_path)
    for many_elements in end_table_label_val:
        for element in many_elements:
            path_to_file = os.path.join(dir_txt, element)
            dst_path = os.path.join(f"{dir_txt}_val", element)
            shutil.move(path_to_file, dst_path)
    for many_elements in end_table_label_test:
        for element in many_elements:
            path_to_file = os.path.join(dir_txt, element)
            dst_path = os.path.join(f"{dir_txt}_test", element)
            shutil.move(path_to_file, dst_path)
    
     # Move image files to corresponding train, validation, and test directories
    for many_elements in end_table_label_train:
        for element in many_elements:
            path_to_file = os.path.join(dir_img, element.replace('.txt','.png'))
            dst_path = os.path.join(f"{dir_img}_train", element.replace('.txt','.png'))
            shutil.move(path_to_file, dst_path)
    for many_elements in end_table_label_val:
        for element in many_elements:
            path_to_file = os.path.join(dir_img, element.replace('.txt','.png'))
            dst_path = os.path.join(f"{dir_img}_val", element.replace('.txt','.png'))
            shutil.move(path_to_file, dst_path)
    for many_elements in end_table_label_test:
        for element in many_elements:
            path_to_file = os.path.join(dir_img, element.replace('.txt','.png'))
            dst_path = os.path.join(f"{dir_img}_test", element.replace('.txt','.png'))
            shutil.move(path_to_file, dst_path)


def transforms_wh_to_r(labels_dir = '/content/Labels'):
    """
    Transforms bounding box annotations from (x, y, w, h) format to (x, y, r) format.

    Args:
    - labels_dir (str): Directory containing the annotation files.

    Returns:
    - None: Modifies the annotation files in-place.
    """

    # List all annotation files in the directory
    labels = os.listdir(labels_dir)

    # Iterate through each annotation file
    for label_file in labels:
        # Open each annotation file for reading
        with open(os.path.join(labels_dir, label_file),'r') as f:
            # Read each line from the file
            lines = [line.strip() for line in f.readlines()]
            corected_lines = []

            # Process each line in the annotation file
            for line in lines:
                # Parse the line into class index, x, y, w, and h
                class_idx, x, y, w, h = (line.replace(',','')).split(' ')
                class_idx, x, y, w, h = int(class_idx), float(x), float(y), float(w), float(h)

                # Calculate the radius (r) based on width (w) and height (h)
                if w > h:
                    r = w/2
                else:
                    r = h/2

                # Adjust x and y coordinates to ensure the bounding box stays within the image boundaries
                if x - r < 0:
                    rezult = r - w/2
                    if x - rezult < 0:
                        x = 0
                    else:
                        x = x - rezult
                elif x + r > 1:
                    rezult = r - w/2
                    if x + rezult > 1:
                        x = 1
                    else:
                        x = x + rezult

                if y - r < 0:
                    rezult = r - h/2
                    if y - rezult < 0:
                        y = 0
                    else:
                        y = y - rezult
                elif y + r > 1:
                    rezult = r - h/2
                    if y + rezult > 1:
                        y = 1
                    else:
                        y = y + rezult

                # Store the corrected bounding box information
                corected_lines.append(f"{class_idx} {x} {y} {r}")

        # Write the corrected bounding box information back to the file
        with open(os.path.join(labels_dir, label_file),'w') as f:
            for corrected_line in corected_lines:
                f.write(corrected_line + '\n')


import csv

def makeCSV(dir_txt = "/content/Etykiety", dir_png_jpg = "/content/Obrazy", csv_file = "/content/dane.csv"):
    """
    Creates a CSV file containing pairs of image file names and corresponding label file names.

    Args:
    - dir_txt (str): The directory containing label files.
    - dir_png_jpg (str): The directory containing image files.
    - csv_file (str): The path of the CSV file to be created.

    Note:
    - This function searches for image files in the specified directory and checks for corresponding label files.
    - It then creates a CSV file with pairs of image file names and corresponding label file names.
    """
    # Get a list of image files in the specified directory
    image_files = [f for f in os.listdir(dir_png_jpg)]

    # Initialize a list to store pairs of image file names and corresponding label file names
    data = []
    for image_file in image_files:
        # Get the base name of the image file (without extension)
        base_name = os.path.splitext(image_file)[0]

        # Generate the corresponding label file name
        txt_file = f"{base_name}.txt"
        
        # Check if the label file exists
        if os.path.isfile(os.path.join(dir_txt, txt_file)):
            # If the label file exists, append the pair of image file name and label file name to the data list
            data.append([image_file, txt_file])

    # Write the data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

import zipfile
from CircleYolov1Module.yolo_circle_custom_dataset import Datasets

def unpackDatasets(dst_folder: str, transform, *pathsDatasets):
    """
    Unpacks datasets from zip files, preprocesses them, and returns a list of datasets.

    Args:
    - dst_folder (str): The destination folder where datasets will be extracted.
    - transform: Transformation to be applied to the datasets.
    - *pathsDatasets: Variable number of paths to zip files containing datasets.

    Returns:
    - list: A list of processed datasets.

    Note:
    - The function assumes a specific folder structure in the zip files with 'Obrazy' (Images) and 'Etykiety' (Labels) directories.
    - It preprocesses the datasets to avoid data leakage and creates train, validation, and test datasets.
    """
    # Labels for dataset splits
    label_split = ['train','val','test']

    #Unpack datasets and preprocess to avoid data leakage
    for idx, arh_path in enumerate(pathsDatasets):
        with zipfile.ZipFile(arh_path, 'r') as arh_zip:
            arh_zip.extractall(dst_folder)
        os.rename('/content/Obrazy', f'/content/Obrazy{idx}')
        os.rename('/content/Etykiety', f'/content/Etykiety{idx}')
        #Preveting data leakage from imgs dataset cuz some of them was cut on multiple smaller imgs
        prevent_data_leakage(f'/content/Etykiety{idx}', f'/content/Obrazy{idx}')

    #Initialize result list
    result = []
    
    #Concate folders and create datasets
    for label in label_split:
        concateFolder(f'/content/Etykiety_{label}', [f'/content/Etykiety{cnt}_{label}' for cnt in range(len(pathsDatasets))]);
        concateFolder(f'/content/Obrazy_{label}', [f'/content/Obrazy{cnt}_{label}' for cnt in range(len(pathsDatasets))]);
        makeCSV(dir_txt=f'/content/Etykiety_{label}', dir_png_jpg=f'/content/Obrazy_{label}', csv_file=f'/content/dane_{label}.csv')
        result.append(Datasets(csv_file=f'/content/dane_{label}.csv', img_dir=f'/content/Obrazy_{label}', label_dir=f'/content/Etykiety_{label}', transform = transform))
    
    return result
    
# ALL DOWN ARE FOR TEsTING DATASETS:
def labels_to_apples(labels, S=7, C=1):
    apples = []

    for I in range(S):
        for J in range(S):
                if labels[I][J][0] != 0:
                    apples.append([(J, I, labels[I][J][(C+1):(C+4)])])

    for i in range(len(apples)):
      apples[i] = [float((apples[i][0][0]+apples[i][0][2][0])/S),
                  float((apples[i][0][1]+apples[i][0][2][1])/S),
                  float(apples[i][0][2][2])]
    return apples


from numpy.random.mtrand import randint
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def testing_dataset_img(custom_dataset, S= 7, C=1, draw_grid = True):
    n = randint(0,len(custom_dataset) - 1)
    img, labels = custom_dataset[n][0], custom_dataset[n][1]
    #print(labels)

    apples = labels_to_apples(labels = labels, S = S, C=C)
    Ch, W, H = img.shape

    img_pil = Image.fromarray((torch.permute(img.to('cpu'), (1, 2, 0)).numpy() * 255).astype('uint8'));
    draw = ImageDraw.Draw(img_pil);
    if draw_grid:
      for s in range(1,S):
        draw.line((s/S*W, 0 , s/S*W, H), fill='yellow', width = 1)
        draw.line((0, s/S*H , W, s/S*H), fill='yellow', width = 1)
    for x, y, r in apples:
      draw.ellipse((x * W - r*W, y * H - r*W, x * W + r*W, y * H + r*W), outline='blue')
    return img_pil


def testing_dataset_img_matrix(custom_dataset):
    plt.figure(figsize=(16, 16), dpi=300)

    for i in range(1, 10):
      img_pil = testing_dataset_img(custom_dataset);
      plt.subplot(3, 3, i)
      plt.imshow(img_pil)
      plt.axis('off')
      plt.subplots_adjust(hspace=0.01, wspace=0.01)
    plt.show()
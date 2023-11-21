import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'O:\kit_19_dataset\kits19\preprocessed_training_data\images'
mask_path = r'O:\kit_19_dataset\kits19\preprocessed_training_data\masks'
all_images_output_folder = r'O:\kit_19_dataset\kits19\preprocessed_training_data\kidney_dataset\all_images'
all_mask_output_folder = r'O:\kit_19_dataset\kits19\preprocessed_training_data\kidney_dataset\all_masks'

def cotain_all_images(img_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scan_folders = sorted([folder for folder in os.listdir(img_path) if folder.startswith("img")])
    for scan_folder in sorted(scan_folders):
        i = scan_folder.split('_')[1]
        print(i)
        img_folder = os.path.join(img_path, scan_folder)
        img_files = sorted(os.listdir(img_folder))
        for index, mask_file in enumerate(img_files):
            mask_file_path = os.path.join(img_folder, mask_file)
            mask_input_image = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            # print("index: ", index)
            cv2.imwrite(f"{output_dir}/img_{i}_{index}.jpg", mask_input_image)
    print("done for grouping all images")

def cotain_all_masks(img_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scan_folders = sorted([folder for folder in os.listdir(img_path) if folder.startswith("mask")])
    for scan_folder in sorted(scan_folders):
        i = scan_folder.split('_')[1]
        print(i)
        img_folder = os.path.join(img_path, scan_folder)
        img_files = sorted(os.listdir(img_folder))
        for index, mask_file in enumerate(img_files):
            mask_file_path = os.path.join(img_folder, mask_file)
            mask_input_image = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            # print("index: ", index)
            cv2.imwrite(f"{output_dir}/mask_{i}_{index}.jpg", mask_input_image)
    print("done for grouping all masks")


def main():
    start_time = time.time()
    cotain_all_images(image_path, all_images_output_folder)
    cotain_all_masks(mask_path, all_mask_output_folder)
    print(f"Elapsed time: {time.time() - start_time} seconds")

    print("All done !!")

if __name__ == '__main__':
    main()
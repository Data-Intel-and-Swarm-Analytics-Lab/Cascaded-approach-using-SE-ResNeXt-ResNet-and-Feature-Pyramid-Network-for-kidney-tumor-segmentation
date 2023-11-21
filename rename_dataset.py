import os
import time
from convert_kidney_as_background import change_masks_background
from preprocessing_techniques import load_images, load_masks

def extract_img_slice():
    start_time = time.time()
    dataset_path = r"O:\kit_19_dataset\kits19\training_data"
    # dataset_path = r"O:\kit_19_dataset\kits19\slice_with_isssue"
    # path = r"O:\kit_19_dataset\kits19\validation_data"
    scan_folders = sorted([folder for folder in os.listdir(dataset_path) if folder.startswith("case")])
    OUTPUT_IMG_DIR = r"O:\kit_19_dataset\kits19\preprocessed_training_data\images"
    OUTPUT_MASK_DIR = r"O:\kit_19_dataset\kits19\preprocessed_training_data\masks"
    # O:\kit_19_dataset\kits19\preprocessed_validation_data

    for scan_folder in sorted(scan_folders):
        i = scan_folder.split('_')[1]
        scan_folder_path = os.path.join(dataset_path, scan_folder)
        # print(scan_folder_path)
        image_file = os.path.join(scan_folder_path, "imaging.nii.gz")
        mask_file = os.path.join(scan_folder_path, "segmentation.nii.gz")

        img_output_path = f"{OUTPUT_IMG_DIR}\img_{i}"
        if not os.path.exists(img_output_path):
            os.makedirs(img_output_path)
         # load and preprocess images
        load_images(image_file, img_output_path)

        mask_output_path = f"{OUTPUT_MASK_DIR}\mask_{i}"
        if not os.path.exists(mask_output_path):
            os.makedirs(mask_output_path)
        # load_masks(mask_file, mask_output_path)
        change_masks_background(mask_file, mask_output_path)
        print("done with scan number ", i)

    print(f"Elapsed time: {time.time() - start_time} seconds")
    print("done loading up the data")

def main():
    extract_img_slice()

if __name__ == '__main__':
    main()
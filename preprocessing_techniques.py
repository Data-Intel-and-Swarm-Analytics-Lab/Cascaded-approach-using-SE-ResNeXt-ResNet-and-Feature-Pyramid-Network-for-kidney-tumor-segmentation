# loading images

import os
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import io, exposure
import numpy as np
from scipy.ndimage import median_filter
from skimage.transform import resize
from torchvision.io import read_image
from PIL import Image

OUTPUT_IMG_PATH = r"O:\preprocess\scan1\img"
OUTPUT_MASK_PATH = r"O:\preprocess\scan1\mask"
# IMAGE_DIR = r"O:\kit_19_dataset\kits19\data\case_00000\imaging.nii.gz"
# MASK_DIR = r"O:\kit_19_dataset\kits19\data\case_00000\segmentation.nii.gz"

def apply_median_filter(image, window_size):
    filtered_image = median_filter(image, size=window_size)
    return filtered_image

def resize_image(image, target_size):
    resized_image = resize(image, target_size, anti_aliasing=True)
    return resized_image
    
def apply_hu(image_slice, level, window):
    max = level + window/2
    min = level - window/2
    clip_image = np.clip(image_slice, min, max)
#     print("min", min, "max", max, "min in clip", clip_image.min(), "max in clip", clip_image.max())
    return clip_image


# def display_image(image):
#     plt.imshow(image, cmap='gray')
#     plt.axis('on')
#     plt.show()

def display_image(image, preprocessed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('on')
    axes[1].imshow(preprocessed, cmap='gray')
    axes[1].set_title('Median filtered Image')
    axes[1].axis('on')
#     plt.tight_layout()
    plt.show()


# loop though images
def print_preprocessed_all_images(img_path):
    load_images = nib.load(img_path).get_fdata()
    
    for slice_index in range(load_images.shape[0]):
        image_slice = load_images[slice_index, :, :]
        print(f"Processing slice {slice_index + 1}/{load_images.shape[0]}")
        image_filter = apply_median_filter(image_slice, 3)
        display_image(image_slice, image_filter)
        
    print("Done!")

def print_preprecessed_images(img_path):
    load_images = nib.load(img_path).get_fdata()
#     image_slice1 = load_images[load_images.shape[0] // 347, :, :]
    slice_index = 314
    image_slice1 = load_images[slice_index, :, :]
    print(load_images.shape)
#     os.makedirs(output_path, exist_ok=True)
#         image_slice = load_images[i,:,:]
        # print("shape of original image -> ", load_images.shape)
    image_filter = apply_median_filter(image_slice1, 3)
#     hu_image = apply_hu(image_slice1, 135, 330)
#     equalized_image = exposure.equalize_hist(image_slice1)
#     normalized_slice = cv2.normalize(image_slice1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     resized_image = resize_image(image_slice1, (256, 256))
    display_image(image_slice1, image_filter)
    print("done !!")
        # print("resize shape => ", resized_image.shape)
        # display_image(resized_image)
#         output_file = os.path.join(output_path, f"_{i}.png")
#         plt.imsave(output_file, resized_image, cmap='gray') 
#     return resized_image

def load_images(path, output_path):
    load_images = nib.load(path).get_fdata()
    os.makedirs(output_path, exist_ok=True)
    for i in range(load_images.shape[0]):
        image_slice = load_images[i,:,:]
        # print("shape of original image -> ", load_images.shape)
        image_filter = apply_median_filter(image_slice, 3)
        hu_image = apply_hu(image_filter, 135, 330)
        equalized_image = exposure.equalize_hist(hu_image)
        normalized_slice = cv2.normalize(equalized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        resized_image = resize_image(normalized_slice, (256, 256))
        # print("resize shape => ", resized_image.shape)
        # display_image(resized_image)
        output_file = os.path.join(output_path, f"_{i}.png")
        plt.imsave(output_file, resized_image, cmap='gray') 
    return resized_image

def load_masks(path, output_path):
    load_images = nib.load(path).get_fdata()
    for i in range(load_images.shape[0]):
        image_slice = load_images[i,:,:]
        mask_img = resize_image(image_slice, (256, 256))
        # print("resize shape => ", mask_img.shape)
        # display_image(resized_image)
        output_file = os.path.join(output_path, f"_{i}.png")
        plt.imsave(output_file, mask_img, cmap='gray')      
    return mask_img

# print_preprecessed_images(r"case_00000/imaging.nii.gz")
# print_preprecessed_images(r"case_00000/segmentation.nii.gz")
# load_images(IMAGE_DIR, OUTPUT_IMG_PATH)
# load_masks(MASK_DIR, OUTPUT_MASK_PATH)
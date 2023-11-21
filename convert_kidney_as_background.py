import os
import cv2
from matplotlib import pyplot as plt
import nibabel
import numpy as np
from skimage.transform import resize



background_label = 0
kidney_label = 1
kidney_tumor_label = 2
MASK_FOLDER= r"O:\preprocess\scan1\mask"
NEW_BACKGROUND = r"O:\preprocess\remasked"
MASK_DIR = r"O:\kit_19_dataset\kits19\data\case_00000\segmentation.nii.gz"


def change_masks_background(path, output_path):
    load_images = nibabel.load(path).get_fdata()
    for i in range(load_images.shape[0]):
        # image_slice = load_images[i,:,:]
        mask_img = load_images[i,:,:]
        # mask_img = resize_image(image_slice, (256, 256))
        converted_image = np.zeros_like(mask_img, dtype=np.uint8)
        converted_image[mask_img == kidney_label] = background_label
        converted_image[mask_img == kidney_tumor_label] = kidney_tumor_label
        normalized_slice = cv2.normalize(converted_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        resize_img = resize_image(normalized_slice, (256, 256))
        output_file = os.path.join(output_path, f"mask_{i}.png")
        plt.imsave(output_file, resize_img, cmap='gray')      
    return mask_img

def resize_image(image, target_size):
    resized_image = resize(image, target_size, anti_aliasing=True)
    return resized_image

# change_masks_background(MASK_DIR, NEW_BACKGROUND)
# print("done!!")
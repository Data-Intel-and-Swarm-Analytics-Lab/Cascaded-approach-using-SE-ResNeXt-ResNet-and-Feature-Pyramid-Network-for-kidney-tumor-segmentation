from copy import deepcopy
import os
import time
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
# from dice_score import DiceLoss
from load_dataset import train_dataloader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage import io

from metrics import dice_coefficient, get_precision, get_recall, jaccard_index
SAVED_MODEL_PATH = r"C:\Users\HP\Desktop\Project_Implementation\tumor_segmetation\coarse_output_model_weights.pth"
TEST_IMAGE_FOR_PREDICTION =  r"O:\preprocess\scan1\img\preprocessed_img_101.png";
IMAGE_FOLDER_PATH = r"O:\preprocess\kidney_dataset\all_images"
MASK_FOLDER_PATH = r"O:\preprocess\kidney_dataset\all_masks"
SAVE_OUTPUT_FOLDER = r"O:\preprocess\kidney_dataset\segmented";
sigmoid = nn.Sigmoid()
num_output_channels =1
coarse_model = timm.create_model('seresnext26d_32x4d', pretrained=True, num_classes=num_output_channels)
coarse_model.conv1[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
coarse_model.global_pool = nn.Identity()
coarse_model.fc = nn.Conv2d(2048, num_output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

loss_fn = nn.BCELoss() 
optimizer = optim.Adam(coarse_model.parameters(), lr=0.001)
# print(model)
best_value =1
NUMBER_OF_EPOCH =30

def train_model():
    best_value = float('inf')
    for epoch in range(NUMBER_OF_EPOCH):
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        coarse_model.train()   
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = coarse_model(images)
            outputs = torch.nn.functional.interpolate(outputs, size=(masks.shape[2], masks.shape[3]), mode='bilinear', align_corners=False)  # Resize outputs to match the spatial dimensions of masks
            outputs = sigmoid(outputs)
            # print("output shape => ",outputs.shape, "mask shape => ",masks.shape)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            predicted_labels = (outputs > 0.5).cpu().numpy().astype(np.uint8)
            ground_truth = masks.cpu().numpy()
            dice_score = dice_coefficient(predicted_labels, ground_truth)
            iou = jaccard_index(predicted_labels, ground_truth)
            pre_score = get_precision(predicted_labels, ground_truth)
            recal = get_recall(predicted_labels, ground_truth)
            print(f"Batch size {batch_idx+1}/{len(train_dataloader)}, dice_score in train loader: {dice_score:.4f}, loss: {loss:.4f}, iou in train loader: {iou:.4f}, pre_score in train loader: {pre_score:.4f}, recal in train loader: {recal:.4f}")

    # Check for improvement
        if loss < best_value:
            best_value = loss
            # Create a deep copy of the best model's state
            best_model_state = deepcopy(coarse_model.state_dict())
            # Save the best model's state
            torch.save(best_model_state, SAVED_MODEL_PATH)
        if (epoch + 1) % 3 == 0:
            with open('coarse_metrics.txt', 'a') as f:
                f.write(f"Epoch {epoch+1}/{NUMBER_OF_EPOCH}, dice_score: {dice_score:.4f}, loss: {loss:.4f}, iou: {iou:.4f}, pre_score: {pre_score:.4f}, recal: {recal:.4f}\n")
                print(f"Epoch {epoch+1}/{NUMBER_OF_EPOCH} , dice_score: {dice_score:.4f}, iou: {iou:.4f}, pre_score: {pre_score:.4f}, recal: {recal:.4f}")


def create_model():
    best_model = timm.create_model('seresnext26d_32x4d', pretrained=False, num_classes=num_output_channels)
    best_model.conv1[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    best_model.global_pool = nn.Identity()
    best_model.fc = nn.Conv2d(2048, num_output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    return best_model

def load_saved_model(best_model, saved_model_path):
    best_model.load_state_dict(torch.load(saved_model_path))
    best_model.eval()
    return best_model

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return input_tensor

def preprocess_mask(m_path):
    m_image = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (256, 256))
    m_image = m_image.astype(np.float32) / 255.0
    m_tensor = torch.from_numpy(m_image).unsqueeze(0).unsqueeze(0)
    return m_tensor

def perform_inference(best_model, input_tensor):
    with torch.no_grad():
        prediction = best_model(input_tensor)
        prediction = torch.sigmoid(prediction)  
        prediction_np = prediction.squeeze().cpu().numpy()
        resized_prediction = cv2.resize(prediction_np, (256, 256))  
    return resized_prediction

def display_results(input_image, mask_image, segmented_output):
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(input_image[0, 0], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Mask Image")
    plt.imshow(mask_image[0, 0], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Segmented Output")
    plt.imshow(segmented_output, cmap='gray')
    plt.show()


# def load_and_infer_model():
#     load_model = create_model()
#     best_model_state = load_saved_model(load_model, SAVED_MODEL_PATH)
#     input_tensor = preprocess_image(TEST_IMAGE_FOR_PREDICTION)
#     mask_image = preprocess_mask(MASK_PATH_1)
#     prediction = perform_inference(best_model_state, input_tensor)
#     display_results(input_tensor, mask_image, prediction)

def save_segmented_output(image_path, output_folder, segmented_output):
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"segmented_{filename}")
    cv2.imwrite(output_path, (segmented_output * 255).astype(np.uint8))


def load_and_infer_model():
    load_model = create_model()
    best_model_state = load_saved_model(load_model, SAVED_MODEL_PATH)
    for filename in os.listdir(IMAGE_FOLDER_PATH):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(IMAGE_FOLDER_PATH, filename)
            mask_path = os.path.join(MASK_FOLDER_PATH, f"mask_{filename}")
            
            input_tensor = preprocess_image(image_path)
            mask_image = preprocess_mask(mask_path)
            prediction = perform_inference(best_model_state, input_tensor)
            display_results(input_tensor, mask_image, prediction)
            save_segmented_output(image_path, SAVE_OUTPUT_FOLDER, prediction)

# MASK_PATH_1 = r"C:\Users\chest\Downloads\Kidney_tumor_seg\fine_output\masks\mask_00000\mask_341.png"
# MASK_PATH_1 = r"fine_output/kidney_dataset/all_masks/mask_00027_327.jpg"

def main():
    start_time = time.time()
    print("training started")
#     train_model()
    load_and_infer_model()  
    print(f"Elapsed time: {time.time() - start_time} seconds")
    print("Done !! => ")     

if __name__ == "__main__":
    main()

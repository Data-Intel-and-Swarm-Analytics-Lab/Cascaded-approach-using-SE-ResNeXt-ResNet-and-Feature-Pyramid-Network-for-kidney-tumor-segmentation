import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = sorted(os.listdir(os.path.join(data_dir, "all_images")))
        self.mask_files = sorted(os.listdir(os.path.join(data_dir, "all_masks")))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, "all_images", self.image_files[idx])
        mask_path = os.path.join(self.data_dir, "all_masks", self.mask_files[idx])
        
        # image = Image.open(image_path).convert("RGB")
        # mask = Image.open(mask_path).convert("RGB")
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        # print("image new ->", image)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Create a dataset instance
# train_data_path =  r"O:\preprocess\scan1"  
train_data_path =  r"O:\kit_19_dataset\kits19\preprocessed_training_data\kidney_dataset"
# traindir = os.path.join(custom_data_path, )
train_dataset = CustomDataset(data_dir=train_data_path, transform=ToTensor())

# Create a DataLoader
batch_size = 30
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# first_batch = next(iter(train_dataloader))
# images = first_batch[0]

# # Check the type of images in the batch
# for image in images:
#     if image.shape[0] == 1:
#         print("Grayscale Image")
#     elif image.shape[0] == 3:
#         print("RGB Image")
#     else:
#         num_channels = image.shape[0]
#         print(f"Multichannel Image with {num_channels} channels")

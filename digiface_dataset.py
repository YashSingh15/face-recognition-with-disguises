import os
import zipfile
import random
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import requests


class DigiFaceDataset(Dataset):
    def __init__(self, root_dir, train=True, download=False):
        super(DigiFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.train = train

        if download:
            self._download()

        # Image paths and labels
        self.image_paths = []
        self.labels = []
        for subj_id in os.listdir(self.root_dir):
            subject_dir = os.path.join(self.root_dir, subj_id)
            for img_name in os.listdir(subject_dir):
                img_path = os.path.join(subject_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(int(subj_id.split("_")[1]))

        # Augmentations
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(112),
                self._appearance_augmentation,  # haven't implemented yet, check paper for details
                self._warping_augmentation,     # haven't implemented yet, check paper for details
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label

    def _download(self):
        # URLs for dataset parts
        urls = [
            # total 720K images (10K identities, 72 images/identity. Used to train the model in variety of poses)
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip",
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_2000-3999_72_imgs.zip",
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_4000-5999_72_imgs.zip",
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_6000-7999_72_imgs.zip",
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_8000-9999_72_imgs.zip",
            # total 500K images (100K identities, 5 images/identity. Used to help model distinguish among various identities)
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_100000-133332_5_imgs.zip",
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_133333-166665_5_imgs.zip",
            "https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_166666-199998_5_imgs.zip",
        ]

        for url in urls:
            zip_path = os.path.join(self.root_dir, os.path.basename(url))
            response = requests.get(url)
            with open(zip_path, "wb") as file:
                file.write(response.content)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.root_dir)
            os.remove(zip_path)

    def _appearance_augmentation(self, img):
        # appearance augmentation described in the paper, but idk how to implement, so returning image unchanged for now
        return img

    def _warping_augmentation(self, img):
        # warping augmentation described in the paper, but idk how to implement, so returning unchaned image for now
        return img

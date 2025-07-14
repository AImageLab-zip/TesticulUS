
from enum import Enum, auto
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys
from PIL import Image
from torchvision.transforms.v2 import functional as F
import utils

# mean, std of real distribution 0.2074
# 0.2074, 0.1460

# mean, std of fake distribution
# 0.1928, 0.2024


def crop_image_split(img, idx):
    roi = [130, 185 + 300 * idx, 300, 300]
    return F.crop(img, roi[0], roi[1], roi[2], roi[3])


def custom_collate(batch, use_augmentation: bool = False, return_ids: bool = False, mode: str = "single_view", norm_mean=0.5, norm_std=0.25,
                   img_ch=1):
    images, labels = zip(*batch)  # Unzip into separate tuples
    transformed_images = []
    if use_augmentation:
        aug = utils.FineTuningDataAug_v2()

    std_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256)),
        v2.Normalize(mean=[norm_mean], std=[norm_std]),
        v2.Grayscale(num_output_channels=img_ch),
    ])
    img_ids = []

    if mode == "single_view":
        for path in images:
            img_ids.append(str(Path(path).name).replace(".bmp", ""))
            path, idx = path.split('#')

            idx = int(idx)
            img = Image.open(path).convert("L")  # Open as grayscale
            img = crop_image_split(img, idx)
            if use_augmentation:
                img = aug(img)
            transformed_images.append(std_transform(img))
    elif mode == "coupled_view":
        for path in images:
            img_ids.append(str(Path(path).name).replace(".bmp", "#0"))
            img_ids.append(str(Path(path).name).replace(".bmp", "#1"))
            img = Image.open(path).convert("L")
            img_lhs = crop_image_split(img, 0)
            img_rhs = crop_image_split(img, 1)
            if use_augmentation:
                img_lhs = aug(img_lhs)
                img_rhs = aug(img_rhs)

            transformed_images.append(std_transform(img_lhs))
            transformed_images.append(std_transform(img_rhs))

        dup_labels = []
        for l in labels:
            dup_labels.append(l)
            dup_labels.append(l)
        labels = dup_labels

    if return_ids:
        return torch.stack(transformed_images), torch.tensor(labels), img_ids
    else:
        return torch.stack(transformed_images), torch.tensor(labels)


class LabelTypes(Enum):
    OMO_DISOMO = auto()
    BINARY_FUNCTIONAL = auto()
    FUNCTIONAL = auto()


class USDataset(Dataset):
    def __init__(self, root_dir, mode: str = "single_view", label_type: str = LabelTypes.OMO_DISOMO):
        super().__init__()
        self.root_dir = Path(root_dir)
        assert self.root_dir.is_dir(), "USDataset: root_dir is not valid"
        self.imgs_dir = self.root_dir.joinpath("data")
        assert self.imgs_dir.is_dir(), "USDataset: data dir not found"
        self.metadata_path = self.root_dir.joinpath("new_meta.xlsx")
        assert self.metadata_path.is_file(), "USDataset: metadata file not found"
        self.metadata = pd.read_excel(str(self.metadata_path))
        self.items = []
        self.dx_label_idx = self.metadata.columns.get_loc(
            "omogeneo (0), disomogeneo (1)")
        self.sx_label_idx = self.metadata.columns.get_loc(
            "omogeneo (0), disomogeneo (1).1")
        self.functional_binary_cls_idx = self.metadata.columns.get_loc(
            "CLASSIFICAZIONE FUNZIONANTE (0) NON FUNZIONANTE (1)")
        self.functional_cls_idx = self.metadata.columns.get_loc(
            "CLASSIFICAZIONE NORMOFUNZIONANTE (0) NON FUNZIONANTE SEMINALE (1) NON FUNZIONANTE ORMONALE (2) NON FUNZIONANTE SIA ORMONALE CHE SEMINALE (3)")
        self.label_type = label_type

        for i, id in enumerate(self.metadata.ID):
            testic_dx = self.imgs_dir.joinpath(f"ID{id:02}_1.bmp")
            testic_sx = self.imgs_dir.joinpath(f"ID{id:02}_2.bmp")
            if not testic_dx.is_file():
                print(f"USDataset: {str(testic_dx)} was not found")
            else:
                if self.label_type == LabelTypes.OMO_DISOMO:
                    idx = self.dx_label_idx
                elif self.label_type == LabelTypes.BINARY_FUNCTIONAL:
                    idx = self.functional_binary_cls_idx
                else:
                    idx = self.functional_cls_idx
                label = self.metadata.iloc[i, idx]
                if label != 0 and label != 1 and label != 2 and label != 3:
                    continue
                if mode == "single_view":
                    self.items.append((str(testic_dx) + "#0", label))
                    self.items.append((str(testic_dx) + "#1", label))
                elif mode == "coupled_view":
                    self.items.append((str(testic_dx), label))

            if not testic_sx.is_file():
                print(f"USDataset: {str(testic_sx)} was not found")
            else:
                if self.label_type == LabelTypes.OMO_DISOMO:
                    idx = self.dx_label_idx
                elif self.label_type == LabelTypes.BINARY_FUNCTIONAL:
                    idx = self.functional_binary_cls_idx
                else:
                    idx = self.functional_cls_idx
                label = self.metadata.iloc[i, idx]
                if label != 0 and label != 1 and label != 2 and label != 3:
                    continue
                if mode == "single_view":
                    self.items.append((str(testic_sx) + "#0", label))
                    self.items.append((str(testic_sx) + "#1", label))
                elif mode == "coupled_view":
                    self.items.append((str(testic_sx), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, label = self.items[index]
        return img_path, label

    def get_class_distribution(self):
        """
        Returns the distribution of classes in the dataset.

        Returns:
            dict: A dictionary with class labels as keys and their counts as values.
        """
        class_distribution = {}
        for _, label in self.items:
            if label in class_distribution:
                class_distribution[label] += 1
            else:
                class_distribution[label] = 1

        # Calculate percentages
        total = len(self.items)
        distribution_with_percentages = {}
        for label, count in class_distribution.items():
            percentage = (count / total) * 100
            distribution_with_percentages[label] = {
                "count": count,
                "percentage": f"{percentage:.2f}%"
            }

        return distribution_with_percentages

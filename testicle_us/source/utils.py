from argparse import Namespace
import sys, torch, os, yaml
from pathlib import Path
from prettytable import PrettyTable
from torchvision.transforms import v2
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import wandb, random, net
from PIL import Image
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image, ImageDraw
import cv2
import numpy as np
from apex.parallel.LARC import LARC
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def load_yaml(inFile=None) -> dict:
    inFile = sys.argv[1] if inFile is None else inFile

    with open(inFile, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    display_configs(config)

    assert Path(config['checkpoint_dir']).is_dir(), "Please provide a valid directory to save checkpoints in."
    assert Path(config['dataset_dir']).is_dir(), "Please provide a valid directory to load dataset."

    return config


def display_configs(configs):
    t = PrettyTable(["Name", "Value"])
    t.align = "r"
    for key, value in configs.items():
        t.add_row([key, value])
    print(t, flush=True)


def display_args(args):
    t = PrettyTable(["Name", "Value"])
    t.align = "r"
    for key, value in args.__dict__.items():
        t.add_row([key, value])
    print(t, flush=True)


def load_device(device_type: str):
    """Loads and returns the appropriate computing device based on the configuration.

    This function checks the "device" key in the given config dictionary.
    If "gpu" is specified, it ensures that CUDA is available and selects the first GPU.
    Otherwise, it defaults to the CPU.

    Args:
        config (dict): A dictionary containing a "device" key with values "gpu" or "cpu".

    Raises:
        AssertionError: If "gpu" is requested but CUDA is not available.

    Returns:
        str: The device identifier, either "cuda:0" for GPU or "cpu".
    """
    if device_type == "gpu":
        assert torch.cuda.is_available(), "Notebook is not configured properly!"
        device = torch.device("cuda:0")
        print(
            "Training network on {}({})".format(torch.cuda.get_device_name(device=device), torch.cuda.device_count())
        )

    else:
        device = torch.device("cpu")
    return device


def load_opt(args: Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt = None
    if args.opt == "adam":
        if isinstance(model, net.ClassificationWrapper):
            # opt used up until hico
            opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr_back, "weight_decay": 1e-5},
                                    {"params": model.cls_head.parameters(), "lr": args.lr_head, "weight_decay": 1e-5}])

            # opt = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr, "weight_decay": 1e-5},])

        else:
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    elif args.opt == "lars":
        base_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        opt = LARC(optimizer=base_optimizer,
                   trust_coefficient=0.001,
                   clip=False)

    if opt is not None:
        return opt
    else:
        raise ValueError("Invalid optimizer")


def load_sched(args: Namespace, opt: torch.optim.Optimizer):
    if args.sched == None:
        return None
    elif args.sched == "poly":
        if args.epochs is not None:

            sched = torch.optim.lr_scheduler.PolynomialLR(
                opt.optim if args.opt == "lars" else opt, total_iters=args.epochs)
        else:
            print(f"The sched was initialized using {args.steps} epochs")
            sched = torch.optim.lr_scheduler.PolynomialLR(
                opt.optim if args.opt == "lars" else opt, total_iters=args.steps)

        return sched
    else:
        raise ValueError("Invalid scheduler")


class DataAugmentation(object):
    def __init__(self, norm_mean=0.5, norm_std=0.25, img_ch=1):
        self.transform = v2.Compose([
            v2.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
            # RandomChoiceTransform(transforms_list=[
            # TemplateMatchingTransform(),
            # RandomShadowSquare(shadow_size=96, p=0.5),
            # ], p=0.5),

            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            # v2.ColorJitter(brightness=0.8, contrast=0.8),
            v2.ColorJitter(0.8, 0.8, 0.8, 0.2),
            # v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            # v2.RandomApply([v2.GaussianBlur(kernel_size=3),], p=0.2),
            v2.Grayscale(num_output_channels=img_ch),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[norm_mean], std=[norm_std]),
        ])

    def __call__(self, x):
        return self.transform(x)


class FineTuningDataAug(object):
    def __init__(self):
        self.transform = v2.Compose([
            v2.RandomResizedCrop(256, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation((0, 180)),
            v2.Grayscale()
        ])

    def __call__(self, x):
        return self.transform(x)


class FineTuningDataAug_v2(object):
    def __init__(self):
        #original
        # self.transform = v2.Compose([
        #     v2.RandomApply([v2.Lambda(lambda x: remove_left_band(x, pixels=25)),], p=0.5),
        #     v2.RandomApply([v2.Lambda(lambda x: remove_right_band(x, pixels=25)),], p=0.5),
        #     v2.RandomHorizontalFlip(),
        #     v2.RandomApply([RandomMarker(f"{str(Path(__file__).parent)}/O_marker.png", n=8),], p=.8),
        #     v2.RandomApply([RandomMarker(f"{str(Path(__file__).parent)}/plus_marker.png", n=5),], p=.8),
        #     v2.RandomApply([RandomDottedLine(),], p=.5),
        #     v2.RandomApply([RandomDottedLine(),], p=.5),
        #     v2.RandomApply([RandomDottedLine(),], p=.5),
        #     v2.RandomApply([TranslateDown(60),], p=0.8),
        #     # Random90Rotation(),
        #     # v2.RandomRotation((0, 180)),
        #     v2.Grayscale()
        # ])        
        self.transform = v2.Compose([
            v2.RandomApply([v2.Lambda(lambda x: remove_left_band(x, pixels=25)),], p=0.25),
            v2.RandomApply([v2.Lambda(lambda x: remove_right_band(x, pixels=25)),], p=0.25),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([RandomMarker(f"{str(Path(__file__).parent)}/O_marker.png", n=5),], p=.3),
            v2.RandomApply([RandomMarker(f"{str(Path(__file__).parent)}/plus_marker.png", n=5),], p=.3),
            v2.RandomApply([RandomDottedLine(),], p=.3),
            v2.RandomApply([RandomDottedLine(),], p=.3),
            v2.RandomApply([RandomDottedLine(),], p=.3),
            v2.RandomApply([TranslateDown(60),], p=0.2),
            v2.Grayscale()
        ])

    def __call__(self, x):
        return self.transform(x)


class FineTuningDataAug_v3(object):
    def __init__(self):
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomApply([TranslateDown(60),], p=0.8),
            Random90Rotation(),
            v2.Grayscale()
        ])

    def __call__(self, x):
        return self.transform(x)


class USCLloss(nn.Module):
    def __init__(self, net: nn.Module, device: torch.device, temp=0.5, lambda_=0.2, use_sup_loss=True, use_unsup_loss=True):
        super(USCLloss, self).__init__()
        self.net = net
        self.device = device
        self.net = net.to(self.device)
        self.temp = temp
        self.lambda_ = lambda_
        self.use_sup_loss = use_sup_loss
        self.use_unsup_loss = use_unsup_loss

    def info_nce(self, i, j, cos_sim: torch.tensor):
        loss = cos_sim[i, j]
        loss = loss / cos_sim[i, :].sum()
        return -torch.log(loss + 1e-8)

    def forward(self, x, xi, xj, y):
        x, y = x.to(self.device), y.to(self.device)
        xi, xj = xi.to(self.device), xj.to(self.device)
        proj_i, digits_i = self.net(xi)
        proj_j, digits_j = self.net(xj)
        proj = torch.cat((proj_i, proj_j))
        norm = proj.norm(dim=-1, keepdim=True)
        cos_sim = (proj @ proj.T) / (norm @ norm.T)
        cos_sim = (cos_sim / self.temp).exp()
        cos_sim = cos_sim * (1.0 - torch.eye(proj.shape[0], device=self.device))
        unsup_loss = 0.0
        for i in range(x.shape[0]):
            unsup_loss += self.info_nce(i, i + x.shape[0], cos_sim)
            unsup_loss += self.info_nce(i + x.shape[0], i, cos_sim)
        unsup_loss = unsup_loss / proj.shape[0]

        sup_loss = torch.nn.functional.cross_entropy(digits_i, y) + torch.nn.functional.cross_entropy(digits_j, y)
        wandb.log({
            "train/sup_loss": sup_loss.item(),
            "train/unsup_loss": unsup_loss.item()
        })
        if self.use_sup_loss and self.use_unsup_loss:
            return unsup_loss + self.lambda_ * sup_loss
        elif self.use_sup_loss:
            return sup_loss
        elif self.use_unsup_loss:
            return unsup_loss


def coupled_loss(logits: torch.tensor, temperature=1):
    batch_size, _ = logits.shape
    loss = 0
    for i in range(batch_size // 2):
        # Get logits for paired samples
        first_view_logits = logits[i * 2]
        second_view_logits = logits[i * 2 + 1]

        # Forward direction: first_view -> second_view
        second_view_probs = F.softmax(second_view_logits / temperature, dim=-1)
        first_view_log_probs = F.log_softmax(first_view_logits / temperature, dim=-1)
        loss_forward = F.kl_div(
            first_view_log_probs,
            second_view_probs,
            reduction='sum'
        )

        # Backward direction: second_view -> first_view
        first_view_probs = F.softmax(first_view_logits / temperature, dim=-1)
        second_view_log_probs = F.log_softmax(second_view_logits / temperature, dim=-1)
        loss_backward = F.kl_div(
            second_view_log_probs,
            first_view_probs,
            reduction='sum'
        )

        # Average the bidirectional losses and scale by temperature squared
        pair_loss = (loss_forward + loss_backward) * (temperature ** 2)
        loss += pair_loss

    return loss / (batch_size // 2)


def compute_binary_metrics(y_true, y_pred):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    acc = accuracy_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def compute_multiclass_metrics(y_true, y_pred, average='macro', return_per_class=False):
    """
    Compute metrics for multiclass classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like  
        Predicted labels
    average : str, default='macro'
        Averaging method for precision, recall, f1. Options:
        - 'macro': Calculate metrics for each label, return unweighted mean
        - 'micro': Calculate metrics globally by counting total TP, FP, FN
        - 'weighted': Calculate metrics for each label, return weighted mean by support
    return_per_class : bool, default=False
        Whether to return per-class metrics
    
    Returns:
    --------
    dict : Dictionary containing computed metrics
    """
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate averaged metrics
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Calculate per-class metrics if requested
    per_class_metrics = {}
    if return_per_class:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {
            f"precision_class_{cls}": prec for cls, prec in zip(classes, precision_per_class)
        }
        per_class_metrics.update({
            f"recall_class_{cls}": rec for cls, rec in zip(classes, recall_per_class)
        })
        per_class_metrics.update({
            f"f1_class_{cls}": f1_cls for cls, f1_cls in zip(classes, f1_per_class)
        })
    
    # Basic metrics dictionary
    metrics = {
        f"precision_{average}": precision,
        f"recall_{average}": recall,
        f"f1_{average}": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "num_classes": len(classes),
        "classes": classes.tolist()
    }
    
    # Add per-class metrics if requested
    if return_per_class:
        metrics.update(per_class_metrics)
    
    return metrics


def log_metrics(metrics, phase="train", fold_n=None, epoch=None):
    log_dict = {}

    # Add prefix to metrics
    for key, value in metrics.items():
        if key != "confusion_matrix":
            log_dict[f"{phase}/{key}"] = value

    # Add confusion matrix plot
    if "confusion_matrix" not in metrics:
        # class_names = ["omo", "disomo"]
        log_dict[f"{phase}/confusion_matrix"] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=metrics.get("y_true", None),
            preds=metrics.get("y_pred", None)
            # class_names=class_names
        )

    # Add fold and epoch info
    if fold_n is not None:
        log_dict["fold"] = fold_n
    if epoch is not None:
        log_dict["epoch"] = epoch

    # Log to wandb
    wandb.log(log_dict)


def update_dictionary(main_dict, new_dict):
    for key, values in new_dict.items():
        mean = sum(values) / len(values)
        if key in main_dict.keys():
            main_dict[key].append(mean)
        else:
            # Create a new entry with a copy of the list
            main_dict[key] = [mean]

    return main_dict


def mean_dict(main_dict: dict, window_dim: int, shift=0):
    mean_dict = {}
    mean_shifted_dict = {}

    for k, v in main_dict.items():
        if len(v) < window_dim + shift:
            continue
        # Using slicing for faster access to window elements
        recent_window = v[-window_dim:]
        shifted_window = v[-(window_dim + shift):-shift]

        # Calculate means directly from slices
        mean_dict[k] = sum(recent_window) / window_dim
        mean_shifted_dict[k] = sum(shifted_window) / window_dim

    return mean_dict, mean_shifted_dict


def remove_left_band(img, pixels=25):
    """Remove a band of specified width from the left side of the image."""
    if isinstance(img, Image.Image):
        # For PIL images
        width, height = img.size
        return img.crop((pixels, 0, width, height))
    elif isinstance(img, torch.Tensor):
        # For tensor images (C x H x W)
        return img[:, :, pixels:]
    else:
        raise TypeError("Input should be a PIL Image or a torch Tensor")


def remove_right_band(img, pixels=25):
    """Remove a band of specified width from the right side of the image."""
    if isinstance(img, Image.Image):
        # For PIL images
        width, height = img.size
        return img.crop((0, 0, width - pixels, height))
    elif isinstance(img, torch.Tensor):
        # For tensor images (C x H x W)
        return img[:, :, :-pixels]
    else:
        raise TypeError("Input should be a PIL Image or a torch Tensor")


class RandomMarker(object):
    """
    Custom transform that pastes a hardcoded image with alpha channel onto input images.

    This transform is compatible with torchvision transform pipelines.
    """

    def __init__(self, overlay_path, position=None, n=1):
        """
        Args:
            overlay_path (str): Path to the RGBA image to overlay
            position (tuple, optional): (x, y) position to paste the overlay.
                                        If None, centers the overlay.
            n (int), max range of times to apply the marker
        """
        self.overlay = Image.open(overlay_path).convert("RGBA")
        self.position = position
        # Choose a random number between 1 and n (inclusive) for the number of overlays
        self.n = random.randint(1, n)

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to overlay onto.
                                       If Tensor, will be converted to PIL Image.

        Returns:
            PIL Image: Image with overlay applied
        """
        # Convert tensor to PIL Image if needed
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3):
                # Handle CHW format used by PyTorch
                img = v2.functional.to_pil_image(img)
            else:
                raise TypeError("Input tensor should be 3D with shape [C, H, W]")

        # Ensure input image is RGB
        if img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Determine position if not specified
        positions = []
        for _ in range(random.choice(torch.arange(self.n))):
            if self.position is None:
                positions.append((
                    random.choice(torch.arange(img.width)),
                    random.choice(torch.arange(img.height))
                ))
            else:
                positions.append(self.position)

        # Create output image with alpha
        result = Image.new("RGBA", img.size)

        # Fill with background image (fully opaque)
        result.paste(img.convert("RGBA"), (0, 0))

        # Paste overlay using its alpha channel
        for position in positions:
            result.paste(self.overlay, position, self.overlay)

        # Convert back to RGB (removing alpha)
        result = result.convert("RGB")

        return result


class TranslateDown(torch.nn.Module):
    """
    Transform that translates an image downward by a specified number of pixels.
    """

    def __init__(self, pixels=20):
        super().__init__()
        self.pixels = pixels

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to translate downward.

        Returns:
            PIL Image or Tensor: Translated image.
        """
        return v2.functional.affine(
            img,
            angle=0,
            translate=(0, random.choice(torch.arange(self.pixels))),  # (horizontal, vertical) translation
            scale=1.0,
            shear=0
        )


class Random90Rotation(torch.nn.Module):
    """Randomly rotate square image by 0, 90, 180, or 270 degrees without interpolation."""

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to rotate.

        Returns:
            PIL Image: Rotated image.
        """
        # Choose rotation: 0, 1, 2 or 3 times 90 degrees
        k = random.randint(0, 3)

        # Skip if k=0 (no rotation)
        if k == 0:
            return img

        # For PIL images, use transpose and/or flip operations
        # These methods don't use interpolation for 90-degree rotations
        if k == 1:  # 90 degrees
            return img.transpose(Image.ROTATE_90)
        elif k == 2:  # 180 degrees
            return img.transpose(Image.ROTATE_180)
        else:  # 270 degrees (or -90)
            return img.transpose(Image.ROTATE_270)


class RandomDottedLine(torch.nn.Module):
    def __init__(self, position=None):
        self.overlay = Image.open(f"{str(Path(__file__).parent)}/plus_marker.png").convert("RGBA")
        self.position = position
        self.n = 2
        self.dot_marker = Image.open(f"{str(Path(__file__).parent)}/dot_marker.png").convert("RGBA")
        self.dot_spacing = 30

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to overlay onto.
                                       If Tensor, will be converted to PIL Image.

        Returns:
            PIL Image: Image with overlay applied
        """
        # Convert tensor to PIL Image if needed
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3):
                # Handle CHW format used by PyTorch
                img = v2.functional.to_pil_image(img)
            else:
                raise TypeError("Input tensor should be 3D with shape [C, H, W]")

        # Ensure input image is RGB
        if img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Determine position if not specified
        positions = []
        for _ in range(self.n):
            if self.position is None:
                positions.append((
                    random.choice(torch.arange(img.width)),
                    random.choice(torch.arange(img.height))
                ))
            else:
                positions.append(self.position)

        # Create output image with alpha
        result = Image.new("RGBA", img.size)

        # Fill with background image (fully opaque)
        result.paste(img.convert("RGBA"), (0, 0))

        # # Paste overlay using its alpha channel
        # for position in positions:
        #     result.paste(self.overlay, position, self.overlay)

        center1 = (
            positions[0][0] + self.overlay.width // 2,
            positions[0][1] + self.overlay.height // 2
        )
        center2 = (
            positions[1][0] + self.overlay.width // 2,
            positions[1][1] + self.overlay.height // 2
        )
        # Calculate vector from overlay1 to overlay2
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]

        # Calculate distance between centers
        distance = ((dx)**2 + (dy)**2)**0.5

        # Place dot at 30 pixels from the first overlay center towards the second
        if distance > 0:  # Avoid division by zero
            # Unit vector in direction from center1 to center2
            dx_unit = dx / distance
            dy_unit = dy / distance

            # Calculate number of dots needed (leaving space at start and end)
            start_offset = 30  # Start dots 30px from first overlay
            end_offset = 30  # End dots 30px before second overlay

            effective_distance = distance - start_offset - end_offset
            num_dots = max(1, int(effective_distance / self.dot_spacing))

            # Place dots along the line
            for i in range(num_dots):
                # Calculate position for this dot
                dot_distance = start_offset + i * \
                    (effective_distance / max(1, num_dots - 1)) if num_dots > 1 else start_offset

                dot_x = int(center1[0] + dot_distance * dx_unit)
                dot_y = int(center1[1] + dot_distance * dy_unit)

                # Calculate top-left position for pasting the dot marker
                dot_pos = (
                    dot_x - self.dot_marker.width // 2,
                    dot_y - self.dot_marker.height // 2
                )

                # Paste the dot marker
                result.paste(self.dot_marker, dot_pos, self.dot_marker)

        # Paste the overlays after calculating the dot position (to ensure overlays are on top)
        for pos in positions:
            result.paste(self.overlay, pos, self.overlay)

        result = result.convert("RGB")

        return result


def remove_bad_ids(dataset, ids_path: str):
    ids_list = torch.load(ids_path)["excluded_ids"]
    excluded_paths = []

    # Generate the paths to exclude
    for id in ids_list:
        lhs_id = id[:-2]
        rhs_id = id[-2:]
        img_path = f"{dataset.imgs_dir}/{lhs_id}.bmp{rhs_id}"
        excluded_paths.append(img_path)

    # Create a new filtered list of items
    filtered_items = []
    for item_path, label in dataset.items:
        if item_path not in excluded_paths:
            filtered_items.append((item_path, label))

    # Update the dataset's items list
    removed_count = len(dataset.items) - len(filtered_items)
    dataset.items = filtered_items

    # Optional: print how many items were removed
    print(f"Removed {removed_count} items from the dataset.")

    return dataset


def flip_bad_ids(dataset, ids_path: str):
    ids_list = torch.load(ids_path)["excluded_ids"]
    excluded_paths = []

    # Generate the paths to exclude
    for id in ids_list:
        lhs_id = id[:-2]
        rhs_id = id[-2:]
        img_path = f"{dataset.imgs_dir}/{lhs_id}.bmp{rhs_id}"
        excluded_paths.append(img_path)

    # Create a new filtered list of items
    filtered_items = []
    for item_path, label in dataset.items:
        if item_path not in excluded_paths:
            filtered_items.append((item_path, label))
        else:
            filtered_items.append((item_path, 1 - label))

    # Update the dataset's items list
    removed_count = len(dataset.items) - len(filtered_items)
    dataset.items = filtered_items

    # Optional: print how many items were removed
    print(f"Removed {removed_count} items from the dataset.")

    return dataset


class RandomShadowSquare(object):
    def __init__(self, shadow_size, p=0.5):
        self.shadow_size = shadow_size
        self.p = p

    def __call__(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(0)
        B, nC, H, W = x.shape
        assert H >= self.shadow_size and W >= self.shadow_size, "Shadow size can't be bigger than the image"

        x_start = torch.randint(low=0, high=W - self.shadow_size, size=(B,))
        y_start = torch.randint(low=0, high=H - self.shadow_size, size=(B,))

        for i in range(B):
            x[i, :, y_start[i]:y_start[i] + self.shadow_size, x_start[i]:x_start[i] + self.shadow_size] = 0

        return x


# class TemplateMatchingTransform:
#     def __init__(self, template_paths: Tuple[str, ...], thresholds: Tuple[float, ...]):
#         self.template_paths = template_paths
#         self.template_thresholds = thresholds

#         self.templates = []
#         self.template_sizes = []

#         for template_path in self.template_paths:
#             template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

#             if template is None:
#                 self.templates.append(None)
#                 self.template_sizes.append((0, 0))
#                 continue

#             template = cv2.resize(template, (0, 0), fx=0.85, fy=0.85)

#             self.templates.append(template)
#             self.template_sizes.append((template.shape[1], template.shape[0]))  # (width, height)

#     def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
#         input_is_tensor = isinstance(img, torch.Tensor)

#         # Handle batch processing
#         if input_is_tensor and img.dim() == 4:  # Batch of tensors [B, C, H, W]
#             batch_size = img.shape[0]
#             orig_dtype = img.dtype
#             orig_device = img.device

#             # Process each image in the batch
#             results = []
#             for i in range(batch_size):
#                 single_img = img[i]
#                 # Keep original tensor properties for debugging
#                 channels = single_img.shape[0]

#                 # Convert to PIL for processing
#                 pil_img = v2.ToPILImage()(single_img)

#                 # Apply template matching - will convert to grayscale internally if needed
#                 result_img = self._apply_template_matching(pil_img)

#                 # Convert back to tensor with same number of channels as input
#                 if channels == 1:
#                     # If input was grayscale, ensure output is too
#                     if result_img.mode != 'L':
#                         result_img = result_img.convert('L')
#                     result_tensor = v2.ToTensor()(result_img)
#                 else:
#                     # If input was RGB/RGBA, ensure output matches
#                     if result_img.mode != pil_img.mode:
#                         result_img = result_img.convert(pil_img.mode)
#                     result_tensor = v2.ToTensor()(result_img)

#                 # Ensure tensor has the same device and dtype as original
#                 result_tensor = result_tensor.to(device=orig_device, dtype=orig_dtype)
#                 results.append(result_tensor)

#             # Stack the results back into a batch
#             return torch.stack(results, dim=0)

#         # Original single image processing
#         if input_is_tensor:
#             orig_dtype = img.dtype
#             orig_device = img.device
#             channels = img.shape[0] if img.dim() == 3 else 1

#             # Convert to PIL
#             pil_img = v2.ToPILImage()(img)
#             original_mode = pil_img.mode

#             # Process the image
#             result_img = self._apply_template_matching(pil_img)

#             # Ensure result has same mode as input
#             if result_img.mode != original_mode:
#                 result_img = result_img.convert(original_mode)

#             # Convert back to tensor and preserve properties
#             result_tensor = v2.ToTensor()(result_img).to(device=orig_device, dtype=orig_dtype)
#             return result_tensor
#         else:
#             # Keep track of original mode
#             original_mode = img.mode

#             # Apply template matching
#             result_img = self._apply_template_matching(img)

#             # Ensure result has same mode as input
#             if result_img.mode != original_mode:
#                 result_img = result_img.convert(original_mode)

#             return result_img

#     def _apply_template_matching(self, image: Image.Image) -> Image.Image:
#         # Store original mode
#         original_mode = image.mode

#         # Convert to grayscale for template matching if needed
#         if image.mode != 'L':
#             gray_image = image.convert('L')
#         else:
#             gray_image = image

#         # Convert PIL image to OpenCV format
#         target_cv = np.array(gray_image)

#         # Create a copy of the original image for drawing (preserve color channels)
#         result_image = image.copy()
#         draw = ImageDraw.Draw(result_image)

#         for idx, (template, threshold) in enumerate(zip(self.templates, self.template_thresholds)):
#             if template is None:
#                 continue

#             w, h = self.template_sizes[idx]  # Width, height

#             # Perform template matching
#             result = cv2.matchTemplate(target_cv, template, cv2.TM_CCOEFF_NORMED)

#             # Find locations where the match exceeds the threshold
#             locations = np.where(result >= threshold)
#             coordinates = list(zip(*locations[::-1]))  # Convert to x,y format

#             # Draw black rectangles over matches
#             for pt in coordinates:
#                 draw.rectangle([pt[0], pt[1], pt[0] + w, pt[1] + h], fill="black")

#         return result_image


class TemplateMatchingTransform:
    def __init__(self, template_paths: Tuple[str, ...], thresholds: Tuple[float, ...]):
        self.template_paths = template_paths
        self.template_thresholds = thresholds

        self.templates = []
        self.template_sizes = []

        for template_path in self.template_paths:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            if template is None:
                self.templates.append(None)
                self.template_sizes.append((0, 0))
                continue

            template = cv2.resize(template, (0, 0), fx=0.85, fy=0.85)

            self.templates.append(template)
            self.template_sizes.append((template.shape[1], template.shape[0]))  # (width, height)

    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        input_is_tensor = isinstance(img, torch.Tensor)

        # Handle batch processing
        if input_is_tensor and img.dim() == 4:  # Batch of tensors [B, C, H, W]
            batch_size = img.shape[0]
            orig_dtype = img.dtype
            orig_device = img.device

            # Process each image in the batch
            results = []
            for i in range(batch_size):
                single_img = img[i]

                # Convert tensor [C,H,W] in range [0,255] to PIL
                if single_img.dtype == torch.uint8:
                    # Already in [0,255]
                    numpy_img = single_img.cpu().numpy().transpose(1, 2, 0)
                    if numpy_img.shape[2] == 1:  # Grayscale
                        numpy_img = numpy_img.squeeze(2)
                    pil_img = Image.fromarray(numpy_img)
                else:
                    # Scale [0,1] to [0,255] if needed
                    numpy_img = (single_img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                    if numpy_img.shape[2] == 1:  # Grayscale
                        numpy_img = numpy_img.squeeze(2)
                    pil_img = Image.fromarray(numpy_img)

                # Remember original format
                original_mode = pil_img.mode

                # Apply template matching
                result_img = self._apply_template_matching(pil_img)

                # Ensure result has same mode as input
                if result_img.mode != original_mode:
                    result_img = result_img.convert(original_mode)

                # Convert back to tensor in the [0,255] range
                numpy_result = np.array(result_img)
                if len(numpy_result.shape) == 2:  # Grayscale
                    numpy_result = np.expand_dims(numpy_result, axis=2)
                # Ensure correct channel order [H,W,C] -> [C,H,W]
                numpy_result = numpy_result.transpose(2, 0, 1)

                # Create tensor and maintain original dtype/device
                result_tensor = torch.from_numpy(numpy_result).to(device=orig_device, dtype=orig_dtype)
                results.append(result_tensor)

            # Stack the results back into a batch
            return torch.stack(results, dim=0)

        # Original single image processing
        if input_is_tensor:
            orig_dtype = img.dtype
            orig_device = img.device

            # Convert tensor to PIL while preserving [0,255] range
            if img.dtype == torch.uint8:
                # Direct conversion for uint8 tensors (already [0,255])
                if img.dim() == 3:  # [C,H,W]
                    numpy_img = img.cpu().numpy().transpose(1, 2, 0)
                    if numpy_img.shape[2] == 1:  # Grayscale
                        numpy_img = numpy_img.squeeze(2)
                    pil_img = Image.fromarray(numpy_img)
                else:  # [H,W]
                    pil_img = Image.fromarray(img.cpu().numpy())
            else:
                # Scale float tensors [0,1] -> [0,255]
                if img.dim() == 3:  # [C,H,W]
                    numpy_img = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                    if numpy_img.shape[2] == 1:  # Grayscale
                        numpy_img = numpy_img.squeeze(2)
                    pil_img = Image.fromarray(numpy_img)
                else:  # [H,W]
                    pil_img = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))

            # Remember original format
            original_mode = pil_img.mode

            # Apply template matching
            result_img = self._apply_template_matching(pil_img)

            # Ensure result has same mode as input
            if result_img.mode != original_mode:
                result_img = result_img.convert(original_mode)

            # Convert back to tensor in the [0,255] range
            numpy_result = np.array(result_img)
            if len(numpy_result.shape) == 2:  # Grayscale
                if img.dim() == 3:  # Original was [C,H,W]
                    numpy_result = np.expand_dims(numpy_result, axis=2)
                    numpy_result = numpy_result.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
            elif len(numpy_result.shape) == 3:  # RGB/RGBA
                numpy_result = numpy_result.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]

            # Create tensor and maintain original device
            result_tensor = torch.from_numpy(numpy_result).to(device=orig_device, dtype=orig_dtype)
            return result_tensor
        else:
            # PIL image case - directly apply template matching
            original_mode = img.mode
            result_img = self._apply_template_matching(img)

            # Ensure result has same mode as input
            if result_img.mode != original_mode:
                result_img = result_img.convert(original_mode)

            return result_img

    def _apply_template_matching(self, image: Image.Image) -> Image.Image:
        # Store original mode
        original_mode = image.mode

        # Convert to grayscale for template matching if needed
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image

        # Convert PIL image to OpenCV format
        target_cv = np.array(gray_image)

        # Create a copy of the original image for drawing (preserve color channels)
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        for idx, (template, threshold) in enumerate(zip(self.templates, self.template_thresholds)):
            if template is None:
                continue

            w, h = self.template_sizes[idx]  # Width, height

            # Perform template matching
            result = cv2.matchTemplate(target_cv, template, cv2.TM_CCOEFF_NORMED)

            # Find locations where the match exceeds the threshold
            locations = np.where(result >= threshold)
            coordinates = list(zip(*locations[::-1]))  # Convert to x,y format

            # Draw black rectangles over matches
            for pt in coordinates:
                draw.rectangle([pt[0], pt[1], pt[0] + w, pt[1] + h], fill="black")

        return result_image


class RandomChoiceTransform(object):
    """
    Apply one randomly selected transformation from a list with specified probability.
    """

    def __init__(self, transforms_list, p=0.5):
        """
        Args:
            transforms_list (list): List of transforms to choose from
            p (float): Probability of applying any transformation
        """
        self.transforms_list = transforms_list
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        if random.random() < self.p:
            # Randomly select one transformation
            transform = random.choice(self.transforms_list)
            return transform(img)
        return img

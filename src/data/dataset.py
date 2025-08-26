"""Dataset classes for autonomous vehicle perception."""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AutonomousVehicleDataset(Dataset):
    """Dataset for autonomous vehicle object detection.
    
    Supports COCO format annotations and common AV datasets.
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        transforms: Optional[Callable] = None,
        image_size: Tuple[int, int] = (640, 640),
        filter_classes: Optional[List[str]] = None
    ):
        """Initialize dataset.
        
        Args:
            images_dir: Directory containing images.
            annotations_file: Path to COCO format annotations.
            transforms: Albumentations transforms.
            image_size: Target image size (width, height).
            filter_classes: List of class names to include. If None, include all.
        """
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.transforms = transforms or self._get_default_transforms()
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create class mappings
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Filter classes if specified
        if filter_classes:
            self.class_filter = set(filter_classes)
            self.filtered_categories = {
                cat_id: name for cat_id, name in self.categories.items()
                if name in self.class_filter
            }
        else:
            self.filtered_categories = self.categories
            self.class_filter = None
        
        # Create image ID to annotations mapping
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            
            # Filter by class if specified
            if self.class_filter is None or \
               self.categories[ann['category_id']] in self.class_filter:
                self.image_annotations[image_id].append(ann)
        
        # Filter images that have annotations
        self.images = [
            img for img in self.coco_data['images']
            if img['id'] in self.image_annotations and 
               len(self.image_annotations[img['id']]) > 0
        ]
        
    def _get_default_transforms(self) -> A.Compose:
        """Get default augmentation transforms."""
        return A.Compose([
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels']
        ))
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.
        
        Returns:
            Dictionary with 'image', 'boxes', 'labels' keys.
        """
        image_info = self.images[idx]
        image_path = self.images_dir / image_info['file_name']
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        annotations = self.image_annotations[image_info['id']]
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, w, h])
            
            # Map category ID to filtered class index
            category_name = self.categories[ann['category_id']]
            if category_name in self.filtered_categories.values():
                # Find index in filtered categories
                filtered_cat_list = list(self.filtered_categories.values())
                label_idx = filtered_cat_list.index(category_name)
                labels.append(label_idx)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(image_info['id'])
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return list(self.filtered_categories.values())
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, List]:
        """Custom collate function for DataLoader."""
        images = [item['image'] for item in batch]
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        
        return {
            'images': torch.stack(images),
            'boxes': boxes,
            'labels': labels,
            'image_ids': torch.stack(image_ids)
        }


def create_data_loaders(
    train_images_dir: str,
    train_annotations: str,
    val_images_dir: str,
    val_annotations: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640)
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.
    
    Args:
        train_images_dir: Training images directory.
        train_annotations: Training annotations file.
        val_images_dir: Validation images directory.
        val_annotations: Validation annotations file.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        image_size: Target image size.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Training transforms (with augmentation)
    train_transforms = A.Compose([
        A.Resize(height=image_size[1], width=image_size[0]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.RandomRotate90(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))
    
    # Validation transforms (no augmentation)
    val_transforms = A.Compose([
        A.Resize(height=image_size[1], width=image_size[0]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels']
    ))
    
    # AV-relevant classes
    av_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']
    
    # Create datasets
    train_dataset = AutonomousVehicleDataset(
        images_dir=train_images_dir,
        annotations_file=train_annotations,
        transforms=train_transforms,
        image_size=image_size,
        filter_classes=av_classes
    )
    
    val_dataset = AutonomousVehicleDataset(
        images_dir=val_images_dir,
        annotations_file=val_annotations,
        transforms=val_transforms,
        image_size=image_size,
        filter_classes=av_classes
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
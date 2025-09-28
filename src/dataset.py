# library imports
from torch.utils.data import Dataset
import torch
import cv2
import albumentations as A
from typing import Tuple, Optional
from enum import StrEnum
from pathlib import Path
from enum import StrEnum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

class DatasetMode(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"

class SolarPanelDataset(Dataset):
    """
    Dataset class for Open Solar Panel Data Madagascar in YOLO format.
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: DatasetMode,
        seed: int,
        img_size: int = 640,  # YOLO standard
        transform: Optional[A.BasicTransform] = None,
        return_format: str = "dict"  # "dict" or "tensor"
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root_dir: Path to dataset root directory containing images/ and annotations/
            mode: Dataset split - train, test or val
            seed: Random seed for reproducibility
            img_size: Input image size (should be multiple of 32 for YOLO)
            transform: Optional albumentations transforms
            return_format: "dict" returns {'boxes': tensor, 'labels': tensor}, 
                         "tensor" returns single tensor [class, x, y, w, h]
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.seed = seed
        self.img_size = img_size
        self.return_format = return_format

        # Validate img_size is multiple of 32 for YOLO
        if img_size % 32 != 0:
            raise ValueError(f"img_size must be multiple of 32 for YOLO, got {img_size}")

        # Set up paths
        self.image_dir = self.root_dir / "images" / self.mode
        self.annotation_file = self.root_dir / f"{mode}_images.json"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Load and process annotations
        self._load_annotations()
        
        # Set up transforms
        self.transform = transform or self._get_default_transform()

    def _load_annotations(self) -> None:
        """Load and process annotations from JSON file"""
        with open(self.annotation_file, 'r') as f:
            import json
            raw_annotations = json.load(f)
        
        # Process annotations to ensure consistent format
        processed_annotations = {}
        valid_count = 0
        
        for image_id, bboxes in raw_annotations.items():
            # Ensure bboxes is a list
            if not isinstance(bboxes, list):
                continue
                
            # Filter out None values and validate bbox format
            valid_bboxes = []
            for bbox in bboxes:
                if (bbox is not None and 
                    isinstance(bbox, dict) and 
                    all(key in bbox for key in ['class', 'x_center', 'y_center', 'width', 'height'])):
                    valid_bboxes.append(bbox)
            
            if valid_bboxes:  # Only keep images with valid bboxes
                processed_annotations[image_id] = valid_bboxes
                valid_count += 1
        
        if not processed_annotations:
            raise ValueError(f"No valid annotations found in {self.annotation_file}")
        
        self.annotations = processed_annotations
        self.image_ids = list(self.annotations.keys())
        
        print(f"Loaded {valid_count} images with valid annotations for {self.mode} split")

    def _get_default_transform(self) -> A.Compose:
        """Get default transform based on dataset mode"""
        if self.mode == DatasetMode.TRAIN:
            return A.Compose(
                [
                    A.LongestMaxSize(max_size=self.img_size),  # Use dynamic img_size
                    A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT),
                    A.RandomCrop(height=self.img_size, width=self.img_size),  # Use dynamic img_size
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GaussNoise(p=0.3),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Normalize(),
                    A.ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format='yolo', 
                    label_fields=['class_labels'],
                    min_visibility=0.3,
                    min_area=0.01
                ),
                seed=self.seed,
            )
        else:
            return A.Compose(
                [
                    A.LongestMaxSize(max_size=self.img_size),  # Use dynamic img_size
                    A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT),
                    A.CenterCrop(height=self.img_size, width=self.img_size),  # Use dynamic img_size
                    A.Normalize(),
                    A.ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format='yolo', 
                    label_fields=['class_labels'],
                    min_visibility=0.3,
                    min_area=0.01
                ),
                seed=self.seed,
            )

    def __len__(self) -> int:
        """Returns the total number of images"""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        Get an image and its bounding boxes.

        Args:
            idx: Index of the image

        Returns:
            Tuple containing:
                - Image tensor (C, H, W)
                - Targets in specified format:
                  * If return_format="dict": {'boxes': tensor(N, 4), 'labels': tensor(N,)}
                  * If return_format="tensor": tensor(N, 5) [class, x_center, y_center, width, height]
        """
        # Get image ID and load image
        image_id = self.image_ids[idx]
        img_path = self.image_dir / f"{image_id}.jpg"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes for this image
        bboxes_data = self.annotations[image_id]
        
        # Extract bounding boxes and labels
        if bboxes_data:
            bboxes = [[
                bbox['x_center'],
                bbox['y_center'], 
                bbox['width'],
                bbox['height']
            ] for bbox in bboxes_data]
            
            labels = [bbox['class'] for bbox in bboxes_data]
            
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        # Apply transforms
        if self.transform:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=labels.tolist()
                )
                
                image = transformed['image']
                bboxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
                
            except Exception as e:
                print(f"Transform failed for image {image_id}: {e}")
                # Fallback: just normalize and convert to tensor
                image = A.Normalize()(image=image)['image']
                image = A.ToTensorV2()(image=image)['image']

        # Convert to tensors and return in specified format
        if self.return_format == "dict":
            targets = {
                'boxes': torch.tensor(bboxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        else:  # tensor format
            if len(bboxes) > 0:
                targets = torch.cat([
                    torch.tensor(labels, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(bboxes, dtype=torch.float32)
                ], dim=1)
            else:
                targets = torch.zeros((0, 5), dtype=torch.float32)

        return image, targets

    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for an image"""
        image_id = self.image_ids[idx]
        return {
            'image_id': image_id,
            'num_boxes': len(self.annotations[image_id]),
            'image_path': str(self.image_dir / f"{image_id}.jpg")
        }


# Custom collate function for DataLoader when using dict format
def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per image
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return torch.stack(images), targets


def create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0):
    """
    Create DataLoader with proper error handling
    
    Args:
        dataset: SolarPanelDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (0 = no multiprocessing)
    
    Returns:
        DataLoader instance
    """
    try:
        # First try with multiprocessing
        if num_workers > 0:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            
            # Test the dataloader
            _ = next(iter(dataloader))
            print(f"DataLoader created successfully with {num_workers} workers")
            return dataloader
            
    except Exception as e:
        print(f"Failed to create DataLoader with {num_workers} workers: {e}")
        print("Falling back to single-threaded DataLoader...")
    
    # Fallback to single-threaded
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # No multiprocessing
        pin_memory=False
    )
    
    print("DataLoader created successfully with single threading")
    return dataloader
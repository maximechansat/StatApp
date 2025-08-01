from torch.utils.data import Dataset
import torch
import pandas as pd
import pathlib
import cv2
import albumentations as A
from typing import Tuple, Optional
from enum import StrEnum


class DatasetMode(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class SolarPanelDataset(Dataset):
    """
    Represents the dataset from the Open Solar Panel Data Madagascar.
    """

    def __init__(
        self,
        json_path: pathlib.Path,
        img_path: pathlib.Path,
        mode: DatasetMode,
        seed: int,
        transform: Optional[A.BasicTransform] = None,
    ) -> None:
        """
        Arguments:
            json_path (pathlib.Path): path of the directory containing the json files with the
                bounding boxes and the images.
                The json files are expected to be named "{mode}_images.json" where mode is one
                of "train", "test" or "val".
            img_path (pathlib.Path): path of the directory containing the images.
            mode (DatasetMode): the mode of the dataset, one of "train", "test" or "val".
            transform (Optional[A.BasicTransform]): albumentations transform to apply to the images.
                If None, a default transform will be applied based on the mode.
            seed (int): random seed use for allocating images to sets and data augmentations.
        """

        self.json_path = json_path
        self.img_path = img_path
        self.seed = seed
        self.mode = mode
        self.images_with_bounding_boxes = pd.read_json(json_path / f"{mode}_images.json")
        self.transform = transform
        if not self.transform:
            if mode == "train":
                self.transform = A.Compose(
                    [
                        A.SmallestMaxSize(max_size_hw=(500, 500)),
                        A.CropNonEmptyMaskIfExists(height=500, width=500),
                        A.RandomCrop(height=299, width=299),
                        A.GaussNoise(),
                        A.D4(),
                        A.Normalize(),
                        A.ToTensorV2(),
                    ],
                    seed=self.seed,
                )

            if mode in ["test", "val"]:
                self.transform = A.Compose(
                    [
                        A.SmallestMaxSize(max_size_hw=(500, 500)),
                        A.CropNonEmptyMaskIfExists(height=500, width=500),
                        A.CenterCrop(height=299, width=299),
                        A.Normalize(),
                        A.ToTensorV2(),
                    ],
                    seed=self.seed,
                )

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """

        return len(self.images_with_bounding_boxes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the desired image bounding box.
        """
        img_name = self.images_with_bounding_boxes.iloc[idx]["img_name"]
        img = cv2.imread(self.img_path / (img_name + ".jpg"))
        assert img is not None, f"Image {img_name} not found in {self.img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bounding_boxes = torch.Tensor(
            self.images_with_bounding_boxes.iloc[idx]["list_bounding_boxes"]
        )
        if self.transform:
            transformed_objects = self.transform(
                image=img, bboxes=bounding_boxes, format="pascal_voc"
            )
            return (transformed_objects["image"], transformed_objects["bboxes"])
        return (torch.Tensor(img), bounding_boxes)

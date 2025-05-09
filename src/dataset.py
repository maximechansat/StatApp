from torch.utils.data import Dataset
import torch
import pandas as pd
import pathlib
import matplotlib
import numpy as np
import cv2
import albumentations as A
from typing import Tuple, List


class SolarPanelDataset(Dataset):
    """
    Represents the dataset from the Open Solar Panel Data Madagascar.
    """

    def __init__(
        self,
        img_path: pathlib.Path,
        xlsx_path: pathlib.Path,
        task: str,
        type: str,
        mode: str,
        probs: List[float],
        seed: int,
        threshold: float = 0,
    ) -> None:
        """
        Arguments:
            img_path (pathlib.Path): path of the directory containing the images.
            xlsx_path (pathlib.Path): path of the xlsx file containing the metadatas.
            task (str): either "cls" (for classification) or "seg" (for segmentation).
            Along with the images, in classiciation mode, the Dataset will return labels and in
            segmentation masks.
            type (str): either "boil" (for boiler), "pan" (for solar panel) or "all".
            The dataset will only return the image containing elements of the type specified.
            threshold (float): proportion of pixels covered by the mask so that the image is
            considered positive. Only useful in classification mode.
            probs (List[float]): proportion of images allocated respectively to the training set,
            the test set and the validation set. May not sum to 1.
            seed (int): random seed use for allocating images to sets and data augmentations.

        Note: the class will mostly be instanciated with mode="seg", type="pan" and mode="cls", type="all".
        """

        self.img_path = img_path
        self.xlsx_path = xlsx_path
        self.task = task
        self.type = type
        self.dfs = pd.read_excel(xlsx_path, sheet_name=[0, 1, 2])
        self.threshold = threshold
        self.labels = {}
        self.seed = seed

        if mode == "train":
            self.mode = 0
        if mode == "test":
            self.mode = 1
        if mode == "val":
            self.mode = 2

        self.probs = probs

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

        self.compute_labels()

    def compute_labels(self) -> None:
        """
        Internal function used to compute the data associated with the panels (which are mask or labels depending on the mode).
        In classification mode, a label of 1 denotes the presence of a solar panel and 0 its absence.
        In segmentation mode, a white pixel denotes the presence of a solar panel and a black one its absence.
        """

        if self.type == "pan":
            self.dfs[0] = self.dfs[0][
                self.dfs[0]["type1"].isin(("pan", "mix", "solar_park"))
            ]
            self.dfs[1] = self.dfs[1][
                self.dfs[1]["img_name"].isin(self.dfs[0]["img_name"])
            ]

        if self.type == "boil":
            self.dfs[0] = self.dfs[0][self.dfs[0]["type1"].isin(("boil", "mix"))]
            self.dfs[1] = self.dfs[1][
                self.dfs[1]["img_name"].isin(self.dfs[0]["img_name"])
            ]

        extended_probs = torch.Tensor(self.probs + [1 - sum(self.probs)])
        previous_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        samples = torch.distributions.categorical.Categorical(extended_probs).sample(
            (len(self.dfs[0]),)
        )
        torch.set_rng_state(previous_state)
        mask = (np.array(samples) == self.mode)

        self.dfs[0] = self.dfs[0].loc[mask]
        self.dfs[1] = self.dfs[1][self.dfs[1]["img_name"].isin(self.dfs[0]["img_name"])]

        solar_elt_names = set(
            self.dfs[1][self.dfs[1]["type1"] == "pan"]["elt_name"]
        )

        for elt_name, value in self.dfs[2].groupby("elt_name"):
            if elt_name in solar_elt_names:
                img_name = int(elt_name.split("z")[0])
                if img_name not in self.labels:
                    self.labels[img_name] = []
                self.labels[img_name].append([*zip(value["lat"], value["long"])])

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """

        return len(self.dfs[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the desired image along with the relevant data (mask or label).
        """

        img_number = self.dfs[0].iloc[idx]["number"]
        img_name = str(self.dfs[0].iloc[idx]["img_name"])
        img = cv2.imread(self.img_path / (img_name + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for vertices in self.labels.get(img_number, []):
            x = np.linspace(0, img.shape[0] - 1, img.shape[0])
            y = np.linspace(0, img.shape[1] - 1, img.shape[1])
            xv, yv = np.meshgrid(x, y)
            points = np.vstack((xv.ravel(), yv.ravel())).T

            polygon_path = matplotlib.path.Path(vertices)
            submask = (
                polygon_path.contains_points(points)
                .reshape(img.shape[1], img.shape[0])
                .T
            )
            mask = np.maximum(mask, submask)
        augmented = self.transform(image=img, mask=mask)
        img, mask = augmented["image"], augmented["mask"]

        if self.task == "seg":
            return img, mask

        return img, torch.sum(mask) / torch.prod(torch.Tensor(mask.shape)) > self.threshold

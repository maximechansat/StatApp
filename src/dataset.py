from torch.utils.data import Dataset
import torch
import pandas as pd
import pathlib
import matplotlib
import numpy as np
import cv2
import albumentations as A
from typing import Tuple
import os


class SolarPanelDataset(Dataset):
    """
    Represents the dataset from the Open Solar Panel Data Madagascar.
    """

    def __init__(
        self,
        img_path: pathlib.Path,
        xlsx_path: pathlib.Path,
        mode: str,
        type: str,
        train: bool,
        p_test: float,
        seed: int,
    ) -> None:
        """
        Args:
            img_path (pathlib.Path): path of the directory containing the images.
            xlsx_path (pathlib.Path): path of the xlsx file containing the metadatas.
            mode (str): either "cls" (for classification) or "seg" (for segmentation).
            Along with the images, in classiciation mode, the Dataset will return labels and in segmentation masks.
            type (str): either "boil" (for boiler), "pan" (for solar panel) or "all".
            The dataset will only return the image containing elements of the type specified.
            p (float): proportion of images allocated to the test set

        Note: the class will mostly be instanciated with mode="seg", type="pan" and mode="cls", type="all".
        """

        self.img_path = img_path
        self.xlsx_path = xlsx_path
        self.mode = mode
        self.type = type
        self.dfs = pd.read_excel(xlsx_path, sheet_name=[0, 1, 2])
        self.labels = {}
        self.seed = seed
        self.train = train
        self.p = p_test
        self.transform = A.Compose(
            [
                A.SmallestMaxSize(max_size_hw=(500, 500)),
                A.CropNonEmptyMaskIfExists(height=500, width=500),
                A.RandomCrop(height=299, width=299),
                A.Normalize(),
                A.ToTensorV2(),
            ],
            seed=self.seed,
        )

        if self.train:
            self.transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size_hw=(500, 500)),
                    A.CropNonEmptyMaskIfExists(height=500, width=500),
                    A.RandomCrop(height=299, width=299),
                    A.D4(),
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

        # Checking if all the masks are well defined
        def is_float(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        invalid_rows = self.dfs[2][
            ~self.dfs[2]["edge_rank"].apply(is_float)
            | ~self.dfs[2]["long"].apply(is_float)
            | ~self.dfs[2]["lat"].apply(is_float)
        ]

        invalid_elt_names = invalid_rows["elt_name"].unique()

        self.dfs[2] = self.dfs[2][~self.dfs[2]["elt_name"].isin(invalid_elt_names)]
        self.dfs[1] = self.dfs[1][~self.dfs[1]["elt_name"].isin(invalid_elt_names)]

        # Checking if the image names in both DataFrame and disk match
        img_names_df = self.dfs[0]["img_name"].astype(str)
        img_names_disk = {f.split(".")[0] for f in os.listdir(self.img_path)}
        valid_img_names = set(img_names_df) & img_names_disk
        unique_img_names = img_names_df.value_counts()[lambda x: x == 1].index
        final_img_names = valid_img_names & set(unique_img_names)
        self.dfs[0] = self.dfs[0][img_names_df.isin(final_img_names)].copy()

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

        self.labels = {}

        rng = np.random.default_rng(seed=self.seed)
        mask = rng.binomial(1, self.p, len(self.dfs[0])) >= 1
        if self.train:
            mask = np.logical_not(mask)
        self.dfs[0] = self.dfs[0].loc[mask]
        self.dfs[1] = self.dfs[1][self.dfs[1]["img_name"].isin(self.dfs[0]["img_name"])]

        if self.mode == "seg":
            solar_elt_names = set(
                self.dfs[1][self.dfs[1]["type1"] == "pan"]["elt_name"]
            )
            for elt_name, value in self.dfs[2].groupby("elt_name"):
                if elt_name in solar_elt_names:
                    img_name = int(elt_name.split("z")[0])
                    if img_name not in self.labels:
                        self.labels[img_name] = []
                    self.labels[img_name].append([*zip(value["lat"], value["long"])])

        if self.mode == "cls":
            solar_elt_names = set(
                self.dfs[1][self.dfs[1]["type1"] == "pan"]["elt_name"]
            )
            for elt_name, value in self.dfs[2].groupby("elt_name"):
                if elt_name in solar_elt_names:
                    img_name = int(elt_name.split("z")[0])
                    self.labels[img_name] = 1

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

        if self.mode == "seg":
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
            transformed = self.transform(image=img, mask=mask)
            return transformed["image"], transformed["mask"]

        if self.mode == "cls":
            return self.transform(image=img)["image"], self.labels.get(img_number, 0)

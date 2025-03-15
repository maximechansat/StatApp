from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2
import pandas as pd
import torchvision
import pathlib
import matplotlib
import numpy as np
from typing import Tuple


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
        transform: v2.Transform = None,
    ) -> None:
        """
        Args:
            img_path (pathlib.Path): path of the directory containing the images.
            xlsx_path (pathlib.Path): path of the xlsx file containing the metadatas.
            mode (str): either "cls" (for classification) or "seg" (for segmentation).
            Along with the images, in classiciation mode, the Dataset will return labels and in segmentation masks.
            type (str): either "boil" (for boiler), "pan" (for solar panel) or "all".
            The dataset will only return the image containing elements of the type specified.


        Note: the class will mostly be instanciated with mode="seg", type="pan" and mode="cls", type="all".
        """

        self.img_path = img_path
        self.xlsx_path = xlsx_path
        self.transform = transform
        self.mode = mode
        self.type = type
        self.dfs = pd.read_excel(xlsx_path, sheet_name=[0, 1, 2])
        self.labels = {}

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

        self.labels = {}

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
        try:
            img = torchvision.io.decode_image(self.img_path / (img_name + ".JPG"))
        except RuntimeError:
            img = torchvision.io.decode_image(self.img_path / (img_name + ".jpg"))
        if self.mode == "seg":
            mask = torch.zeros((1, img.size(1), img.size(2)))

            for vertices in self.labels.get(img_number, []):
                x = np.linspace(0, img.size(1) - 1, img.size(1))
                y = np.linspace(0, img.size(2) - 1, img.size(2))
                xv, yv = np.meshgrid(x, y)
                points = np.vstack((xv.ravel(), yv.ravel())).T

                polygon_path = matplotlib.path.Path(vertices)
                submask = torch.Tensor(
                    polygon_path.contains_points(points).reshape(
                        (img.size(2), img.size(1))
                    )
                ).T
                mask = torch.logical_or(submask, mask)

            mask = mask.float()

            if self.transform:
                img, mask = self.transform(img, mask)

            return img, mask

        if self.mode == "cls":
            if self.transform:
                return self.transform(img), self.labels.get(img_number, 0)
            return img, self.labels.get(img_number, 0)

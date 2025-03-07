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

    def __init__(self, img_path: pathlib.Path, xlsx_path: pathlib.Path, transform: v2.Transform = None) -> None:
        self.img_path = img_path
        self.xlsx_path = xlsx_path
        self.transform = transform
        self.dfs = pd.read_excel(xlsx_path, sheet_name=[0, 1, 2])

        self.compute_vertices()

    def compute_vertices(self) -> None:
        """
        Internal function used to compute the vertices defining the masks of the solar panels.
        """

        self.vertices = {}
        solar_elt_names = set(self.dfs[1][self.dfs[1]["type1"] == "pan"]["elt_name"])

        for elt_name, value in self.dfs[2].groupby("elt_name"):
            if elt_name in solar_elt_names:
                img_name = int(elt_name.split("z")[0])
                if img_name not in self.vertices:
                    self.vertices[img_name] = []
                self.vertices[img_name].append([*zip(value["lat"], value["long"] )])

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """

        return len(self.dfs[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the desired image, along with the mask delimiting the potential solar panels (the boilers are ignored).
        """

        img_number = self.dfs[0].loc[idx]["number"]
        img_name = self.dfs[0].loc[idx]["img_name"]
        img_filename = img_name + ".JPG"
        img = torchvision.io.decode_image(self.img_path / img_filename)
        mask = torch.zeros((1, img.size(1), img.size(2)))

        for vertices in self.vertices.get(img_number, []):
            x = np.linspace(0, img.size(1) - 1, img.size(1))
            y = np.linspace(0, img.size(2) - 1, img.size(2))
            xv, yv = np.meshgrid(x, y)
            points = np.vstack((xv.ravel(), yv.ravel())).T

            polygon_path = matplotlib.path.Path(vertices)
            submask = torch.Tensor(polygon_path.contains_points(points).reshape((img.size(2), img.size(1)))).T
            mask = torch.logical_or(submask, mask)

        mask = mask.float()

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

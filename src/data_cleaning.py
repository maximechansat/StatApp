import pandas as pd
import numpy as np
import os
import pathlib
import random

p_train, p_test, p_val = 0.8, 0.1, 0.1
xlsx_path = pathlib.Path("../../data/solar_panel_data_madagascar.xlsx")
output_path = pathlib.Path("../../data/solar_panel_dataset")
image_path = pathlib.Path("../../data/img")
seed = 56

# Set seed for reproducibility
random.seed(seed)
np.random.seed(seed)

# 1. Load data
print("Loading and filtering dataset...")
images, elements, coordinates = pd.read_excel(xlsx_path, sheet_name=[0, 1, 2]).values()

coordinates = coordinates[pd.to_numeric(coordinates["lat"], errors="coerce").notnull()]
coordinates = coordinates[pd.to_numeric(coordinates["long"], errors="coerce").notnull()]
coordinates["lat"] = coordinates["lat"].astype(float)
coordinates["long"] = coordinates["long"].astype(float)

elements = elements[elements["type1"] == "pan"]
elements = elements[elements["elt_name"].isin(coordinates["elt_name"])]

images = images[images["img_origin"] == "D"]
images = images[images["type1"] != "boil"]
images = images[images["img_name"].isin(elements["img_name"])]

# ==========================================
# NEW: ROBUST CITY PREPROCESSING
# ==========================================
# 1. Fill NaNs and ensure string type
images["city"] = images["city"].fillna("Unknown").astype(str)

# 2. Remove trailing numbers and the spaces before them (e.g., "Fanambana 140" -> "Fanambana")
images["city"] = images["city"].str.replace(r"\s*\d+$", "", regex=True)

# 3. Strip whitespace and enforce Title Case (merges "tulear", "Tulear", and "Tulear ")
images["city"] = images["city"].str.strip().str.title()
print(f"Cleaned unique cities count: {images['city'].nunique()}")
# ==========================================

# 2. Bounding Box Calculations
grouped_coordinates = coordinates.groupby("elt_name").agg({"long": list, "lat": list})
bounding_boxes = grouped_coordinates.apply(
    lambda row: [
        (min(row["long"]) + max(row["long"])) / 2,
        (min(row["lat"]) + max(row["lat"])) / 2,
        max(row["long"]) - min(row["long"]),
        max(row["lat"]) - min(row["lat"]),
    ],
    axis=1,
)

grouped_bounding_boxes = (
    bounding_boxes.groupby("elt_name").agg(bounding_boxes=list).reset_index()
)
elements_with_grouped_bounding_boxes = pd.merge(
    elements, grouped_bounding_boxes, on="elt_name", how="left"
)
grouped_elements_with_grouped_bounding_boxes = (
    elements_with_grouped_bounding_boxes.groupby("img_name")["bounding_boxes"]
    .agg(list_bounding_boxes="sum")
    .reset_index()
)

images_with_list_bounding_boxes = pd.merge(
    images, grouped_elements_with_grouped_bounding_boxes, on="img_name", how="left"
)

# Keep the cleaned 'city' column
images_with_list_bounding_boxes = images_with_list_bounding_boxes[
    ["img_name", "list_bounding_boxes", "width_pixel", "height_pixel", "city"]
]

# 3. Perform City-Based Group Split
print("Executing City-Based split...")
city_counts = images_with_list_bounding_boxes["city"].value_counts().reset_index()
city_counts.columns = ["city", "img_count"]

# Shuffle cities to ensure randomness in assignment
city_counts = city_counts.sample(frac=1, random_state=seed).reset_index(drop=True)

N_total = len(images_with_list_bounding_boxes)
target_train = int(p_train * N_total)
target_val = int(p_val * N_total)

train_cities, val_cities, test_cities = [], [], []
train_count, val_count, test_count = 0, 0, 0

# Greedily assign cities
for _, row in city_counts.iterrows():
    c, count = row["city"], row["img_count"]
    if train_count + count <= target_train or train_count < target_train * 0.9:
        train_cities.append(c)
        train_count += count
    elif val_count + count <= target_val or val_count < target_val * 0.9:
        val_cities.append(c)
        val_count += count
    else:
        test_cities.append(c)
        test_count += count

print(
    f"Final Split Totals -> Train: {train_count}, Val: {val_count}, Test: {test_count}"
)

# 4. Map the DataFrames based on assigned cities
df_images = (
    images_with_list_bounding_boxes[
        images_with_list_bounding_boxes["city"].isin(train_cities)
    ].reset_index(drop=True),
    images_with_list_bounding_boxes[
        images_with_list_bounding_boxes["city"].isin(test_cities)
    ].reset_index(drop=True),
    images_with_list_bounding_boxes[
        images_with_list_bounding_boxes["city"].isin(val_cities)
    ].reset_index(drop=True),
)

modes = ["train", "test", "val"]

# 5. Write out the YAML and text files
os.makedirs(output_path, exist_ok=True)
with open(output_path / "solar_panel_dataset.yaml", "w") as f:
    f.write(f"path: {output_path.resolve()}\n")
    for k in range(3):
        f.write(f"{modes[k]}: images/{modes[k]}\n")
    f.write("\nnames:\n  0: solar_panel")

print("Writing label files and copying images...")
for k in range(3):
    df = df_images[k]
    mode = modes[k]
    os.makedirs(output_path / "images" / mode, exist_ok=True)
    os.makedirs(output_path / "labels" / mode, exist_ok=True)

    for _, image in df.iterrows():
        label_file_path = output_path / "labels" / mode / f"{image.img_name}.txt"
        with open(label_file_path, "w") as f:
            for bounding_box in image.list_bounding_boxes:
                x, w = (
                    bounding_box[0] / image.width_pixel,
                    bounding_box[2] / image.width_pixel,
                )
                y, h = (
                    bounding_box[1] / image.height_pixel,
                    bounding_box[3] / image.height_pixel,
                )
                f.write(f"0 {x} {y} {w} {h}\n")

        # Copy instead of rename to preserve original dataset
        import shutil

        source_img = image_path / f"{image.img_name}.jpg"
        dest_img = output_path / "images" / mode / f"{image.img_name}.jpg"
        if source_img.exists():
            shutil.copy2(source_img, dest_img)

print("Dataset successfully created!")

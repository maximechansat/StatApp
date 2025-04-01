from dataset import SolarPanelDataset
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

data_path = Path("../data")
p_train, p_val = 0.8, 0.2
batch_size = 32
sub_batch_size = 50
lr = 0.001
epochs = 10

xlsx_path = data_path / "solar_panel_data_madagascar.xlsx"
img_path = data_path / "img"
weights_path = data_path / "WEIGHTS"
seg_weights_path = weights_path / "model_bdappv_seg.pth"

geo_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(size=(299, 299), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ]
)
photo_transform = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


dataset = SolarPanelDataset(
    img_path, xlsx_path, "seg", "pan", photo_transform, geo_transform
)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [p_train, p_val])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(seg_weights_path, weights_only=False).to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def dice_coefficient(prediction, target, epsilon=1e-07, threshold=0.5):
    prediction_copy = prediction.clone()

    prediction_copy = (prediction_copy - torch.min(prediction_copy)) / (
        torch.max(prediction_copy) - torch.min(prediction_copy) + epsilon
    )
    binary_prediction = (
        (prediction_copy >= threshold).long().squeeze(1).detach().cpu().numpy()
    )

    intersection = abs(torch.sum(binary_prediction * target))
    union = abs(torch.sum(binary_prediction) + torch.sum(target))
    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice


def train_one_epoch(epoch_index, tb_writer):
    last_loss, last_dc = 0.0, 0.0
    running_train_loss, running_train_dc = 0.0, 0.0
    for i, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        y_pred = model(img)
        optimizer.zero_grad()

        dc = dice_coefficient(y_pred, mask)
        loss = loss_fn(y_pred, mask)

        running_train_loss += loss.item()
        running_train_dc += dc.item()

        loss.backward()
        optimizer.step()

        if i % sub_batch_size == 0:
            last_loss = running_train_loss / sub_batch_size
            last_dc = running_train_dc / sub_batch_size
            print(f"BATCH {i+1} LOSS: {last_loss}, DICE: {last_dc}")
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            tb_writer.add_scalar("DICE/train", last_dc, tb_x)
            running_train_loss = 0.0
            running_train_dc = 0.0

    return last_loss, last_dc


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/solar_dataset_segmentation_{timestamp}")
best_val_dc = 10**12

for epoch in tqdm(range(epochs)):
    print(f"EPOCH {epoch + 1}:")
    model.train()
    avg_train_loss, avg_train_dc = train_one_epoch(epoch, writer)

    model.eval()
    running_val_loss = 0
    running_val_dc = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)["out"]
            loss = loss_fn(y_pred, mask)
            dc = dice_coefficient(y_pred, mask)

            running_val_loss += loss.item()
            running_val_dc += dc.item()

    avg_val_loss = running_val_loss / (idx + 1)
    avg_val_dc = running_val_dc / (idx + 1)
    print(f"LOSS train {avg_train_loss} valid {avg_val_loss}")
    print(f"DICE train {avg_train_dc} valid {avg_val_dc}")

    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": avg_train_loss, "Validation": avg_val_loss},
        epoch + 1,
    )
    writer.add_scalars(
        "Training vs. Validation DICE",
        {"Training": avg_train_dc, "Validation": avg_val_dc},
        epoch + 1,
    )
    writer.flush()

    if avg_val_dc < best_val_dc:
        best_val_dc = avg_val_dc
        model_path = weights_path / f"model_{timestamp}_{epoch}"
        torch.save(model.state_dict(), model_path)

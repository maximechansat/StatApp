from dataset import SolarPanelDataset
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def jaccard(prediction, target, epsilon=1e-07, threshold=0.5):
    # Ensure same shape
    prediction = (prediction >= threshold).long().detach()
    target = target.long()

    # Flatten (B, 1, H, W) → (B, H*W)
    prediction = prediction.view(prediction.shape[0], -1)
    target = target.view(target.shape[0], -1)

    intersection = torch.sum(prediction * target, dim=1)
    union = torch.sum(prediction, dim=1) + torch.sum(target, dim=1) - intersection

    jaccard_index = (intersection + epsilon) / (union + epsilon)
    return jaccard_index.mean()


def train_one_epoch(
    epoch_index,
    tb_writer,
    model,
    loss_fn,
    optimizer,
    train_dataloader,
    device,
    threshold,
):

    running_train_loss, running_train_jac = 0, 0

    for i, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].to(device)
        mask = img_mask[1].squeeze().float().to(device)

        y_pred = model(img)["out"].squeeze()

        bin_pred = (y_pred >= threshold).long()
        jac = jaccard(bin_pred, mask)
        loss = loss_fn(y_pred, mask)

        running_train_loss += loss.item()
        running_train_jac += jac.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 50 == 0:
            last_loss = running_train_loss / 50
            last_jac = running_train_jac / 50
            print(f"Batch {i+1}, loss: {last_loss:>7f}, Jaccard: {last_jac:>5f}")
            tb_x = epoch_index * len(train_dataloader) + i
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            tb_writer.add_scalar("Jaccard/train", last_jac, tb_x)
            running_train_loss = 0.0
            running_train_jac = 0.0


data_path = Path("../data")
batch_size = 15
lr = 0.001
epochs = 10
seed = 1048596
p = 0.3
num_workers = 8
threshold = 0.5

xlsx_path = data_path / "solar_panel_data_madagascar.xlsx"
img_path = data_path / "img"
weights_path = data_path / "WEIGHTS"
seg_weights_path = weights_path / "model_bdappv_seg.pth"

torch.manual_seed(seed)

train_dataset = SolarPanelDataset(img_path, xlsx_path, "seg", "pan", True, p, seed)
test_dataset = SolarPanelDataset(img_path, xlsx_path, "seg", "pan", False, p, seed)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(seg_weights_path, weights_only=False, map_location=device)
model = model.to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/solar_dataset_segmentation_{timestamp}")
best_test_loss = float("inf")

print("Training on: ", device)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}:")

    model.train()
    train_one_epoch(epoch, writer, model, loss_fn, optimizer, train_dataloader, device, threshold)
    model.eval()

    running_test_loss = 0
    running_test_jac = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            img = img_mask[0].to(device)
            mask = img_mask[1].squeeze().float().to(device)

            y_pred = model(img)["out"].squeeze()

            bin_pred = (y_pred >= threshold).long()
            jac = jaccard(bin_pred, mask)
            loss = loss_fn(y_pred, mask)

            running_test_loss += loss.item()
            running_test_jac += jac.item()

    avg_test_loss = running_test_loss / len(test_dataloader)
    avg_test_jac = running_test_jac / len(test_dataloader)

    print(f"Validation: loss: {avg_test_loss}, Jaccard {avg_test_jac}")
    writer.add_scalars("Loss/test", avg_test_loss, epoch)
    writer.add_scalars("Jaccard/test", avg_test_jac, epoch)
    writer.flush()

    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        model_path = weights_path / f"model_{timestamp}_{epoch}"
        torch.save(model.state_dict(), model_path)

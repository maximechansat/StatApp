from dataset import SolarPanelDataset
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


data_path = Path("../data")
batch_size = 16
sub_batch_size = 50
lr = 0.001
epochs = 10
seed = 1048596
p = 0.2
num_workers = 32

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
model = torch.compile(torch.load(seg_weights_path, weights_only=False))
model = model.to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def jaccard(prediction, target, epsilon=1e-07, threshold=0.5):
    binary_prediction = (
        (prediction >= threshold).long().squeeze(1).detach().cpu().numpy()
    )
    prediction = binary_prediction.view(binary_prediction.shape[0], -1)
    target = target.view(target.shape[0], -1)

    intersection = torch.sum(binary_prediction * target, dim=1)
    union = (
        torch.sum(binary_prediction, dim=1) + torch.sum(target, dim=1) - intersection
    )

    jaccard_index = (intersection + epsilon) / (union + epsilon)

    return jaccard_index


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/solar_dataset_segmentation_{timestamp}")
best_test_loss = 10**12


def train_one_epoch(epoch_index, tb_writer):
    running_train_loss, running_train_jac = 0, 0
    for i, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].to(device)
        mask = img_mask[1].to(device)

        y_pred = model(img)["out"]

        jac = jaccard(y_pred, mask)
        loss = loss_fn(y_pred, mask)
        running_train_loss += loss.item()
        running_train_jac += jac.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if i % sub_batch_size == 0:
            last_loss = running_train_loss / sub_batch_size
            last_jac = running_train_jac / sub_batch_size
            print(f"Batch {i+1}, loss: {last_loss:>7f}, Jaccard: {last_jac:>5f}")
            tb_x = epoch_index * len(train_dataloader) + i
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            tb_writer.add_scalar("Jaccard/train", last_jac, tb_x)
            running_train_loss = 0.0
            running_train_jac = 0.0


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}:")

    model.train()
    train_one_epoch(epoch, writer)
    model.eval()

    running_test_loss = 0
    running_test_jac = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)["out"]
            loss = loss_fn(y_pred, mask)
            dc = jaccard(y_pred, mask)

            running_test_loss += loss.item()
            running_test_jac += dc.item()

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

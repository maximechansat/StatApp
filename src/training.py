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


def train(
    model,
    loss_fn,
    metrics,
    optimizer,
    epochs,
    device,
    train_dataloader,
    test_dataloader,
    model_output_path,
    tb_writer,
    chunk_size,
    logging,
):

    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}:")
        for training in (True, False):
            dataloader = train_dataloader if training else test_dataloader
            if training:
                model.train()
            else:
                model.eval()

            torch.set_grad_enabled(training)

            running_loss = 0
            running_metrics = {metric: 0 for metric in metrics}

            for i, data in enumerate(tqdm(dataloader, position=0, leave=True)):
                img = data[0].to(device)
                target = data[1].squeeze().float().to(device)

                pred = model(img)["out"].squeeze()

                loss = loss_fn(pred, target)
                running_loss += loss.item()
                for metric, value in metrics:
                    running_metrics[metric] += metrics[metric](pred, target).item()

                if training:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if (i + 1) % chunk_size == 0 and training and logging:
                    avg_loss = running_loss / chunk_size
                    avg_metrics = {
                        metric: value / chunk_size for metric, value in running_metrics
                    }
                    log = f"Chunk {i // chunk_size}: loss: {avg_loss:>7f}"
                    for metric, value in avg_metrics.items():
                        log += f"; {metric}: {value:>7f}"
                    print(log)

                    index = epoch * len(train_dataloader) + i
                    tb_writer.add_scalar("Loss/train", avg_loss, index)
                    for metric, value in metrics.items():
                        tb_writer.add_scalar(f"{metric}/train", value, index)

                    running_loss = 0
                    running_metrics = {metric: 0 for metric in metrics}

                if not training:
                    avg_loss = running_loss / len(dataloader)
                    avg_metrics = {
                        metric: value / len(dataloader)
                        for metric, value in running_metrics
                    }

                    if logging:
                        log = f"Test {i // chunk_size}: loss: {avg_loss:>7f}"
                        for metric, value in avg_metrics.items():
                            log += f"; {metric}: {value:>7f}"
                        print(log)

                    tb_writer.add_scalar("Loss/test", avg_loss, epoch)
                    for metric, value in avg_metrics.items():
                        tb_writer.add_scalar(f"{metric}/test", value, epoch)
                        writer.flush()

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        model_path = model_output_path / f"model_{timestamp}_{epoch}"
                        torch.save(model.state_dict(), model_path)


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
from dataset import SolarPanelDataset
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, Callable
from metrics import *


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model_output_path: Path,
    tb_writer: SummaryWriter,
    chunk_size: int,
    logging: bool,
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
                long_target = target.long()

                pred = model(img)["out"].squeeze()
                detached_pred = pred.detach()

                loss = loss_fn(pred, target)
                running_loss += loss.item()
                for metric, value in metrics.items():
                    running_metrics[metric] += metrics[metric](
                        detached_pred, long_target
                    ).item()

                if training:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if (i + 1) % chunk_size == 0 and training and logging:
                    index = epoch * len(train_dataloader) + i
                    avg_loss = running_loss / chunk_size
                    log = f"Chunk {i // chunk_size}: Loss: {avg_loss:>7f}"
                    tb_writer.add_scalar("Loss/train", avg_loss, index)
                    for metric, value in running_metrics.items():
                        avg_value = value / chunk_size
                        log += f"; {metric}: {avg_value:>7f}"
                        tb_writer.add_scalar(f"{metric}/train", avg_value, index)
                    print(log)
                    tb_writer.flush()

                    running_loss = 0
                    running_metrics = {metric: 0 for metric in metrics}

            if not training:
                avg_loss = running_loss / len(dataloader)
                if logging:
                    log = f"Test: Loss: {avg_loss:>7f}"
                    tb_writer.add_scalar("Loss/test", avg_loss, epoch)
                    for metric, value in running_metrics.items():
                        avg_value = value / len(dataloader)
                        log += f"; {metric}: {avg_value:>7f}"
                        tb_writer.add_scalar(f"{metric}/test", avg_value, epoch)
                    print(log)
                    tb_writer.flush()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    model_path = model_output_path / f"model_{epoch}.pth"
                    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":

    data_path = Path("../data")
    batch_size = 15
    lr = 0.0001
    epochs = 25
    seed = 1048596
    p_test = 0.2
    num_workers = 8
    epsilon = 1e-7
    threshold = 0.5
    chunk_size = 50

    mode = "seg"

    xlsx_path = data_path / "solar_panel_data_madagascar.xlsx"
    img_path = data_path / "img"
    weights_path = data_path / "WEIGHTS"
    seg_weights_path = weights_path / f"model_bdappv_{mode}.pth"
    runs_path = data_path / "runs"

    torch.manual_seed(seed)

    if mode == "seg":
        train_dataset = SolarPanelDataset(
            img_path, xlsx_path, "seg", "pan", True, p_test, seed
        )
        test_dataset = SolarPanelDataset(
            img_path, xlsx_path, "seg", "pan", False, p_test, seed
        )

    elif mode == "cls":
        train_dataset = SolarPanelDataset(
            img_path, xlsx_path, "cls", "all", True, p_test, seed
        )
        test_dataset = SolarPanelDataset(
            img_path, xlsx_path, "cls", "all", False, p_test, seed
        )

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
    model = torch.compile(model).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    if mode == "seg":
        metrics = {"Jaccard": composer(jaccard, threshold, epsilon)}

    if mode == "cls":
        metrics = {
            "F1": composer(f1, threshold, epsilon),
            "Accuracy": composer(accuracy, threshold, epsilon),
            "Precision": composer(precision, threshold, epsilon),
            "Recall": composer(recall, threshold, epsilon),
        }

    timestamp = datetime.now().strftime("%H:%M:%S_%d-%m-%Y")
    model_name = "DeepLabV3" if mode == "seg" else "InceptionV3"
    model_output_path = runs_path / f"{model_name}_{timestamp}"
    tb_writer = SummaryWriter(model_output_path)

    train(
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
        True,
    )

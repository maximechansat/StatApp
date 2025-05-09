from dataset import SolarPanelDataset
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, Callable
import metrics
from collections import OrderedDict


def train(
    model: torch.nn.Module,
    mode: str,
    loss_fn: torch.nn.Module,
    metrics_dict: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
    optimizer: torch.optim.Optimizer,
    activation: Callable[[torch.Tensor, torch.Tensor], float],
    epochs: int,
    device,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model_output_path: Path,
    tb_writer: SummaryWriter,
    chunk_size: int,
    logging: bool,
    validation: bool = False
):

    best_loss = float("inf")
    index = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}:")
        for training in (True, False):
            if validation:
                training = False
            if training:
                print("Training")
            if not training and not validation:
                print("Testing")
            dataloader = train_dataloader if training else test_dataloader
            if training:
                model.train()
            else:
                model.eval()

            torch.set_grad_enabled(training)

            running_loss = 0
            running_metrics = {metric: 0 for metric in metrics_dict}

            for i, data in enumerate(tqdm(dataloader, position=0, leave=True)):
                img = data[0].to(device)
                target = data[1].squeeze().float().to(device)
                long_target = target.long()

                pred = model(img)
                if mode == "seg":
                    pred = pred["out"]
                if mode == "cls" and training:
                    pred = pred.logits[:, 1]
                if mode == "cls" and not training:
                    pred = pred[:, 1]
                pred = pred.squeeze()
                detached_pred = activation(pred.detach())

                loss = loss_fn(pred, target)
                running_loss += loss.item()
                for metric, value in metrics_dict.items():
                    running_metrics[metric] += metrics_dict[metric](
                        detached_pred, long_target
                    ).item()

                if training:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if (i + 1) % chunk_size == 0 and training and logging:
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
                    running_metrics = {metric: 0 for metric in metrics_dict}

                index += 1

            if validation:
                avg_metrics = {}
                avg_loss = running_loss / len(dataloader)
                for metric, value in running_metrics.items():
                    avg_metrics[metric] = value / len(dataloader)
                return avg_loss, avg_metrics

            if not training:
                avg_loss = running_loss / len(dataloader)
                if logging:
                    log = f"Test: Loss: {avg_loss:>7f}"
                    tb_writer.add_scalar("Loss/test", avg_loss, index)
                    for metric, value in running_metrics.items():
                        avg_value = value / len(dataloader)
                        log += f"; {metric}: {avg_value:>7f}"
                        tb_writer.add_scalar(f"{metric}/test", avg_value, index)
                    print(log)
                    tb_writer.flush()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    model_path = model_output_path / f"model_{epoch}.pth"
                    torch.save(model, model_path)


if __name__ == "__main__":

    data_path = Path("../data")
    batch_size = 128
    lr = 0.0001
    epochs = 25
    seed = 1048596
    probs = [0.8, 0.1, 0.1]
    num_workers = 8
    epsilon = 1e-7
    prediction_threshold = 0.5
    data_threshold = 0.01

    chunk_size = 10

    mode = "seg"

    xlsx_path = data_path / "solar_panel_data_madagascar.xlsx"
    img_path = data_path / "img"
    weights_path = data_path / "WEIGHTS" / f"model_bdappv_{mode}.pth"
    runs_path = data_path / "runs"

    torch.manual_seed(seed)

    if mode == "seg":
        train_dataset = SolarPanelDataset(
            img_path, xlsx_path, "seg", "pan", "train", probs, seed
        )
        test_dataset = SolarPanelDataset(
            img_path, xlsx_path, "seg", "pan", "test", probs, seed
        )

    elif mode == "cls":
        train_dataset = SolarPanelDataset(
            img_path, xlsx_path, "cls", "all", "train", probs, seed, data_threshold
        )
        test_dataset = SolarPanelDataset(
            img_path, xlsx_path, "cls", "all", "test", probs, seed, data_threshold
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

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(weights_path, weights_only=False, map_location=device)
    model = torch.compile(model).to(device)

    loss_fn = metrics.JaccardWithLogitLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    def activation(x):
        return (torch.nn.functional.sigmoid(x) >= prediction_threshold).long()

    if mode == "seg":
        metrics_dict = {"Jaccard": metrics.jaccard}

    if mode == "cls":
        metrics_dict = {
            "F1": metrics.f1,
            "Accuracy": metrics.accuracy,
            "Precision": metrics.precision,
            "Recall": metrics.recall,
        }

    timestamp = datetime.now().strftime("%H:%M:%S_%d-%m-%Y")
    model_name = "DeepLabV3" if mode == "seg" else "InceptionV3"
    model_output_path = runs_path / f"{model_name}_{timestamp}"
    tb_writer = SummaryWriter(model_output_path)

    train(
        model,
        mode,
        loss_fn,
        metrics_dict,
        optimizer,
        activation,
        epochs,
        device,
        train_dataloader,
        test_dataloader,
        model_output_path,
        tb_writer,
        chunk_size,
        True,
    )

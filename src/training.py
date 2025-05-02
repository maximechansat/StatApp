from dataset import SolarPanelDataset
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, Callable


def convert(prediction: torch.Tensor, threshold: float, epsilon: float) -> torch.Tensor:
    """
    Converts logits to prediction.

    Applies min-max scaling to the input tensor and consider positive the entries greater than a
    given threshold.

    Arguments:
        prediction (torch.Tensor): The input tensor, containing floats.
        threshold (float): A number between 0 and 1, defining the frontier between negative and
        positve.
        epsilon (float): A small number used to prevent divisions by zero.

    Returns:
        torch.Tensor: The prediction tensors, containing 0s and 1s.
    """
    min_max = torch.max(prediction) - torch.min(prediction)
    min_max_prediction = (prediction - torch.min(prediction)) / (min_max + epsilon)
    return (min_max_prediction >= threshold).long()


def accuracy(prediction, target):
    """
    Compute the accuracy of the prediction, defined as (TP + TN) / (TP + FP).

    The tensors must contain only 0s and 1s and be of the same (but arbitrary) shape.

    Arguments:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target (ie ground truth) tensor.

    Returns:
        torch.Tensor: The accuracy of the prediction.
    """
    torch.sum(prediction == target) / len(prediction)


def recall(prediction, target, epsilon):
    """
    Compute the recall of the prediction, defined as TP / (TP + FN).

    The tensors must contain only 0s and 1s and be of the same (but arbitrary) shape.

    Arguments:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target (ie ground truth) tensor.
        epsilon (float): A small number used to prevent divisions by zero.

    Returns:
        torch.Tensor: The recall of the prediction.
    """
    tp = torch.sum(prediction * target)
    fn = torch.sum((1 - prediction) * target)
    return tp / (tp + fn + epsilon)


def f1(prediction, target, epsilon):
    """
    Compute the F1 of the prediction, defined as the harmonic mean of the accuracy and the recall.

    The tensors must contain only 0s and 1s and be of the same (but arbitrary) shape.

    Arguments:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target (ie ground truth) tensor.
        epsilon (float): A small number used to prevent divisions by zero.

    Returns:
        torch.Tensor: The F1 of the prediction.
    """
    acc = accuracy(prediction, target)
    rec = recall(prediction, target, epsilon)
    return 2 * acc * rec / (acc + rec + epsilon)


def jaccard(prediction, target, epsilon):
    """
    Compute the Jaccard index of the prediction, defined as the mean of the intersection divided by the union of the detected areas.

    The tensors must contain only 0s and 1s and be of the same (but arbitrary) shape.

    Arguments:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target (ie ground truth) tensor.
        epsilon (float): A small number used to prevent divisions by zero.

    Returns:
        torch.Tensor: The Jaccard index of the prediction.
    """
    # Flatten (B, 1, H, W) → (B, H*W)
    prediction = prediction.view(prediction.shape[0], -1)
    target = target.view(target.shape[0], -1)

    intersection = torch.sum(prediction * target, dim=1)
    union = torch.sum(prediction, dim=1) + torch.sum(target, dim=1) - intersection

    jaccard_index = (intersection) / (union + epsilon)
    return jaccard_index.mean()


def composer(metric, threshold, epsilon):
    """
    An utility function used to compose the conversion function (convert) with a given metric.


    Arguments:
        metric ((torch.Tensor, torch.Tensor) -> torch.Tensor): The metric.
        threshold (float): A number between 0 and 1, defining the frontier between negative and
        positve.
        epsilon (float): A small number used to prevent divisions by zero.

    Returns:
        (torch.Tensor, torch.Tensor) -> torch.Tensor: The modified metric.
    """

    def converted_metric(prediction, target):
        return metric(convert(prediction, threshold, epsilon), target, epsilon)

    return converted_metric


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
            training = False
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
                    avg_metrics = {}
                    for metric, value in running_metrics.items():
                        avg_metrics[metric] = value / len(dataloader)
                    if logging:
                        log = f"Test {i // chunk_size}: loss: {avg_loss:>7f}"
                        for metric, value in avg_metrics.items():
                            log += f"; {metric}: {value:>7f}"
                        print(log)

                    tb_writer.add_scalar("Loss/test", avg_loss, epoch)
                    for metric, value in avg_metrics.items():
                        tb_writer.add_scalar(f"{metric}/test", value, epoch)
                    tb_writer.flush()

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        model_path = model_output_path / f"model_{timestamp}_{epoch}"
                        torch.save(model.state_dict(), model_path)


data_path = Path("../data")
batch_size = 15
lr = 0.0001
epochs = 25
seed = 1048596
p = 0.3
num_workers = 8
epsilon = 1e-7
threshold = 0.5
chunk_size = 50

mode = "seg"

xlsx_path = data_path / "solar_panel_data_madagascar.xlsx"
img_path = data_path / "img"
weights_path = data_path / "WEIGHTS"
seg_weights_path = weights_path / f"model_bdappv_{mode}.pth"
model_output_path = data_path / "runs"

torch.manual_seed(seed)

if mode == "seg":
    train_dataset = SolarPanelDataset(img_path, xlsx_path, "seg", "pan", True, p, seed)
    test_dataset = SolarPanelDataset(img_path, xlsx_path, "seg", "pan", False, p, seed)

elif mode == "cls":
    train_dataset = SolarPanelDataset(img_path, xlsx_path, "cls", "all", True, p, seed)
    test_dataset = SolarPanelDataset(img_path, xlsx_path, "cls", "all", False, p, seed)

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
    }

timestamp = datetime.now().strftime("%H:%M:%S_%d/%m/%Y")
model_name = "DeepLabV3" if mode == "cls" else "InceptionV3"
tb_writer = SummaryWriter(model_output_path / f"{model_name}_{timestamp}")

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

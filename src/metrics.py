import torch
from typing import Dict, Callable


def linear_activation(prediction: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
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
    return min_max_prediction


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
    prediction = prediction.view(prediction.shape[0], -1)
    target = target.view(target.shape[0], -1)
    tp = torch.sum(prediction * target, dim=1)
    tn = torch.sum((1 - prediction) * (1 - target), dim=1)
    return ((tp + tn) / prediction.shape[1]).mean()


def precision(prediction, target):
    """
    Compute the precision of the prediction, defined as TP / (TP + FP).

    The tensors must contain only 0s and 1s and be of the same (but arbitrary) shape.

    Arguments:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target (ie ground truth) tensor.

    Returns:
        torch.Tensor: The accuracy of the prediction.
    """
    prediction = prediction.view(prediction.shape[0], -1)
    target = target.view(target.shape[0], -1)
    tp = torch.sum(prediction * target, dim=1)
    return (tp / prediction.shape[1]).mean()


def recall(prediction, target, epsilon=1e-5):
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
    prediction = prediction.view(prediction.shape[0], -1)
    target = target.view(target.shape[0], -1)
    tp = torch.sum(prediction * target, dim=1)
    fn = torch.sum((1 - prediction) * target, dim=1)
    return (tp / (tp + fn + epsilon)).mean()


def f1(prediction, target, epsilon=1e-5):
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


def jaccard(prediction, target, epsilon=1e-5):
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


class JaccardWithLogitLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardWithLogitLoss, self).__init__()

    def forward(self, prediction, target, epsilon=0):

        prediction = torch.functional.sigmoid(prediction)
        prediction = prediction.view(prediction.shape[0], -1)
        target = target.view(target.shape[0], -1)
        intersection = torch.sum(prediction * target, dim=1)
        union = torch.sum(prediction, dim=1) + torch.sum(target, dim=1) - intersection
        jaccard_index = 1 - (intersection) / (union + epsilon)
        return torch.sum(jaccard_index)

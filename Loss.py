import torch
from torch import nn


class BFWithLogitsLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.

    Parameters:
    - gamma (float): Steepness parameter for the focal loss. Default is 2.
    - beta (float): Shift parameter for the sigmoid function. Default is 0.
    - reduction (str): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'. Default is 'mean'.
    """

    def __init__(self, gamma=2, beta=0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute the Binary Focal Loss between `logits` and `targets`.

        Parameters:
        - logits (torch.Tensor) [N,1]: The predicted values from the model.
        - targets (torch.Tensor) [N,1]: The ground truth labels (binary).

        Returns:
        - torch.Tensor: The computed loss value.
        """
        scaled_targets = targets * 2 - 1

        xt = logits * scaled_targets
        xt_star = torch.sigmoid(self.gamma * xt + self.beta)
        xt_star = torch.clamp(xt_star, min=1e-10, max=1)

        FL_star = ((-1) * torch.log(xt_star)) / self.gamma

        result = self._apply_reduction(FL_star)
        return result

    def _apply_reduction(self, loss):
        """
        Apply the specified reduction method to the loss.

        Parameters:
        - loss (torch.Tensor): The computed focal loss values.

        Returns:
        - torch.Tensor: The reduced loss value according to the specified reduction method.
        """
        if self.reduction == "mean":
            result = loss.mean()
        elif self.reduction == "sum":
            result = loss.sum()
        else:
            result = loss
        return result

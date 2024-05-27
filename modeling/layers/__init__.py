import torch
from modeling.layers.deviation_loss import DeviationLoss
from modeling.layers.binary_focal_loss import BinaryFocalLoss
from torch.nn import CrossEntropyLoss

def build_criterion(criterion, has_cuda):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss(has_cuda)
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss(has_cuda)
    elif criterion == "cross":
        print("cross entropy")
        return CrossEntropyLoss()
    else:
        raise NotImplementedError
import torch
from modeling.layers.deviation_loss import DeviationLoss
from modeling.layers.binary_focal_loss import BinaryFocalLoss

def build_criterion(criterion, cuda):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss(cuda)
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss(cuda)
    else:
        raise NotImplementedError
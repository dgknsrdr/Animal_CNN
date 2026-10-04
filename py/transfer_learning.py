from torchvision.models import EfficientNet_B0_Weights , efficientnet_b0
import torch.nn as nn



def EfficienModel(num_classes: int = 45, pretrained: bool = True):

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

    efficient_model = efficientnet_b0(weights=weights)

    for param in efficient_model.parameters():
        param.requires_grad = False

    efficient_model.classifier[1] = nn.Linear(
        efficient_model.classifier[1].in_features,
        num_classes
    )

    return efficient_model
from torch import nn
from torchvision import models


def model(num_classes):
    model_tl = models.densenet121(pretrained=True)
    for param in model_tl.features.parameters():
        param.requires_grad = False
    hidden_sizes = [512, 64]
    model_tl.classifier = nn.Sequential(
        nn.Linear(model_tl.classifier.in_features, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], num_classes),
        nn.LogSoftmax(dim=1),
    )

    return model_tl

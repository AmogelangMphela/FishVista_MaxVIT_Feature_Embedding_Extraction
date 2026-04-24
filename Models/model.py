from torchvision.models import maxvit_t, MaxVit_T_Weights
import torch.nn as nn

def custom_model(model_name: str, num_classes: int, pretrained:bool=True):
    if model_name == 'maxvit_t':
        if pretrained:
            weights = MaxVit_T_Weights.DEFAULT
            model = maxvit_t(weights = weights)
            model.classifier[5] = nn.Linear(in_features = 512, out_features = num_classes, bias = True)
            print("model created")
        else:
            model = maxvit_t()
            model.classifier[5] = nn.Linear(in_features = 512, out_features = num_classes, bias = True)

    return model


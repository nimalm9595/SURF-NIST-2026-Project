import sys
import os
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pytorch.pathmnist_cnn import Net

def extract_layer_metadata(model):
    layers = []

    for layer_name, layer in model.named_modules():
        if layer_name == "":
            continue
    
        layer_info = {
            "layer_name" : layer_name,
            "layer_type" : type(layer).__name__,
            "layer_details" : str(layer)
        }

        if isinstance(layer, nn.Conv2d):
            layer_info["in_channels"] = layer.in_channels
            layer_info["out_channels"] = layer.out_channels
            layer_info["kernel_size"] = layer.kernel_size
            layer_info["stride"] = layer.stride
            layer_info["padding"] = layer.padding
    
        elif isinstance(layer, nn.BatchNorm2d):
            layer_info["num_features"] = layer.num_features
            layer_info["eps"] = layer.eps
            layer_info["momentum"] = layer.momentum
    
        elif isinstance(layer, nn.MaxPool2d):
            layer_info["kernel_size"] = layer.kernel_size
            layer_info["stride"] = layer.stride
            layer_info["padding"] = layer.padding
    
        elif isinstance(layer, nn.Linear):
            layer_info["in_features"] = layer.in_features
            layer_info["out_features"] = layer.out_features
    
        elif isinstance(layer, nn.ReLU):
            layer_info["activation_function"] = "ReLU"
    
        layers.append(layer_info)

    return layers

def extract_model_metadata():
    model = Net(in_channels=3, num_classes=9)

    metadata = {
        "source_model_file": "models/pytorch/pathmnist_cnn.py",
        "framework": "PyTorch",
        "model_name": "MedMNIST_CNN",
        "data_set": "PathMNIST",
        "input_channels": 3,
        "num_classes": 9,
        "model_type": type(model).__name__,
        "layers": extract_layer_metadata(model) 
    }

    return metadata

if __name__ == "__main__":
    metadata = extract_model_metadata()

    with open("medmnist_pytorch_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(json.dumps(metadata, indent=4))







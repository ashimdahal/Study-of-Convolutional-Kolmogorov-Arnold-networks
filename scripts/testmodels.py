import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchmetrics

from convkan import ConvKAN, LayerNorm2D

import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

import pandas as pd
import os

from torchsummary import summary
from fvcore.nn import FlopCountAnalysis

from lenetfastkan import LeNet5_KAN
from alexnetkan import AlexNetKAN
from lenetcnn import LeNet5

import time

device = "cuda:0"

def load_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_ds = datasets.ImageNet(
                            root="./imagenet/val/", 
                            split='val', 
                            transform=transform,
                        )
    # targets = train_ds.targets
    # _, test_train = train_test_split(np.arange(len(targets)), test_size = 0.05, stratify = targets, random_state=42 )
    # train_ds = Subset(train_ds, test_train)
    #
    # valid_targets = val_ds.targets
    # _, test_val = train_test_split(np.arange(len(valid_targets)), test_size = 0.05, stratify = valid_targets, random_state=42)
    # val_ds = Subset(val_ds, test_val)

    print(f" testing validation size: {len(val_ds)}")

    return val_ds


def load_model(snapshot_path):
    
    if "alexnetkan" in snapshot_path:
        model = AlexNetKAN()
    elif "lenetfastkan" in snapshot_path:
        model = LeNet5_KAN()
    elif "lenetcnn" in snapshot_path:
        model = LeNet5()

    snapshot = torch.load(f"./model_snapshots/{snapshot_path}", 
                          map_location = device)

    new_model_state = {}

    model_state = snapshot["MODEL_STATE"]
    for key,val in model_state.items():
        if key.startswith("module."):
            new_model_state[key[7:]] = val 
        else:
            new_model_state[key] = val 

    model.load_state_dict(new_model_state)
    model.eval()
    model.to(device)
    return model, snapshot

def prepare_data(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size= batch_size,
        # num_workers=0,
        shuffle=False,
        pin_memory=True,
    )

def top_k_metrics(output, target, num_classes, k=5):
    top_k_acc = torchmetrics.Accuracy(top_k=k, num_classes=num_classes, task='multiclass')
    precision = torchmetrics.Precision(num_classes=num_classes, average='macro', top_k=k, task='multiclass')
    recall = torchmetrics.Recall(num_classes=num_classes, average='macro', top_k=k, task='multiclass')
    f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro', top_k=k, task='multiclass')
    
    # Compute metrics
    top_k_acc_value = top_k_acc(output, target)
    precision_value = precision(output, target)
    recall_value = recall(output, target)
    f1_value = f1(output, target)

    out = {
        f"top_{k}_acc": top_k_acc_value.item(),
        f"top_{k}_precision": precision_value.item(), 
        f"top_{k}_recall": recall_value.item(),
        f"top_{k}_f1": f1_value.item()
    }

    return out 

def concatenated_classification_report(output, target,name, num_classes=1000, k=5, top_k = False):

    pred_classes = torch.argmax(output, dim=1).cpu().numpy()

    classification_report_sklearn = classification_report(
        target.cpu().numpy(),
        pred_classes,
        output_dict=True
    )
    classification_report_df = pd.DataFrame(classification_report_sklearn).transpose()

    if top_k:
        top_k_metrics_ = top_k_metrics(output, target, num_classes, k)
        top_k_metrics_df = pd.DataFrame([top_k_metrics_])

        classification_report_df = pd.concat([classification_report_df, top_k_metrics_df], axis=0, ignore_index=True)

    classification_report_df.to_csv(f"{name} report.csv", index=False)

    print("report saved")

def load_history_and_plot_graph(hist, epochs, name= "test"):
    valid_loss = []
    valid_acc = []
    train_loss = []
    for i in range(len(hist)):
        valid_loss.append(hist[i]['valid_loss'])
        valid_acc.append(hist[i]['valid_acc'])
        train_loss.append(hist[i]['train_loss'])

    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(epochs), valid_loss, label = "validation loss")
    plt.plot(np.arange(epochs), train_loss, label = "training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training vs Validation Loss {name}")
    plt.savefig(f"./graphs/loss {name}.png", dpi=300, bbox_inches="tight")
    print("printed")

    plt.figure()
    plt.plot(np.arange(epochs), valid_acc, label = "validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # plt.legend()
    plt.title(f"Top-1 Validation Accuracy over Epochs {name}")
    plt.savefig(f"./graphs/validation {name}.png", dpi=300, bbox_inches="tight")
    print("printed")

@torch.inference_mode()
@torch.no_grad()
def calculate_output(dataset, model):

    batch_size = 32
    dl = prepare_data(dataset, batch_size)

    print("computing outputs")
    all_y_preds = []
    all_y_inputs = []
    for x,y in dl:
        y_pred = model(x.to(device)).cpu().numpy()

        all_y_preds.append(y_pred)
        all_y_inputs.append(y)

    all_outputs = np.concatenate(all_y_preds, axis=0)
    all_targets = np.concatenate(all_y_inputs, axis=0)

    all_outputs = torch.from_numpy(all_outputs)
    all_targets = torch.from_numpy(all_targets)

    return all_outputs, all_targets

def main():
    model_details = {
        "AlexNet KAN":"alexnetkan",
    }

    for name, snapshot_path in model_details.items():
        print(f"Printing Results for {name}")
        model, snapshot = load_model(snapshot_path)
        hist = snapshot["HIST"]
        epochs = snapshot["EPOCHS_RUN"]

        dataset = load_data()
        
        dl = prepare_data(dataset, 1)
        sample = next(iter(dl))[0].to(device)

        flops = FlopCountAnalysis(model, sample)
        print(f"flops: {flops.total()}")

        start = time.time()
        out = model(sample)
        end = time.time()
        print(f"Time taken for {name} inference: {end-start}s")


        print(summary(model, (1, 32,32), device="cuda"))
        
        print(f" length of testing data {len(dataset)}")

        # load_history_and_plot_graph(hist, epochs, name)

        with torch.no_grad():
            all_outputs, all_targets = calculate_output(dataset, model)

        concatenated_classification_report(all_outputs, all_targets, name)
        print("*"*100)

if __name__ == "__main__":
    main()

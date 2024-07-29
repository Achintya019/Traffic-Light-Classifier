import os
import math
import torch
import argparse
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from utils.tool import *
from utils.datasets import *
from module.loss import DetectorLoss, CombinedLoss
from trafficlight_cls import LightClassifier
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
import numpy as np
from collections import Counter
import mlflow
import mlflow.pytorch
from datetime import datetime

class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root, transform=None)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        # Load the image and label
        path, target = self.samples[index]
        image = self.loader(path)

        # Apply Albumentations transformations
        if self.albumentations_transform is not None:
            image = self.albumentations_transform(image=np.array(image))['image']

        return image, target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer_model, optimizer_center, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_cl_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device) / 255.0
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            # Forward pass
            features, logits = model(inputs)
            loss, ce_loss, cl_loss = criterion(features, labels, logits)

            # Backward pass and optimization
            loss.backward()
            optimizer_model.step()
            optimizer_center.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_ce_loss += ce_loss.item() * inputs.size(0)
            running_cl_loss += cl_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_ce_loss = running_ce_loss / len(train_dataloader.dataset)
        epoch_cl_loss = running_cl_loss / len(train_dataloader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Loss: {epoch_loss:.4f}, '
              f'CE Loss: {epoch_ce_loss:.4f}, '
              f'Center Loss: {epoch_cl_loss:.4f}')
        
        # Log training metrics to mlflow
        mlflow.log_metric('loss', epoch_loss, step=epoch)
        mlflow.log_metric('ce_loss', epoch_ce_loss, step=epoch)
        mlflow.log_metric('center_loss', epoch_cl_loss, step=epoch)

        # Validation phase
        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            val_running_loss = 0.0
            val_running_ce_loss = 0.0
            val_running_cl_loss = 0.0
            correct = 0
            total = 0
            torch.save(model.state_dict(), f"weights/mlflow/tflt_weight_loss_{epoch_loss:.4f}_{epoch}-epoch.pth")
                                    
            with torch.no_grad():
                for val_inputs, val_labels in val_dataloader:
                    val_inputs = val_inputs.to(device) / 255.0
                    val_labels = val_labels.to(device)

                    # Forward pass
                    val_features, val_logits = model(val_inputs)
                    val_loss, val_ce_loss, val_cl_loss = criterion(val_features, val_labels, val_logits)

                    # Statistics
                    val_running_loss += val_loss.item() * val_inputs.size(0)
                    val_running_ce_loss += val_ce_loss.item() * val_inputs.size(0)
                    val_running_cl_loss += val_cl_loss.item() * val_inputs.size(0)

                    _, predicted = torch.max(val_logits, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
            val_epoch_ce_loss = val_running_ce_loss / len(val_dataloader.dataset)
            val_epoch_cl_loss = val_running_cl_loss / len(val_dataloader.dataset)
            val_accuracy = correct / total

            print(f'Validation - Epoch {epoch}/{num_epochs - 1}, '
                  f'Val Loss: {val_epoch_loss:.4f}, '
                  f'Val CE Loss: {val_epoch_ce_loss:.4f}, '
                  f'Val Center Loss: {val_epoch_cl_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')
            
            # Log validation metrics to mlflow
            mlflow.log_metric('val_loss', val_epoch_loss, step=epoch)
            mlflow.log_metric('val_ce_loss', val_epoch_ce_loss, step=epoch)
            mlflow.log_metric('val_center_loss', val_epoch_cl_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_accuracy, step=epoch)

if __name__ == "__main__":
    
    # Set the tracking URI to the MLflow server on a different host id
    #mlflow.set_tracking_uri("http://10.10.10.210:5001")
    
    # Initialize mlflow with run name
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)

    # Define directories
    data_dir = '/home/achintya-trn0175/Downloads/trafficLightClassification/FINAL_DATASET_TRAFFICLIGHT'

    # Define transformations for the training and validation sets
    data_transforms = {
        'train': A.Compose([
            A.Resize(192, 192),  # Resize images to a fixed size
            A.HorizontalFlip(p=0.5),  # Data augmentation: random horizontal flip
            A.Rotate(limit=15, p=0.5),  # Random rotations
            ToTensorV2()  # Convert images to PyTorch tensors
        ]),
        'val': A.Compose([
            A.Resize(192, 192),  # Resize images to a fixed size
            ToTensorV2()  # Convert images to PyTorch tensors
        ]),
    }

    # Create dataset using ImageFolder
    full_dataset = CustomDataset(root=data_dir, transform=data_transforms['train'])

    # Define class names
    class_names = full_dataset.classes
    print("Classes:", class_names)

    # Log class names to mlflow
    mlflow.log_param('class_names', class_names)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Calculate class weights for WeightedRandomSampler (only for training set)
    train_labels = [full_dataset.targets[i] for i in train_indices]
    class_counts = Counter(train_labels)
    
    class_weights = {class_: 1.0 / count for class_, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Update the transform for the validation set
    val_dataset.dataset.transform = data_transforms['val']

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = LightClassifier(len(class_names), False).to(device)

    lossfunc = CombinedLoss(num_classes=len(class_names), 
                            feat_dim=192,
                            device=device)
    
    optimizer = optim.SGD(params=model.parameters(),
                          lr=0.001,
                          momentum=0.949,
                          weight_decay=0.0005)
    
    optimizer_center = torch.optim.SGD(lossfunc.center_loss.parameters(), lr=0.001)   

    # Log hyperparameters to mlflow
    mlflow.log_param('learning_rate', 0.001)
    mlflow.log_param('momentum', 0.949)
    mlflow.log_param('weight_decay', 0.0005)
    mlflow.log_param('num_epochs', 1500)

    train_model(
        model,
        train_loader,
        val_loader,
        lossfunc,
        optimizer,
        optimizer_center,
        1500
    )

    mlflow.end_run()

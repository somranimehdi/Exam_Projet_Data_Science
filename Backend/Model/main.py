import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import recall_score, f1_score

from train import train
from TrafficSignClassifier import TrafficSignClassifier
from TrafficSignsDataset import TrafficSignsDataset
from evaluate import evaluate_model_on_test_data
from utils import load_data, plot_losses, load_processed_data

classes = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 
           'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 
           'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']

def main():
    num_classes = 15
    batch_size = 64
    epochs = 15
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load training data
    train_image_dir = "Dataset/output/train/images"
    train_label_dir = "Dataset/output/train/labels"
    train_data, train_labels = load_data(train_image_dir, train_label_dir)
    print(f"Loaded {len(train_data)} training images with labels.")

    # Load processed data
    train2_image_dir = "Dataset/processed_data/train/images"
    train2_label_dir = "Dataset/processed_data/train/labels"
    train2_data, train2_labels = load_processed_data(train2_image_dir, train2_label_dir, classes)
    print(f"Loaded {len(train2_data)} processed training images with labels.")

    # Load validation data
    valid_image_dir = "Dataset/output/valid/images"
    valid_label_dir = "Dataset/output/valid/labels"
    valid_data, valid_labels = load_data(valid_image_dir, valid_label_dir)
    print(f"Loaded {len(valid_data)} validation images with labels.")

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(0, shear=0.2),
        transforms.RandomAffine(0, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Check shapes of data before concatenation
    print(f"train2_data shape: {train2_data.shape}")
    print(f"train_data shape: {train_data.shape}")
    print(f"train2_labels shape: {train2_labels.shape}")
    print(f"train_labels shape: {train_labels.shape}")

    # Ensure the data and labels match
    assert len(train2_data) == len(train2_labels), "Mismatch between train2 data and labels count!"
    assert len(train_data) == len(train_labels), "Mismatch between train data and labels count!"
    
    # Concatenate the training datasets and labels
    concatenated_data = np.concatenate((train2_data, train_data), axis=0)
    concatenated_labels = np.concatenate((train2_labels, train_labels), axis=0)

    print(f"Concatenated data shape: {concatenated_data.shape}")
    print(f"Concatenated labels shape: {concatenated_labels.shape}")

    # Prepare the datasets
    train_dataset = TrafficSignsDataset(concatenated_data, concatenated_labels, transform=transform)
    valid_dataset = TrafficSignsDataset(valid_data, valid_labels, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = TrafficSignClassifier(num_classes=num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, val_losses = train(model, train_loader, valid_loader, criterion, optimizer, device, epochs=epochs)

    # Save the model
    torch.save(model.state_dict(), "traffic_sign_classifier.pth")

    # Test the model
    test_image_dir = "Dataset/output/test/images"
    test_label_dir = "Dataset/output/test/labels"
    testSet_loss, testSet_accuracy, testSet_recall, testSet_f1 = evaluate_model_on_test_data(
        model, test_image_dir, test_label_dir, criterion, device, transform
    )
    print(f"Test Loss: {testSet_loss:.4f}, Test Accuracy: {testSet_accuracy:.4f}")
    print(f"Test Recall: {testSet_recall:.4f}, Test F1 Score: {testSet_f1:.4f}")

    # Plot the losses
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()

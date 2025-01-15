import torch
from TrafficSignsDataset import TrafficSignsDataset
from utils import load_data, plot_confusion_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Class labels
classes = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 
           'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 
           'Speed Limit 80', 'Speed Limit 90', 'Stop']

def evaluate_model_on_test_data(model, image_dir, label_dir, criterion, device, transform):
    # Load test data
    test_data, test_labels = load_data(image_dir, label_dir)
    test_dataset = TrafficSignsDataset(test_data, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []  # For storing predicted labels
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Prediction and accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store labels, predicted labels, and probabilities for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            all_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = correct / total

    # Calculate recall and F1 score
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader, classes, device)

    # Plot ROC curve
   

    # Return loss, accuracy, recall, and F1 score
    return test_loss, test_accuracy, recall, f1

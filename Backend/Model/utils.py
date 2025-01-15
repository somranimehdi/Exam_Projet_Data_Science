import os
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Data Preparation
def load_data(image_dir, label_dir):
    data = []
    labels = []
    
    # Get all image filenames
    image_files = os.listdir(image_dir)
    
    for img_name in image_files:
        try:
            # Full image path
            img_path = os.path.join(image_dir, img_name)
            
            # Full label path (replace .jpg/.png with .txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)
            
            # Ensure the label file exists
            if not os.path.exists(label_path):
                print(f"Label file {label_name} not found for image {img_name}. Skipping.")
                continue
            
            # Load image
            image = Image.open(img_path).resize((30, 30))
            data.append(np.array(image))
            
            # Load label
            with open(label_path, "r") as f:
                label = int(f.read().strip())  # Assume single-line label files
            labels.append(label)
        
        except Exception as e:
            print(f"Error loading image {img_name} or label: {e}")
    
    return np.array(data), np.array(labels)

def load_processed_data(image_dir, label_dir, classes):
    data = []
    labels = []
    
    # Get all image filenames
    image_files = os.listdir(image_dir)
    
    for img_name in image_files:
        try:
            # Full image path
            img_path = os.path.join(image_dir, img_name)
            
            # Full label path (replace .jpg/.png with .txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)
            
            # Ensure the label file exists
            if not os.path.exists(label_path):
                print(f"Label file {label_name} not found for image {img_name}. Skipping.")
                continue
            
            # Load image
            image = Image.open(img_path).resize((30, 30))
            data.append(np.array(image))
            
            # Load label (text) and map to index
            with open(label_path, "r") as f:
                label_text = f.read().strip()  # Read the label as text
                if label_text not in classes:
                    print(f"Label '{label_text}' not found in classes. Skipping.")
                    continue
                label = classes.index(label_text)  # Map text to index
            labels.append(label)
        
        except Exception as e:
            print(f"Error loading image {img_name} or label: {e}")
    
    return np.array(data), np.array(labels)
# Plotting Function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(model, test_loader, classes, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.show()
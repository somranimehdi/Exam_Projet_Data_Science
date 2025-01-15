import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Function to load images and their corresponding labels from directory
def load_images_and_labels(image_dir, label_dir):
    images = []
    labels = []
    
    # Iterate through all label files in the directory
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
        
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found for label {label_path}. Skipping.")
            continue
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}. Skipping.")
            continue
        
        # Read label file and extract class info
        with open(label_path, "r") as f:
            annotations = f.readlines()
        
        for annotation in annotations:
            data = annotation.strip().split()
 
            
            class_id = data[0]
            labels.append(class_id)
            images.append(image)
    
    return images, labels

# Function to plot sample images from the dataset
def plot_sample_images(images, labels, num_images=5, class_names=None):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        image = images[i]
        label = labels[i]
        axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        axes[i].set_title(f'Class: {class_names[int(label)]}' if class_names else f'Label: {label}')
        axes[i].axis('off')
    
    plt.show()

# Function to plot class distribution (how many samples for each class)
def plot_class_distribution(labels, class_names=None):
    # Count the occurrences of each label
    label_counts = Counter(labels)
    
    # Sorting the labels based on counts
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Plotting the class distribution
    classes_sorted, counts_sorted = zip(*sorted_labels)
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes_sorted, counts_sorted)
    plt.xticks(classes_sorted, [class_names[int(x)] for x in classes_sorted] if class_names else classes_sorted, rotation=90)
    plt.xlabel('Traffic Signs')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

# Function to display basic dataset information
def display_dataset_info(images, labels, class_names=None):
    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")
    
    if class_names:
        print("\nClasses in the dataset:")
        print(", ".join(class_names))
    else:
        print("\nClasses (numeric IDs):")
        print(set(labels))

# Main function to perform data exploration
def main():
    # Dataset directories
    train_image_dir = "Dataset/augmented_data/train/images"
    train_label_dir = "Dataset/augmented_data/train/labels"
    valid_image_dir = "Dataset/augmented_data/valid/images"
    valid_label_dir = "Dataset/augmented_data/valid/labels"
    test_image_dir = "Dataset/augmented_data/test/images"
    test_label_dir = "Dataset/augmented_data/test/labels"
    
    # Class names (you can replace these with actual names of your classes)
    class_names = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']
    
    # Load data
    train_images, train_labels = load_images_and_labels(train_image_dir, train_label_dir)
    valid_images, valid_labels = load_images_and_labels(valid_image_dir, valid_label_dir)
    test_images, test_labels = load_images_and_labels(test_image_dir, test_label_dir)
    
    # Concatenate all images and labels from train, valid, and test sets
    all_images = train_images + valid_images + test_images
    all_labels = train_labels + valid_labels + test_labels
    
    # Display dataset information for the concatenated data
    display_dataset_info(all_images, all_labels, class_names)
    
    # Plot sample images from the concatenated dataset
    plot_sample_images(all_images, all_labels, num_images=5, class_names=class_names)
    
    # Plot class distribution in the concatenated dataset
    plot_class_distribution(all_labels, class_names)

if __name__ == "__main__":
    main()

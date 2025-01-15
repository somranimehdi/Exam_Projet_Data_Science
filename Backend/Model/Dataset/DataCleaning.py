import os

# Function to load images and their corresponding labels from directory
def load_images_and_labels(image_dir, label_dir):
    images_to_delete = []  # Keep track of image files to delete

    # Iterate through all label files in the directory
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
        
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found for label {label_path}. Skipping.")
            continue
        
        # Read label file and check for class 2
        delete_image = False
        with open(label_path, "r") as f:
            annotations = f.readlines()
        
        # If any annotation contains class '2' (Speed Limit 10), mark the image for deletion
        for annotation in annotations:
            data = annotation.strip().split()
            if len(data) < 5:
                continue
            
            class_id = data[0]
            if class_id == '2':  # If the class ID is 2, mark for deletion
                delete_image = True
        
        if delete_image:
            images_to_delete.append(image_path)  # Store image to delete

    return images_to_delete

# Function to delete images and their corresponding label files
def delete_images(images_to_delete, image_dir, label_dir):
    for image_file in images_to_delete:
        label_file = image_file.replace(".jpg", ".txt")
        
        # Delete image and corresponding label file if they exist
        if os.path.exists(image_file):
            print(f"Deleting image: {image_file}")
            os.remove(image_file)
        
        if os.path.exists(label_file):
            print(f"Deleting label: {label_file}")
            os.remove(label_file)

# Main function
def main():
    # Dataset directories
    train_image_dir = "Dataset/Dataset/train/images"
    train_label_dir = "Dataset/Dataset/train/labels"
    valid_image_dir = "Dataset/Dataset/valid/images"
    valid_label_dir = "Dataset/Dataset/valid/labels"
    test_image_dir = "Dataset/Dataset/test/images"
    test_label_dir = "Dataset/Dataset/test/labels"
    
    # Load data from all sets (train, valid, and test)
    train_images_to_delete = load_images_and_labels(train_image_dir, train_label_dir)
    valid_images_to_delete = load_images_and_labels(valid_image_dir, valid_label_dir)
    test_images_to_delete = load_images_and_labels(test_image_dir, test_label_dir)
    
    # Combine lists of images to delete
    all_images_to_delete = train_images_to_delete + valid_images_to_delete + test_images_to_delete
    
    # Delete images and labels with class '2' (Speed Limit 10)
    delete_images(all_images_to_delete, train_image_dir, train_label_dir)
    delete_images(all_images_to_delete, valid_image_dir, valid_label_dir)
    delete_images(all_images_to_delete, test_image_dir, test_label_dir)
    
    print("Deletion complete.")

if __name__ == "__main__":
    main()

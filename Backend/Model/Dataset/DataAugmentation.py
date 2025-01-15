import os
import cv2
import numpy as np
import random
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths

# Set up paths
train_image_dir = "Dataset/Dataset/train/images"
train_label_dir = "Dataset/Dataset/train/labels"


output_dir = "augmented_data"

# Create output directories for augmented data
os.makedirs(os.path.join(output_dir, "train/images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train/labels"), exist_ok=True)


# Function to load images and labels
def load_images_and_labels(image_dir, label_dir):
    images = []
    labels = []
    image_paths = list(paths.list_images(image_dir))

    for image_path in image_paths:
        label_file = image_path.replace(image_dir, label_dir).replace(".jpg", ".txt")
        if not os.path.exists(label_file):
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        with open(label_file, "r") as f:
            annotations = f.readlines()
        
        for annotation in annotations:
            data = annotation.strip().split()
            if len(data) < 5:
                continue
            class_id = int(data[0])
            images.append(image)
            labels.append(class_id)
    
    return images, labels

# Function to balance classes using augmentation
def balance_classes(images, labels):
    class_images = {}
    for img, label in zip(images, labels):
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(img)
    
    # Find the class with the maximum number of images
    max_class_count = max([len(imgs) for imgs in class_images.values()])

    augmented_images = []
    augmented_labels = []

    for label, imgs in class_images.items():
        num_images = len(imgs)
        if num_images < max_class_count:
            # Augment class to balance
            augment_count = max_class_count - num_images
            augment_images = random.choices(imgs, k=augment_count)
            augmented_images.extend(imgs + augment_images)
            augmented_labels.extend([label] * max_class_count)
        else:
            augmented_images.extend(imgs)
            augmented_labels.extend([label] * num_images)

    return augmented_images, augmented_labels

# Apply augmentation to the images
def augment_images(images, labels, augment_factor=1.5):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        image = np.expand_dims(image, axis=0)
        i = 0  # Image counter
        
        for batch in datagen.flow(image, batch_size=1, save_to_dir=output_dir, save_prefix=str(label), save_format="jpg"):
            augmented_images.append(batch[0].astype(np.uint8))
            augmented_labels.append(label)
            i += 1
            if i >= augment_factor:
                break

    return augmented_images, augmented_labels

# Main function to process and augment dataset
def main():
    # Load images and labels from the training, validation, and test sets
    train_images, train_labels = load_images_and_labels(train_image_dir, train_label_dir)


    # Balance the classes in training data
    augmented_train_images, augmented_train_labels = balance_classes(train_images, train_labels)

    # Apply augmentation to the training data
    augmented_train_images, augmented_train_labels = augment_images(augmented_train_images, augmented_train_labels)

    # Save augmented images and labels into output directories for train
    for i, img in enumerate(augmented_train_images):
        cv2.imwrite(f"{output_dir}/train/images/{i}.jpg", img)
        with open(f"{output_dir}/train/labels/{i}.txt", "w") as f:
            f.write(f"{augmented_train_labels[i]}\n")



    # Display total number of augmented images
    print(f"Total augmented train images: {len(augmented_train_images)}")


if __name__ == "__main__":
    main()

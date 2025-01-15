import os
import cv2


class YOLOProcessor:
    def __init__(self, input_images_dir, input_labels_dir, output_dir, dataset_type="train"):
        self.input_images_dir = input_images_dir
        self.input_labels_dir = input_labels_dir
        self.output_dir = os.path.join(output_dir, dataset_type)
        self.output_images_dir = os.path.join(self.output_dir, "images")
        self.output_labels_dir = os.path.join(self.output_dir, "labels")

        # Create output directories if they don't exist
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)

    def process_images_and_labels(self):
        for label_file in os.listdir(self.input_labels_dir):
            if not label_file.endswith(".txt"):
                continue
            
            label_path = os.path.join(self.input_labels_dir, label_file)
            image_path = os.path.join(self.input_images_dir, label_file.replace(".txt", ".jpg"))
            
            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found for label {label_path}. Skipping.")
                continue
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image {image_path}. Skipping.")
                continue
            
            height, width, _ = image.shape
            
            # Read the label file and extract bounding boxes
            with open(label_path, "r") as f:
                annotations = f.readlines()
            
            for idx, annotation in enumerate(annotations):
                data = annotation.strip().split()
                if len(data) < 5:
                    continue
                
                class_id = data[0]
                x_center, y_center, bbox_width, bbox_height = map(float, data[1:])
                
                # Convert YOLO format to bounding box coordinates
                x_min = int((x_center - bbox_width / 2) * width)
                y_min = int((y_center - bbox_height / 2) * height)
                x_max = int((x_center + bbox_width / 2) * width)
                y_max = int((y_center + bbox_height / 2) * height)
                
                # Ensure bounding box coordinates are within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width, x_max)
                y_max = min(height, y_max)
                
                # Crop the image
                cropped_image = image[y_min:y_max, x_min:x_max]
                
                if cropped_image.size == 0:
                    print(f"Invalid crop for {label_file} at index {idx}. Skipping.")
                    continue
                
                # Save the cropped image
                output_image_path = os.path.join(self.output_images_dir, f"{os.path.splitext(label_file)[0]}_{idx}.jpg")
                cv2.imwrite(output_image_path, cropped_image)
                
                # Save the label for the cropped image
                output_label_path = os.path.join(self.output_labels_dir, f"{os.path.splitext(label_file)[0]}_{idx}.txt")
                with open(output_label_path, "w") as f:
                    f.write(class_id)
                
                print(f"Saved cropped image: {output_image_path} with label: {class_id}")


# Main function to process datasets
def main():
    dataset_types = ["train", "valid", "test"]

    for dataset_type in dataset_types:
        input_images_dir = f"DataSet/{dataset_type}/images"
        input_labels_dir = f"DataSet/{dataset_type}/labels"
        output_dir = "output"

        if not os.path.exists(input_images_dir) or not os.path.exists(input_labels_dir):
            print(f"Skipping {dataset_type} dataset. Missing required directories.")
            continue

        print(f"Processing {dataset_type} dataset...")
        processor = YOLOProcessor(input_images_dir, input_labels_dir, output_dir, dataset_type)
        processor.process_images_and_labels()


if __name__ == "__main__":
    main()

import io
from fastapi import UploadFile
import torch
from PIL import Image
from torchvision import transforms
from Backend.Model.TrafficSignClassifier import TrafficSignClassifier

classes =['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def load_model(model_class, model_path, num_classes, device):
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.to(device)
    model.eval() 
    return model

def inference_imagePath(model, image_path,device ,transform=transform):
    image = Image.open(image_path).convert("RGB").resize((30, 30))
    image = transform(image).unsqueeze(0).to(device) 

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

def inference(model, file: UploadFile, device, transform=transform):
    file_content = file.file.read()
    image = Image.open(io.BytesIO(file_content)).convert("RGB").resize((30, 30))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    print("hay l proba",confidence.item())
    return classes[predicted_class.item()],confidence.item()

def main():
    num_classes = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/home/bibolil/Documents/GitHub/TrafficSignsProject/Backend/Model/traffic_sign_classifier.pth"

    model = load_model(TrafficSignClassifier, model_path, num_classes, device)
    print("Model loaded successfully!")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_image_path="Dataset/output/test/images/10a18f3d-79d7-447f-ab9e-6de78b09477f_jpg.rf.8ce6805731f28537f453028a8d5a4885_0.jpg"

    predicted_class = inference_imagePath(model, test_image_path, transform, device)
    print(f"Predicted class: {predicted_class}")
    print(f"predicted class label: {classes[predicted_class]}")

if __name__== "__main__":
    main()
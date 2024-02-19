import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ScorePredictorCNN(nn.Module):
    def __init__(self):
        super(ScorePredictorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class PhotoPredictorDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.image_names = os.listdir(images_folder)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

def predict_scores(model_path, images_folder, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScorePredictorCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = PhotoPredictorDataset(images_folder=images_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with open(output_file, 'w') as f:
        with torch.no_grad():
            for images, img_names in dataloader:
                images = images.to(device)
                outputs = model(images)
                scores = outputs.cpu().numpy().flatten()
                for img_name, score in zip(img_names, scores):
                    f.write(f"{img_name} {score}\n")

# Paths
model_path = "score_predictor_model.pth"
images_folder = "new_photos"
output_file = "predicted_scores.txt"

# Predict scores and save to file
predict_scores(model_path, images_folder, output_file)
print("Prediction completed. Scores are saved in 'predicted_scores.txt'.")


def sort_predicted_scores(input_file, output_file=None):
    # Use the same file for input and output if output_file is not specified
    if output_file is None:
        output_file = input_file
    
    # Read the predicted scores and their corresponding image names
    with open(input_file, 'r') as file:
        lines = file.readlines()
        score_entries = [line.strip().split() for line in lines]
    
    # Convert scores to float for sorting
    score_entries = [(entry[0], float(entry[1])) for entry in score_entries]
    
    # Sort entries based on scores in descending order
    score_entries.sort(key=lambda x: x[1], reverse=True)
    
    # Write the sorted entries back to a file
    with open(output_file, 'w') as file:
        for img_name, score in score_entries:
            file.write(f"{img_name} {score:.3f}\n")

# Example usage
input_file = "predicted_scores.txt"
output_file = "sorted_predicted_scores.txt"  # Optional: specify if you want to write to a new file
sort_predicted_scores(input_file, output_file)

print(f"Entries in '{input_file}' have been sorted and saved to '{output_file}'.")

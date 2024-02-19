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

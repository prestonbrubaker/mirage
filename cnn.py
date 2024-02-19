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


class PhotoScoreDataset(Dataset):
    def __init__(self, images_folder, scores_file, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.scores = {}
        with open(scores_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                self.scores[parts[0]] = float(parts[1])
        self.image_names = list(self.scores.keys())
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        score = self.scores[img_name]
        if self.transform:
            image = self.transform(image)
        score = torch.tensor(score, dtype=torch.float)
        return image, score

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = PhotoScoreDataset(images_folder="photos", scores_file="scores.txt", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using " + str(device))

model = ScorePredictorCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 20  # Adjust as needed

def evaluate_test_set(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients
        for images, scores in dataloader:
            images, scores = images.to(device), scores.to(device).view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, scores)
            total_loss += loss.item()
    return total_loss / len(dataloader)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, scores in train_dataloader:
        images, scores = images.to(device), scores.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    test_loss = evaluate_test_set(model, test_dataloader, criterion, device)
    
    print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_dataloader)}, Test MSE: {test_loss}")
    with open('model_history.txt', 'a') as file:
        file.write(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_dataloader)}, Test MSE: {test_loss} \n")

model_path = "score_predictor_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

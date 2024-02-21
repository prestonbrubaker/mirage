import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os

# Model Parameters
latent_dim = 256  # Example latent space dimension
LATENT_DIM = latent_dim



class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

       # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # Output: 16x128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 256x16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer input
            nn.Linear(128*16*16, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 128*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),  # Unflatten to 256x16x16 for conv transpose input
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: 32x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Output: 1x256x256
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class VAEScorePredictor(VariationalAutoencoder):
    def __init__(self, latent_dim):
        super(VAEScorePredictor, self).__init__(latent_dim)
        
        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),  # Input from concatenated mu and log_var
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a value between 0 and 1
        )
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        
        # Concatenate mu and log_var for prediction
        pred_input = torch.cat((mu, log_var), dim=1)
        score_pred = self.predictor(pred_input)
        
        return recon_x, mu, log_var, score_pred

# Function to load the model
def load_model(path, device):
    model = VariationalAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(path))
    return model

class ScoreDataset(Dataset):
    def __init__(self, folder_path, scores_file, transform=None):
        self.transform = transform
        self.images_folder = folder_path
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
        if self.transform:
            image = self.transform(image)
        score = torch.tensor([self.scores[img_name]], dtype=torch.float)
        return image, score



def test_predictor(model, dataloader, device):
    model.eval()
    test_loss = 0.0
    prediction_loss = nn.MSELoss()
    
    with torch.no_grad():
        for images, scores in dataloader:
            images, scores = images.to(device), scores.to(device)
            _, mu, log_var, score_pred = model(images)
            loss = prediction_loss(score_pred, scores)
            test_loss += loss.item()
    
    return test_loss / len(dataloader)



def train_predictor(model, train_dataloader, test_dataloader, optimizer, device, num_epochs):
    prediction_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, scores in train_dataloader:
            images, scores = images.to(device), scores.to(device)
            _, mu, log_var, score_pred = model(images)
            loss = prediction_loss(score_pred, scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        test_mse = test_predictor(model, test_dataloader, device)
        print(f'Epoch {epoch+1}, Train MSE Loss: {avg_loss:.6f}, Test MSE: {test_mse:.6f}')
        with open('latent_model_history.txt', 'a') as file:
            file.write(f'Epoch {epoch+1}, Train MSE Loss: {avg_loss:.6f}, Test MSE: {test_mse:.6f} \n')

        if(epoch % 25 == 0):
            save_path = 'latent_to_scores.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = ScoreDataset(folder_path='photos', scores_file='scores.txt', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model = VAEScorePredictor(LATENT_DIM).to(device)
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.decoder.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.predictor.parameters(), lr=0.01)

train_predictor(model, train_dataloader, test_dataloader, optimizer, device, num_epochs=1000)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np

class SolarPanelDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, delimiter=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(self.root_dir + img_name).convert('L')  

        # Convert grayscale image to RGB
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        crack_label = self.data.iloc[idx, 1]
        inactive_label = self.data.iloc[idx, 2]

        return image, crack_label, inactive_label
    
from torch.nn import BCEWithLogitsLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()
        return focal_loss
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 30
batch_size = 16
learning_rate = 0.001
random_state = 42
val_size = 0.2


csv_file = '/home/woody/iwso/iwso092h/dl/data.csv'
root_dir = '/home/woody/iwso/iwso092h/dl/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.5969,), (0.1143,))
])

dataset = SolarPanelDataset(csv_file, root_dir, transform=transform)

train_dataset, val_dataset = train_test_split(dataset, test_size=val_size, random_state=random_state)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

crack_labels = [label for _, label, _ in dataset]
class_counts = np.bincount(crack_labels)
num_samples = len(dataset)
class_weights = 1.0 / (class_counts / num_samples)
class_weights = torch.Tensor(class_weights).to(device)

num_classes = 2  
model = models.vgg19(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[0].in_features

fc_layer = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

model.classifier = fc_layer

model = model.to(device)

gamma = 2  # Focal Loss hyperparameter
criterion = FocalLoss(gamma, alpha=0.5)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (images, crack_labels, _) in enumerate(train_dataloader):
        images = images.to(device)
        crack_labels = crack_labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, crack_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, crack_labels, _ in val_dataloader:
            images = images.to(device)
            crack_labels = crack_labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, crack_labels)

            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}")

    lr_scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"/home/woody/iwso/iwso092h/dl/model_op/model_epoch_{epoch + 1}.pth")
        
losses_df = pd.DataFrame({'Train Loss': train_losses, 'Val Loss': val_losses})
losses_df.to_csv('/home/woody/iwso/iwso092h/dl/model_op/losses.csv', index=False)
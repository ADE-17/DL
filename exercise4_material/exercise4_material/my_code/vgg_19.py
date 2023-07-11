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
        if self.transform:
            image = self.transform(image)

        crack_label = self.data.iloc[idx, 1]
        inactive_label = self.data.iloc[idx, 2]

        return image, crack_label, inactive_label
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 50
batch_size = 16
learning_rate = 0.001
random_state = 42
val_size = 0.2


csv_file = '/home/woody/iwso/iwso092h/dl/data.csv'
root_dir = '/home/woody/iwso/iwso092h/dl/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),  # Random rotation (augmentation)
    transforms.RandomHorizontalFlip(),  # Random horizontal flip (augmentation)
    transforms.ToTensor(),
])

dataset = SolarPanelDataset(csv_file, root_dir, transform=transform)

# Split the dataset into train and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=val_size, random_state=random_state)

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Calculate class weights
crack_labels = [label for _, label, _ in dataset]
class_counts = np.bincount(crack_labels)
num_samples = len(dataset)
class_weights = 1.0 / (class_counts / num_samples)
class_weights = torch.Tensor(class_weights).to(device)

# Create the VGG19 model
num_classes = 2  # Number of classes: crack and no crack
model = models.vgg19(pretrained=True)

# Freeze all the parameters in the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer and add additional layers
in_features = model.classifier[6].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Dropout regularization
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),  # Dropout regularization
    nn.Linear(128, num_classes)
)

# Add batch normalization layers
model.features.add_module("BatchNorm", nn.BatchNorm2d(64))

# Move the model to the device
model = model.to(device)

# Define the loss function (Focal Loss) and optimizer
gamma = 2  # Focal Loss hyperparameter
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a learning rate scheduler
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Reduce learning rate every 3 epochs

# Train the model
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (images, crack_labels, _) in enumerate(train_dataloader):
        images = images.to(device)
        crack_labels = crack_labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, crack_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    # Validate the model
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, crack_labels, _ in val_dataloader:
            images = images.to(device)
            crack_labels = crack_labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, crack_labels)

            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Adjust learning rate
    lr_scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"/home/woody/iwso/iwso092h/dl/model_op/model_epoch_{epoch + 1}.pth")
        
# Save train and val losses to a CSV file
losses_df = pd.DataFrame({'Train Loss': train_losses, 'Val Loss': val_losses})
losses_df.to_csv('/home/woody/iwso/iwso092h/dl/model_op/losses.csv', index=False)
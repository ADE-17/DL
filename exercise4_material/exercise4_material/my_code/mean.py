import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class SolarPanelDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, delimiter=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(self.root_dir + img_name).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image

csv_file = '/home/woody/iwso/iwso092h/dl/data.csv'
root_dir = '/home/woody/iwso/iwso092h/dl/'

dataset = SolarPanelDataset(csv_file, root_dir, transform=ToTensor())

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

mean = torch.zeros(1)
std = torch.zeros(1)
num_samples = 0

count = 0
for images in dataloader:
    count = count + 1
    print(count)
    batch_size = images.size(0)
    images = images.view(batch_size, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    num_samples += batch_size

mean /= num_samples
std /= num_samples

print(f"Mean: {mean}")
print(f"Std: {std}")